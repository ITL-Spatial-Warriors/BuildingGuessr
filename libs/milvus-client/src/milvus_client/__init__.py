"""Lightweight Milvus Lite client wrapper used by the API service.

This module exposes a thin convenience class around ``pymilvus.MilvusClient``
configured to use file-backed Milvus Lite for local development and tests.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

from pymilvus import MilvusClient, DataType


class MilvusLite:
    """Thin convenience wrapper around pymilvus MilvusClient for Milvus Lite.

    It initializes a file-backed Milvus Lite DB, ensures collection schema, and
    exposes minimal upsert/search helpers for the MVP.
    """

    def __init__(self, db_path: str | None = None) -> None:
        """Initialize the Milvus Lite client.

        Args:
            db_path: Optional path to the local Milvus Lite database file. If
                not provided, the value from the ``MILVUS_LITE_PATH``
                environment variable is used; otherwise defaults to
                ``./milvus-lite.db``.
        """
        # Allow env override, default to ./milvus-lite.db under current cwd
        db_file = db_path or os.getenv("MILVUS_LITE_PATH", os.path.abspath("milvus-lite.db"))
        # MilvusClient automatically selects Lite when given a local path/URI
        self.client = MilvusClient(uri=db_file)

    def ensure_places_collection(self, dim: int) -> None:
        """Create and load the ``Places`` collection if it does not exist.

        The collection contains a vector field named ``vec`` (with the given
        dimension) and scalar metadata fields (e.g., latitude/longitude,
        address, image URI, source, timestamp, extras JSON). A COSINE
        ``AUTOINDEX`` is created for the vector field.

        Args:
            dim: Embedding dimension for the ``vec`` field.

        Returns:
            None
        """
        name = "Places"
        if name not in self.client.list_collections():
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("place_id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("vec", DataType.FLOAT_VECTOR, dim=dim)
            schema.add_field("lat", DataType.FLOAT)
            schema.add_field("lon", DataType.FLOAT)
            schema.add_field("address", DataType.VARCHAR, max_length=512, nullable=True)
            schema.add_field("image_uri", DataType.VARCHAR, max_length=512)
            schema.add_field("source", DataType.VARCHAR, max_length=16)
            schema.add_field("ts", DataType.INT64)
            schema.add_field("extras", DataType.JSON)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vec",
                index_type="AUTOINDEX",
                metric_type="COSINE",
                index_name="vec_idx",
            )
            self.client.create_collection(name, schema=schema, index_params=index_params)

        state = self.client.get_load_state(collection_name=name)
        if state.get("state", "").endswith("NotLoad"):
            self.client.load_collection(name)

    def upsert_place(self, row: dict[str, Any]) -> None:
        """Upsert a single place entity into the ``Places`` collection.

        Args:
            row: Mapping of field names to values matching the ``Places``
                schema (e.g., ``place_id``, ``vec``, ``lat``, ``lon``,
                ``address``, ``image_uri``, ``source``, ``ts``, ``extras``).

        Returns:
            None
        """
        self.client.upsert(collection_name="Places", data=[row])

    def search_topk(
        self,
        query_vec: list[float],
        topk: int = 3,
        exclude_place_id: str | None = None,
        only_gallery: bool = True,
        output_fields: Iterable[str] = (
            "place_id",
            "lat",
            "lon",
            "address",
            "image_uri",
            "source",
            "ts",
            "extras",
        ),
    ) -> list[dict[str, Any]]:
        """Search top-k nearest ``Places`` for a given query vector.

        Performs a vector search over the ``vec`` field using COSINE metric and
        returns a list of dictionaries with selected fields and a derived
        ``score`` (``1 - distance``). Optionally excludes a specific
        ``place_id`` (e.g., the just-upserted query) and/or restricts results
        to gallery items (``source == "gallery"``).

        Args:
            query_vec: Query embedding vector.
            topk: Number of nearest results to return.
            exclude_place_id: Optional ``place_id`` to exclude from results.
            only_gallery: If ``True``, restrict results to entities with
                ``source == "gallery"``.
            output_fields: Names of fields to include in the result entities.

        Returns:
            A list of dictionaries describing the nearest candidates, including
            basic fields and an ``evidence`` block with raw distance.
        """
        filters = []
        if only_gallery:
            filters.append('source == "gallery"')
        if exclude_place_id:
            filters.append(f'place_id != "{exclude_place_id}"')
        expr = " AND ".join(filters) if filters else None

        res = self.client.search(
            collection_name="Places",
            data=[query_vec],
            anns_field="vec",
            limit=topk,
            search_params={"params": {}},
            output_fields=list(output_fields),
            consistency_level="Strong",
            filter=expr,
        )
        hits = res[0]
        out: list[dict[str, Any]] = []
        for h in hits:
            score = 1.0 - h["distance"]
            e = h["entity"]
            out.append(
                {
                    "place_id": e["place_id"],
                    "lat": e["lat"],
                    "lon": e["lon"],
                    "score": score,
                    "source": e["source"],
                    "evidence": {
                        "distance": h["distance"],
                        "gallery_image_uri": e["image_uri"],
                        "query_image_uri": None,
                        "bboxes_on_query": (e.get("extras") or {}).get("bboxes"),
                    },
                }
            )
        return out


__all__ = ["MilvusLite"]
