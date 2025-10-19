"""Milvus Lite client helper for local vector search.

Provides a singleton MilvusClient bound to a local DB file and a thin
search wrapper returning raw hit dicts for downstream mapping.
"""

from __future__ import annotations

from typing import Any, Iterable
import threading

import numpy as np
from pymilvus import MilvusClient

_client_lock = threading.Lock()
_client: MilvusClient | None = None


def get_client(db_path: str) -> MilvusClient:
    """Return a shared MilvusClient for the given local DB path.

    Args:
        db_path: Filesystem path to Milvus Lite DB (e.g., ./moscow2019_MegaLoc.db).

    Returns:
        MilvusClient: Connected client instance.
    """

    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = MilvusClient(uri=db_path)
    return _client


def search_places(
    *,
    db_path: str,
    collection: str,
    vector_field: str,
    query_vec: np.ndarray,
    top_k: int,
    output_fields: Iterable[str] = ("place_id", "lat", "lon", "address", "image_uri"),
) -> list[dict[str, Any]]:
    """Search nearest neighbors for the given query vector.

    Args:
        db_path: Milvus Lite DB path.
        collection: Target collection name.
        vector_field: Vector field name storing embeddings.
        query_vec: Query embedding array of shape (1, D) float32 (L2-normalized).
        top_k: Number of results to return.
        output_fields: Scalar fields to return in entities.

    Returns:
        List of hit dicts with keys: 'id', 'distance', 'entity'.
    """

    client = get_client(db_path)
    vec = query_vec.reshape(-1).astype(np.float32).tolist()
    res = client.search(
        collection_name=collection,
        anns_field=vector_field,
        data=[vec],
        limit=int(top_k),
        output_fields=list(output_fields),
    )
    return res[0] if res else []
