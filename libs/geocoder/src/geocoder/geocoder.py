from math import tan, atan, sin, cos, pi
from geopy import Nominatim
from geopy.location import Location
from typing import Union, Tuple
import numpy as np

class Geocoder:
    def __init__(self):
        self.g = Nominatim(user_agent="GetLoc")

    def gen_dists(self, start: float, stop: float, step: float, deg: float) -> Tuple[float, ...]:
        """Generates sequence of distances from camera from which the building may stand
        Returns the list of distances
        """

        res = []
        add = start
        while (add < stop):
            res.append(add)
            add += step
            step *= deg 
        return res

    def getAdress(self, lon: float, lat: float, ca: float, img_w: int, lbound: int, rbound: int, view_ang: float, step: int = 10, n_steps: int = 10) -> Tuple[Union[Location, None], Tuple[Union[float, None], Union[float, None]]]:
        """Calculate adress and coordinates image's building that fits within rbound and lbound pixels gate using image location and OSM geocoder 
        Returns the Location object (use the 'address' field to get an address string) and the tuple of latitude and longitude of the house in the bounds
        """
        
        ca = ca / 180 * pi
        view_ang = view_ang / 180 * pi

        center = (rbound + lbound) // 2
        
        add_ang = (center - img_w // 2) / img_w * view_ang

        azim = ca + add_ang
        dists = self.gen_dists(10, 1000, 25, 1.3)

        points_to_search = [(lat + dist * cos(azim) * 0.000009 / 1.7, lon + dist * sin(azim) * 0.000009)
                             for dist in dists]
        
        houses_found = []
        for p in points_to_search:
            try:
                adress = self.g.reverse(p, language='ru')
                if 'house_number' in adress.raw["address"].keys():
                    houses_found.append(adress)
            except:                                                                                       
                continue
        for hf in houses_found:
            p = (hf.latitude, hf.longitude)
            vec_to_building = np.array(points_to_search[0]) - np.array([lat, lon])
            vec_view = np.array(p) - np.array([lat, lon])
            vec_to_building[0], vec_view[0] = vec_to_building[0] * 1.7, vec_view[0] * 1.7
            if np.dot(vec_view, vec_to_building) / np.linalg.norm(vec_to_building) / np.linalg.norm(vec_view) > cos(view_ang / 2) / 2:
                return hf, p
        else:
            return None, (None, None)