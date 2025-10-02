from math import tan, atan, sin, cos, pi
from geopy import Nominatim
from geopy.location import Location

class Geocoder:
    def __init__(self):
        self.g = Nominatim(user_agent="GetLoc")

    def getAdress(self, lon: float, lat: float, ca: float, img_w: int, rbound: int, lbound: int, view_ang: float, step: int = 10) -> tuple[Location | None, tuple[float | None, float | None]]:
        """Calculate adress and coordinates of building on image that fits within rbound and lbound pixel gate using image location and OSM geocoder 
        Returns the Location object (use the address field to get an address string) and the tuple of latitude and longitude of the house in the bounds
        """
        
        ca = ca / 180 * pi
        view_ang = view_ang / 180 * pi

        center = (rbound - lbound) // 2
        f = img_w // 2 / tan(view_ang / 2)
        add_ang = atan((center - img_w // 2) / f)

        azim = ca + add_ang

        points_to_search = [(lon + step * i * sin(azim) * 0.000009, 
                            lat + step * i * cos(azim) * 1.7 * 0.000009) for i in range(1, 11)]
        
        houses_found = []
        for p in points_to_search:
            try:
                adress = self.g.reverse(p, language='ru')
                adress_str = adress.address
                if 'house_number' in adress.raw.keys():
                    houses_found.append(adress)
            except:
                continue
        if len(houses_found) > 0:
            p = (houses_found[0].latitude, houses_found[0].longitude)
            return houses_found[0], p
        else:
            return None, (None, None)