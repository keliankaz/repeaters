from __future__ import annotations
import pandas as pd
import numpy as np
import os
from obspy.clients.fdsn import Client
import warnings
from pathlib import Path
from catalog import Catalog
from typing import Optional, Union

base_dir = Path(__file__).parents[1]

class EarthquakeCatalog(Catalog):
    def __init__(
        self,
        catalog: Optional[pd.DataFrame] = None,
        filename: Optional[Union[str, Path]] = None,
        use_other_catalog: bool = False,
        kwargs: Optional[dict] = None,
        other_catalog: Optional[Catalog] = None,
        other_catalog_buffer: float = 0.0,
        reload: bool = False,
    ):
        if catalog is None:
            if kwargs is None:
                kwargs = {}

            if use_other_catalog and other_catalog is not None:
                metadata = {
                    "starttime": other_catalog.start_time,
                    "endtime": other_catalog.end_time,
                    "latitude_range": other_catalog.latitude_range
                    + np.array([-1, 1]) * other_catalog_buffer,
                    "longitude_range": other_catalog.longitude_range
                    + np.array([-1, 1]) * other_catalog_buffer,
                }
                metadata.update(kwargs)
            elif not use_other_catalog:
                metadata = kwargs
            else:
                raise ValueError("No other catalog provided")

            _catalog = self.get_and_save_catalog(filename, reload=reload, **metadata)
            self.catalog = self._add_time_column(_catalog, "time")
        else:
            self.catalog = catalog

        super().__init__(self.catalog)

        self._stress_drop = 3e6  # Pa

    @staticmethod
    def _add_time_column(df, column):
        """
        Adds a column to a dataframe.
        """
        df[column] = pd.to_datetime(df["time"], unit="d")
        return df

    @staticmethod
    def get_and_save_catalog(
        filename: Union[str, Path] = "_temp_local_catalog.csv",
        starttime: str = "2019-01-01",
        endtime: str = "2020-01-01",
        latitude_range: list[float] = [-90, 90],
        longitude_range: list[float] = [-180, 180],
        minimum_magnitude: float = 4.5,
        use_local_client: bool = False,
        default_client_name: str = "IRIS",
        reload: bool = True,
    ) -> pd.DataFrame:
        """
        Gets earthquake catalog for the specified region and minimum event
        magnitude and writes the catalog to a file.

        By default, events are retrieved from the NEIC PDE catalog for recent
        events and then the ISC catalog when it becomes available. These default
        results include only that catalog's "primary origin" and
        "primary magnitude" for each event.
        """

        if longitude_range[1] > 180:
            longitude_range[1] = 180
            warnings.warn("Longitude range exceeds 180 degrees. Setting to 180.")

        if longitude_range[0] < -180:
            longitude_range[0] = -180
            warnings.warn("Longitude range exceeds -180 degrees. Setting to -180.")

        if latitude_range[1] > 90:
            latitude_range[1] = 90
            warnings.warn("Latitude range exceeds 90 degrees. Setting to 90.")

        if latitude_range[0] < -90:
            latitude_range[0] = -90
            warnings.warn("Latitude range exceeds -90 degrees. Setting to -90.")

        def is_within(lat_range_querry, lon_range_querry, lat_range, lon_range):
            """
            Checks if a point is within a latitude and longitude range.
            """
            return (
                (lat_range[0] <= lat_range_querry[0] <= lat_range[1])
                and (lon_range[0] <= lon_range_querry[0] <= lon_range[1])
                and (lat_range[0] <= lat_range_querry[1] <= lat_range[1])
                and (lon_range[0] <= lon_range_querry[1] <= lon_range[1])
            )

        local_client_coverage = {
            "GEONET": [[-49.18, -32.28], [163.52, 179.99]],
        }

        # Note that using local client supersedes the any specified default_client_name
        if use_local_client:
            ## use local clients if lat and long are withing the coverage of the local catalogs
            index = []
            for i, key in enumerate(local_client_coverage.keys()):
                if is_within(
                    latitude_range, longitude_range, *local_client_coverage[key]
                ):
                    index.append(i)
                else:
                    index.append(i)

            if len(index) > 1:
                raise ValueError("Multiple local clients found")
            elif len(index) == 1:
                if default_client_name is not None:
                    warnings.warn("Using local client instead of default client")
                client_name = list(local_client_coverage.keys())[index[0]]
            else:
                client_name = default_client_name
        else:
            client_name = default_client_name

        querry = dict(
            starttime=starttime,
            endtime=endtime,
            minmagnitude=minimum_magnitude,
            minlatitude=latitude_range[0],
            maxlatitude=latitude_range[1],
            minlongitude=longitude_range[0],
            maxlongitude=longitude_range[1],
        )

        if not (
            reload is False
            and os.path.exists(filename)
            and np.load(
                os.path.splitext(filename)[0] + "_metadata.npy", allow_pickle=True
            ).item()
            == querry
        ):
            warnings.warn(f"Reloading {filename}")

            # Use obspy api to ge  events from the IRIS earthquake client
            client = Client(client_name)
            cat = client.get_events(**querry)

            # Write the earthquakes to a file
            f = open(filename, "w")
            f.write("time,lat,lon,depth,mag\n")
            for event in cat:
                loc = event.preferred_origin()
                lat = loc.latitude
                lon = loc.longitude
                dep = loc.depth
                time = loc.time.matplotlib_date
                mag = event.preferred_magnitude().mag
                f.write("{},{},{},{},{}\n".format(time, lat, lon, dep, mag))
            f.close()

            # Save querry to metadatafile
            np.save(os.path.splitext(filename)[0] + "_metadata.npy", querry)
        else:
            warnings.warn(f"Using existing {filename}")

        df = pd.read_csv(filename, na_values="None")

        # remove rows with NaN values, reset index and provide a warning is any rows were removed
        if df.isna().values.any():
            warnings.warn(
                f"{sum(sum(df.isna().values))} NaN values found in catalog. Removing rows with NaN values."
            )
            df = df.dropna()
            df = df.reset_index(drop=True)

        df.depth = df.depth / 1000  # convert depth from m to km

        return df
    
    def calculate_moment(self):
        """
        Calculates seismic moment for each earthquake in the catalog.
        Uses the formula: M0 = 10^(1.5 * Mw + 9.1)
        Where M0 is seismic moment in N·m and Mw is moment magnitude.
        """
        # Assuming 'mag' in the catalog is moment magnitude (Mw)
        self.catalog['moment'] = 10 ** (1.5 * self.catalog['mag'] + 9.1)
        return self.catalog['moment']

class WaldhauserEarthquakeCatalog(EarthquakeCatalog):
    
    def __init__(
        self, 
        filename: Optional[Union[str, Path]] = None,
    ):
        
        self.filename = filename
        self.catalog = self.read_catalog()
        
        super().__init__(catalog=self.catalog)
        
        
    def read_catalog(self):
        df = pd.read_csv(self.filename, skiprows=13)
        df["time"] = pd.to_datetime(df["DateTime"])
        df["mag"] = df["Magnitude"]
        df["lat"] = df["Latitude"]
        df["lon"] = df["Longitude"]
        df["depth"] = df["Depth"]
        
        return df