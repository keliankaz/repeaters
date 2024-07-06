# %%

from __future__ import annotations
import pandas as pd
from sklearn.neighbors import BallTree
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy
from cartopy import crs
import shapely
from typing import Optional, Literal, Union, Tuple
import copy
import warnings
from pathlib import Path

base_dir = Path(__file__).parents[1]

# constants of shame
EARTH_RADIUS_KM = 6371
DAY_PER_YEAR = 365
SEC_PER_DAY = 86400


def get_xyz_from_lonlat(
    lon: np.ndarray, lat: np.ndarray, depth_km: Optional[np.ndarray] = None
) -> np.ndarray:
    """Converts longitude, latitude, and depth to x, y, and z Cartesian
    coordinates.

    Args:
        lon: The longitude, in degrees.
        lat: The latitude, in degrees.
        depth_km: The depth, in kilometers.

    Returns:
        The Cartesian coordinates (x, y, z), in kilometers.
    """
    # Check the shapes of the input arrays
    if lon.shape != lat.shape:
        raise ValueError("lon and lat must have the same shape")

    assert -180 <= lon.all() <= 180, "Longitude must be between -180 and 180"
    assert -90 <= lat.all() <= 90, "Latitude must be between -90 and 90"
    assert depth_km is None or depth_km.all() >= 0, "Depth must be positive"

    # Assign zero depth if not provided:
    if depth_km is None:
        depth_km = np.zeros_like(lat)

    # Convert to radians
    lat_rad = lat * np.pi / 180
    lon_rad = lon * np.pi / 180

    # Calculate the distance from the center of the earth using the depth
    # and the radius of the earth (6371 km)
    r = EARTH_RADIUS_KM - depth_km

    # Calculate the x, y, z coordinates
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return np.array([x, y, z]).T


class Scaling:
    """A collection of scaling relationships for earthquakes"""

    @staticmethod
    def magnitude_to_size(
        MW: np.ndarray, stress_drop_Pa=3e6, out_unit: Literal["km", "m"] = "km"
    ) -> np.ndarray:
        # M0 = mu * A * D ~ \delta \sigma * a^3                   # TODO: MISSING CONSTANTS HERE!
        # Mw = (2/3) * (log10(M0) - 9.1)
        # a ~ [(1/(\delta \sigma)) 10^((3/2 * Mw) + 9.1)]^(1/3)   # TODO: CHECK THIS! e.g. dyne cm vs Pa
        # for SSEs \delta \sigma ~ 10 kPa
        # for Earthquake \delta \sigma ~ 3 MPa (default)
        # returns the dimensions of the earthquake in km

        unit_conversion_factor = {"km": 1 / 1000, "m": 1}

        return (10 ** ((3 / 2) * MW + 9.1) / stress_drop_Pa) ** (
            1 / 3
        ) * unit_conversion_factor[out_unit]



class Catalog:
    def __init__(
        self,
        catalog: pd.DataFrame,
        mag_completeness: Optional[float] = None,
        units: Optional[dict] = None,
    ):
        self.raw_catalog = (
            catalog.copy()
        )  # Save a copy of the raw catalog in case of regret

        _catalog = catalog.copy()
        self.catalog: pd.DataFrame = _catalog
        self.__mag_completeness = mag_completeness
        self.__mag_completeness_method = None

        self.units = {k: None for k in self.catalog.keys()}
        if units is not None:
            assert set(units.keys()).issubset(
                set(self.catalog.keys())
            ), "Invalid keys in units"
            self.units.update(units)

        self.__update__()

    def __update__(self):
        self.catalog = self.catalog.sort_values(by="time")

        # Save catalog attributes to self
        self.start_time = self.catalog["time"].min()
        self.end_time = self.catalog["time"].max()
        self.duration = self.end_time - self.start_time

        if "lat" in self.catalog.keys() and "lon" in self.catalog.keys():
            self.latitude_range = (self.catalog["lat"].min(), self.catalog["lat"].max())
            self.longitude_range = (
                self.catalog["lon"].min(),
                self.catalog["lon"].max(),
            )

        assert "time" in self.catalog.keys() is not None, "No time column"
        assert "mag" in self.catalog.keys() is not None, "No magnitude column"

        # check whether the catalog has locations (which is preferred)
        for key in ["lat", "lon", "depth"]:
            if key not in self.catalog.keys():
                warnings.warn(
                    f"Catalog does not have {key} column, this may cause errors."
                )

    @property
    def mag_completeness(
        self,
        magnitude_key: str = "mag",
        method: Literal["minimum", "maximum curvature"] = "minimum",
        filter_catalog: bool = True,
    ):
        if (
            self.__mag_completeness is None
            or self.__mag_completeness_method is not method
        ) and magnitude_key in self.catalog.keys():
            f = {
                "minimum": lambda M: min(M),
                "maximum curvature": lambda M: np.histogram(M)[1][
                    np.argmax(np.histogram(M)[0])
                ]
                + 0.2,
            }
            self.__mag_completeness = f[method](self.catalog[magnitude_key])
            self.__mag_completeness_method = method

            if filter_catalog:
                self.catalog = self.catalog[self.catalog.mag >= self.__mag_completeness]

        return self.__mag_completeness

    @mag_completeness.setter
    def mag_completeness(self, value):
        self.catalog = self.catalog[self.catalog.mag >= value]
        self.__mag_completeness = value

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, index: int) -> pd.Series:
        return self.catalog.iloc[index]

    def __getslice__(
        self,
        start: int,
        stop: int,
        step: Optional[int] = None,
    ) -> Catalog:
        new = copy.deepcopy(self)
        new.catalog = self.catalog[start:stop:step]
        new.__update__()

        return new

    def __iter__(self):
        return self.catalog.iterrows()

    def __add__(self, other) -> Catalog:
        combined_catalog = pd.concat(
            [self.catalog, other.catalog], ignore_index=True, sort=False
        )
        new = copy.deepcopy(self)
        new.catalog = combined_catalog
        new.__update__()

        return new

    def __radd__(self, other):
        return self.__add__(other)

    def slice_by(
        self,
        col_name: str,
        start=None,
        stop=None,
    ) -> Catalog:
        if start is None:
            start = self.catalog[col_name].min()
        if stop is None:
            stop = self.catalog[col_name].max()

        assert start <= stop
        in_range = (self.catalog[col_name] >= start) & (self.catalog[col_name] <= stop)

        new = copy.deepcopy(self)
        new.catalog = self.catalog.loc[in_range]
        new.__update__()

        return new

    def get_time_slice(self, start_time, end_time):
        return self.slice_by("time", start_time, end_time)

    def get_space_slice(self, latitude_range, longitude_range):
        return self.slice_by("lat", *latitude_range).slice_by("lon", *longitude_range)

    def get_polygon_slice(
        self,
        polygonal_boundary: np.ndarray,
    ) -> Catalog:
        polygon = shapely.geometry.Polygon(polygonal_boundary)

        new = copy.deepcopy(self)
        new.catalog = new.catalog[
            new.catalog.apply(
                lambda row: polygon.contains(shapely.geometry.Point(row.lon, row.lat)),
                axis=1,
            )
        ]

        return new

    def filter_duplicates(
        self,
        buffer_radius_km: float = 100,
        buffer_time_days: float = 30,
        stategy: Literal[
            "keep first", "keep last", "keep largest", "referece"
        ] = "keep first",
        ref_preference=None,
    ) -> Catalog:
        """returns a new catalog with duplicate events removed

        Checks for duplicates within `buffer_radius_km` and `buffer_time_seconds` of each other.

        """

        self.catalog.reset_index(drop=True, inplace=True)
        indices = self.intersection(
            self,
            buffer_radius_km=buffer_radius_km,
            buffer_time_days=buffer_time_days,
            return_indices=True,
        )[1]

        if ref_preference is not None:
            ranking = {i_ref: i for i, i_ref in enumerate(ref_preference)}

        indices_to_drop = []
        for i, neighbors in enumerate(indices):
            # Expects that all events in the catalog will have at least themselves as a neighbor
            if (
                len(neighbors) > 1
                and len(self.catalog.iloc[neighbors].ref.unique()) > 1
            ):  # don't drop if all events are the same reference
                if stategy == "keep first":
                    keep = self.catalog.iloc[neighbors].time.argmin()

                if stategy == "keep last":
                    keep = self.catalog.iloc[neighbors].time.argmax()

                if stategy == "keep largest":
                    keep = self.catalog.iloc[neighbors].mag.argmax()

                if stategy == "reference":
                    if ref_preference is None:
                        raise ValueError("Must specify reference event")
                    references = self.catalog.ref.iloc[neighbors]
                    rank = np.array([ranking[ref] for ref in references])
                    keep = rank.argmin()
                indices_to_drop.append(np.delete(neighbors, keep))

        new = copy.deepcopy(self)
        if len(indices_to_drop) > 0:
            indices_to_drop = np.unique(np.concatenate(indices_to_drop))
            new.catalog = new.catalog.drop(indices_to_drop)

        return new

    def intersection(
        self,
        other: Catalog,
        buffer_radius_km: float = 50.0,
        buffer_time_days=None,
        return_indices=False,
    ) -> Catalog:
        """returns a new catalog with the events within `buffer_radius_km`  and `buffer_time_days` of each other"""

        # For each event we get the indices of events in other that are within `buffer_radius_km` of it
        tree = BallTree(
            np.deg2rad(self.catalog[["lat", "lon"]].values),
            metric="haversine",
        )

        indices = tree.query_radius(
            np.deg2rad(other.catalog[["lat", "lon"]].values),
            r=buffer_radius_km / EARTH_RADIUS_KM,
            return_distance=False,
        )

        # For each event we get the indices of the events in other that are within `buffer_time_second` of it
        if buffer_time_days is not None:
            for i, t in enumerate(other.catalog.time.values):
                indices[i] = indices[i][
                    np.abs(self.catalog.time.values[indices[i]] - t)
                    / np.timedelta64(1, "D")
                    < buffer_time_days
                ]

        unique_indices = np.unique(np.concatenate(indices))

        new = copy.deepcopy(self)
        new.catalog = self.catalog.iloc[unique_indices]
        new.__update__()

        if return_indices:
            return new, indices
        else:
            return new

    def get_neighboring_indices(
        self,
        other: Catalog,
        buffer_radius_km: float = 50.0,
        return_distances: bool = False,
    ):
        """gets the indices of events in `other` that are within `buffer_radius_km` from self.

        The ouput therefore has dimensions [len(other),k] where k is the number of neighbors for each event.

        For instance:

        ```
        [other[indices] for indices in self.get_neighboring_indices(other)]
        ```

        Returns a list of catalogs for each neighborhood of events in self.

        If `return_distances` is True, then the output is a tuple of (indices, distances) where distances is a list of arrays of distances to each neighbor.

        """

        tree = BallTree(
            np.deg2rad(other.catalog[["lat", "lon"]]),
            metric="haversine",
        )
        if return_distances is True:
            I, R = tree.query_radius(
                np.deg2rad(self.catalog[["lat", "lon"]]),
                r=buffer_radius_km / EARTH_RADIUS_KM,
                return_distance=return_distances,
            )

            R *= EARTH_RADIUS_KM

            OUTPUT = (I, R)

        else:
            OUTPUT = tree.query_radius(
                np.deg2rad(self.catalog[["lat", "lon"]]),
                r=buffer_radius_km / EARTH_RADIUS_KM,
                return_distance=return_distances,
            )

        return OUTPUT

    def get_clusters(
        self,
        column: Union[list, str],
        number_of_clusters: int,
    ) -> list[Catalog]:
        if type(column) is str:
            assert column in self.catalog.columns
            X = np.atleast_2d(self.catalog[column].values).T
        elif type(column) is list:
            for col in column:
                assert col in self.catalog.columns
            X = self.catalog[column].values
        kmeans = KMeans(
            n_clusters=number_of_clusters,
        ).fit(X)

        subcatalogs = []
        for i in range(number_of_clusters):
            new = copy.deepcopy(self)
            new.catalog = self.catalog.loc[kmeans.labels_ == i]
            new.__update__()
            subcatalogs.append(new)

        idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
        subcatalogs = [subcatalogs[i] for i in idx]

        return subcatalogs

    def plot_time_series(
        self, column: str = "mag", type="scatter", ax=None
    ) -> plt.axes.Axes:
        """
        Plots a time series of a given column in a dataframe.
        """
        if ax is None:
            fig, ax = plt.subplots()

        if column == "mag" and self.mag_completeness is not None:
            bottom = self.mag_completeness - 0.05
        else:
            bottom = 0

        if type == "scatter":
            markers, stems, _ = ax.stem(
                self.catalog["time"],
                self.catalog[column],
                markerfmt=".",
                bottom=bottom,
            )
            plt.setp(stems, linewidth=0.5, alpha=0.5)
            plt.setp(markers, markersize=0.5, alpha=0.5)

        elif type == "hist":
            ax.hist(self.catalog["time"], bins=500)
            ax.set_yscale("log")

        ax.set_xlabel("Time")
        ax.set_ylabel(column)
        axb = ax.twinx()
        sns.ecdfplot(self.catalog["time"], c="C1", stat="count", ax=axb)

        return ax

    def plot_space_time_series(
        self,
        p1: list[float, float] = None,  # lon, lat
        p2: list[float, float] = None,  # lon, lat
        column: str = "mag",
        k_largest_events: Optional[int] = None,
        plot_histogram: bool = True,
        kwargs: dict = None,
        ax: Optional[plt.axes.Axes] = None,
    ) -> plt.axes.Axes:
        if ax is None:
            fig, ax = plt.subplots()

        if p1 is None and p2 is None:
            p1 = np.array([self.longitude_range[0], self.latitude_range[0]])
            p2 = np.array([self.longitude_range[1], self.latitude_range[1]])

        default_kwargs = {
            "alpha": 0.5,
            "color": "C0",
        }

        if kwargs is None:
            kwargs = {}
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

        p1, p2, x = [
            get_xyz_from_lonlat(np.atleast_2d(ll)[:, 0], np.atleast_2d(ll)[:, 1])
            for ll in [p1, p2, self.catalog[["lon", "lat"]].values]
        ]

        distance_along_section = np.matmul((x - p1), (p2 - p1).T) / np.linalg.norm(
            p2 - p1
        )

        marker_size = getattr(self.catalog, column) if isinstance(column, str) else 1

        ax.scatter(
            self.catalog.time,
            distance_along_section,
            **kwargs,
            s=marker_size,
        )

        if k_largest_events is not None:
            I = np.argsort(self.catalog[column].values)[-k_largest_events:]
            ax.scatter(
                self.catalog.time.values[I],
                distance_along_section[I],
                **dict(kwargs, marker="*", s=60),
            )

        if plot_histogram is True:
            # horizonta histogram of distance along section on the right side of the plot pointing left
            axb = ax.twiny()
            axb.hist(
                distance_along_section,
                orientation="horizontal",
                density=True,
                alpha=0.3,
            )

            axb.set(
                xlim=np.array(axb.get_xlim()[::-1]) * 10,
                xticks=[],
            )

        return ax

    def plot_depth_cross_section(
        self,
        p1: list[float, float] = None,
        p2: list[float, float] = None,
        width_km: float = None,
        column: str = "mag",
        k_largest_events: Optional[int] = None,
        plot_histogram: bool = True,
        kwargs: dict = None,
        ax: Optional[plt.axes.Axes] = None,
    ) -> plt.axes.Axes:
        for column_name in ["lon", "lat", "depth", column]:
            assert (
                column_name in self.catalog.columns
            ), f"column {column_name} not in catalog"  # TODO: make this assertion as part of the catalog class itself?

        if ax is None:
            fig, ax = plt.subplots()

        if p1 is None and p2 is None:
            p1 = np.array([self.longitude_range[0], self.latitude_range[0]])
            p2 = np.array([self.longitude_range[1], self.latitude_range[1]])

        default_kwargs = {
            "alpha": 0.5,
            "c": "C0",
            "s": getattr(self.catalog, column) if column in self.catalog.keys() else 1,
        }

        if kwargs is None:
            kwargs = {}
        default_kwargs.update(kwargs)
        kwargs = default_kwargs

        p1, p2, x = [
            get_xyz_from_lonlat(np.atleast_2d(ll)[:, 0], np.atleast_2d(ll)[:, 1])
            for ll in [p1, p2, self.catalog[["lon", "lat"]].values]
        ]

        distance_along_section = np.squeeze(
            np.matmul((x - p1), (p2 - p1).T) / np.linalg.norm(p2 - p1)
        )

        depth = self.catalog.depth.values

        if width_km is not None:
            distance_orthogonal_to_section = np.sqrt(
                np.linalg.norm(x - p1, axis=1) ** 2 - distance_along_section**2
            )
            index = distance_orthogonal_to_section < width_km
            if np.sum(index) == 0:
                warnings.warn(
                    "No data in the specified area, consider increasing width_km or checking if lat lon are correct"
                )
            distance_along_section = distance_along_section[index]
            depth = depth[index]
            if column in self.catalog.keys():
                default_kwargs["s"] = default_kwargs["s"][index]
            mag = self.catalog.mag[index].values

        sh = ax.scatter(
            distance_along_section,
            depth,
            **kwargs,
        )

        if k_largest_events is not None:
            I = np.argsort(mag)[-k_largest_events:]
            ax.scatter(
                distance_along_section[I],
                depth[I],
                **dict(kwargs, marker="*", s=60),
            )

        ax.set(
            ylabel="Depth (km)",
            xlabel="Distance along section (km)",
            ylim=(np.max(depth), 0),
        )

        if plot_histogram is True:
            # horizonta histogram of distance along section on the right side of the plot pointing left
            axb = ax.twiny()
            axb.hist(
                depth,
                orientation="horizontal",
                density=True,
                alpha=0.3,
                color=sh.get_facecolor(),
            )

            axb.set(
                xlim=np.array(axb.get_xlim()[::-1]) * 10,
                xticks=[],
            )

        return ax

    def plot_base_map(
        self,
        extent: Optional[np.ndarray] = None,
        usemap_proj = crs.PlateCarree(),
        ax=None,
    ) -> plt.axes.Axes:
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": usemap_proj})

        # set appropriate extents: (lon_min, lon_max, lat_min, lat_max)
        if extent is None:
            buffer = 1
            if self.longitude_range is None or self.latitude_range is None:
                extent = (
                    np.array(
                        [
                            self.catalog["lon"].min(),
                            self.catalog["lon"].max(),
                            self.catalog["lat"].min(),
                            self.catalog["lat"].max(),
                        ]
                    )
                    + np.array([-1, 1, -1, 1]) * buffer
                )

            else:
                extent = (
                    np.array(self.longitude_range + self.latitude_range)
                    + np.array([-1, 1, -1, 1]) * buffer
                )

        if extent[0] < -180:
            extent[0] = -179.99
        if extent[1] > 180:
            extent[1] = 179.99

        if extent[2] < -90:
            extent[2] = 90
        if extent[3] > 90:
            extent[3] = 90

        ax.set_extent(
            extent,
            crs=usemap_proj,
        )

        ax.add_feature(cartopy.feature.COASTLINE,lw=0.5, color="silver")
        ax.add_feature(cartopy.feature.LAND, color="whitesmoke")

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        return ax

    def plot_map(
        self,
        column: str = "mag",
        scatter_kwarg: Optional[dict] = None,
        k_largest_events: Optional[int] = None,
        extent: Optional[np.ndarray] = None,
        usemap_proj = crs.PlateCarree(),
        ax=None,
    ) -> plt.axes.Axes:
        ax = self.plot_base_map(extent=extent, usemap_proj = usemap_proj, ax=ax)

        if scatter_kwarg is None:
            scatter_kwarg = {}
        default_scatter_kawrg = {
            "s": self.catalog[column],
            "c": "lightgray",
            "marker": "o",
            "edgecolors": "brown",
            "transform": crs.PlateCarree(),
        }
        default_scatter_kawrg.update(scatter_kwarg)

        ax.scatter(
            self.catalog["lon"],
            self.catalog["lat"],
            **default_scatter_kawrg,
        )

        if k_largest_events is not None:
            I = np.argsort(self.catalog[column].to_numpy())[-k_largest_events:]
            ax.scatter(
                self.catalog["lon"].values[I],
                self.catalog["lat"].values[I],
                **dict(scatter_kwarg, marker="*", s=60),
            )

        return ax

    def plot_hist(
        self, columm: str = "mag", log_scale: bool = True, ax=None
    ) -> plt.axes.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(self.catalog[columm], log=log_scale)
        ax.set_xlabel(columm)

        return ax

    def plot_scaling(
        self, column: str = "duration", log_scale: bool = True, ax=None
    ) -> plt.axes.Axes:
        if ax is None:
            fig, ax = plt.subplots()
        assert self.catalog[column] is not None, "No duration column"
        ax.scatter(self.catalog["mag"], self.catalog[column])
        ax.set(
            xlabel="Magnitude",
            ylabel=column,
        )
        if log_scale:
            ax.set_yscale("log")

        return ax

    def plot_summary(
        self, kwarg={"time series": None, "map": None, "hist": None}, ax=None
    ) -> Tuple[plt.axes.Axes, plt.axes.Axes, plt.axes.Axes, plt.axes.Axes]:
        if ax is None:
            fig = plt.figure(figsize=(6.5, 7))
            gs = fig.add_gridspec(4, 3)
            ax1 = fig.add_subplot(gs[0:2, 0:2], projection=crs.PlateCarree())
            ax2 = fig.add_subplot(gs[0:2, 2])
            ax3 = fig.add_subplot(gs[2, :])
            ax4 = fig.add_subplot(gs[3, :])
        else:
            print("Bold decision")
            ax1, ax2, ax3, ax4 = ax

        self.plot_map(ax=ax1)
        self.plot_hist(ax=ax2)
        self.plot_time_series(ax=ax3)
        self.plot_space_time_series(ax=ax4)

        plt.tight_layout()

        return (ax1, ax2, ax3, ax4)

    def read_catalog(self, filename):
        """
        Reads a catalog from a file and returns a dataframe.
        """
        raise NotImplementedError


# %%
