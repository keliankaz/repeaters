#%% 
import numpy as np
import copy
import pandas as pd
from typing import Optional, Literal
from datetime import datetime, timedelta
from typing_extensions import Self
from matplotlib import pyplot as plt
import warnings
import re
from pathlib import Path

import itertools
import networkx as nx

from catalog import Catalog
from earthquake import EarthquakeCatalog

default_catalogs_dir = Path(__file__).parents[0] / "data"


class RepeaterCatalog(EarthquakeCatalog):
    def __init__(self):

        _catalog = self.load_catalog()
        super().__init__(catalog=_catalog)
        assert "family" in self.catalog.columns

    ## all repeater catalogs need the following
    def get_families(self) -> list[Catalog]:
        out = []
        for i in self.catalog.family.unique():
            family = self.get_categorical_slice("family", int(i))
            out.append(family)
        return out

    def load_catalog(self):
        return NotImplementedError

    def get_nodes(self):
        return NotImplementedError

    def get_edges(self):
        return NotImplementedError

    ## Useful operations for repeater catalogs
    def get_family(self, family_index):
        return self.get_categorical_slice("family", int(family_index))

    def create_graph(self):
        G = nx.Graph()
        G.add_nodes_from(self.get_nodes())
        G.add_edges_from(self.get_edges())
        return G

    def create_connected_graph(self):
        G = self.create_graph()
        components = list(nx.connected_components(G))
        for component in components:
            for u, v in itertools.combinations(component, 2):
                if not G.has_edge(u, v):
                    G.add_edge(u, v)

    ##  Useful plotting operations for repeater catalogs
    def plot_sequence(self, family_index, ax=None):
        repeater_sequence = self.get_family(family_index)
        ax = repeater_sequence.plot_time_series(ax=ax)
        return ax

    @staticmethod
    def COV(t, unbiased=True):
        if not isinstance(t,np.ndarray):
            t = (t - np.min(t))/np.timedelta64(1,'D')
        
        t = np.sort(t)
        
        dt = np.diff(t)
        
        bias_correction = (1 + 1/(4*len(dt))) if unbiased else 1 # https://en.wikipedia.org/wiki/Coefficient_of_variation
        
        return bias_correction * np.std(dt) / np.mean(dt)
    
    @staticmethod
    def characteristic_recurrence_interval(t, statistic: callable = np.median):
        if not isinstance(t,np.ndarray):
            t = (t - np.min(t))/np.timedelta64(1,'D')
            
        t = np.sort(t)
        dt = np.diff(t)
        
        return statistic(dt)

class WaldhauserRepeaterCatalog(RepeaterCatalog):
    filename = default_catalogs_dir / "waldhauser_shaff_repeaters.txt"

    def __init__(
        self,
    ):
        super().__init__()

    def load_catalog(self):

        data = []
        header = None
        with open(self.filename, "r") as file:
            for line in file:
                if line.startswith("#"):
                    # Parse header line
                    header_match = re.match(
                        r"#\s*(\d+)\s+([-.\dNaN]+)\s+([-.\dNaN]+)\s+([-.\dNaN]+)\s+([-.\dNaN]+)\s+([-.\dNaN]+)\s+([-.\dNaN]+)\s+([-.\dNaN]+)\s+([-.\dNaN]+)\s+([-.\dNaN]+)\s+([-.\dNaN]+)\s+([-.\dNaN]+)\s+([-.\dNaN]+)\s+(\S+)",
                        line,
                    )

                    if header_match:
                        header = header_match.groups()
                else:
                    # Parse event line
                    event_match = re.match(
                        r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-.\d]+)\s+(\d+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s+(\d+)",
                        line,
                    )
                    if event_match:
                        event = event_match.groups()
                        if header:
                            combined = header + event
                            data.append(combined)
                        else:
                            print(
                                f"No header found for event line: {line.strip()}"
                            )  # Debug print to catch cases with missing headers

        columns = [
            "NEV",
            "LATm",
            "LONm",
            "DEPm",
            "DMAGm",
            "DMAGs",
            "RCm",
            "RCs",
            "RCcv",
            "RCm1",
            "RCs1",
            "RCcv1",
            "CCm",
            "seqID",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "DAYS",
            "lat",
            "lon",
            "depth",
            "EX",
            "EY",
            "EZ",
            "mag",
            "DMAG",
            "DMAGE",
            "CCm_event",
            "evID",
        ]

        df = pd.DataFrame(data, columns=columns)

        # hack
        for col in columns:
            if col != "seqID":
                df[col] = df[col].apply(pd.to_numeric, errors="coerce")

        df["time"] = pd.to_datetime(
            df[["year", "month", "day", "hour", "minute", "second"]]
        )

        id_map = {seqID: i for i, seqID in enumerate(df.seqID.unique())}
        df["family"] = [id_map[seqID] for seqID in df.seqID]
        
        df['COV'] = df['RCcv']

        return df


class IgarashiRepeaterCatalog(RepeaterCatalog):
    filename = default_catalogs_dir / "igarashi_2020.txt"

    def __init__(
        self,
    ):
        super().__init__()

    def load_catalog(self):

        df = pd.read_csv(
            self.filename,
            sep="\s+",
            header=1,
            names=[
                "family",
                "date",
                "lat",
                "lon",
                "depth",
                "mag",
            ]
        )

        df['date_string'] = df['date'].astype(str)
        df['seconds'] = (df['date_string'].str[12:]).astype(float) * 100
        
        _date_string = df['date_string'].str[:12]
        
        # annoying 60 min
        annoying_bool = _date_string.str[-2:]=='60'
        
        _date_string[annoying_bool] = (
            _date_string[annoying_bool].str[:8] 
            + ( 
               _date_string[annoying_bool].str[8:10].astype(int)+1 
            ).astype(str) 
            + '00'
        )
        
        df['time'] = pd.to_datetime(_date_string,format='%Y%m%d%H%M') + df['seconds']*np.timedelta64(1,'s')
        
        # probably should make a method in repeaters for these:
        df['NEV'] = df.groupby("family")['time'].transform(len) 
        df = df.loc[df['NEV']>1]
        
        df['dt'] = df.groupby("family")['time'].transform('diff')/np.timedelta64(1,'D')
        df['COV'] = df.groupby("family")['dt'].transform(lambda dt: np.std(dt)/np.mean(dt))
        df['RCm'] = df.groupby("family")['dt'].transform(lambda dt: np.nanmedian(dt))
         

        return df

class TakaAkiRepeaterCatalog(RepeaterCatalog):
    """Notes about the catalog:
    
    - Anecdotally the catalog does not contain the exact set of events as e.g. the Waldhauser catalog or the Nadeau catalogs. 
    - Event locations are not explicitely relocated
    - The location of the last event is assigned to the whole family
    
    """
    
    def __init__(
        self,
        filename: Literal[
            "SJBPK.freq8-24Hz_maxdist3_coh9500_linkage_cluster.txt",
            "SJBPK.freq8-24Hz_maxdist3_coh9650_linkage_cluster.txt",
            "SJBPK.freq8-24Hz_maxdist3_coh9700_linkage_cluster.txt",
            "SJBPK.freq8-24Hz_maxdist3_coh9800_linkage_cluster.txt",
        ] = "SJBPK.freq8-24Hz_maxdist3_coh9500_linkage_cluster.txt",
        
        dirname: str = "CRE_TAKAAKI_2022",
    ):
        
        self.filename = default_catalogs_dir / dirname / filename
        super().__init__()
    
    @staticmethod    
    def _deciyear_to_datetime(deciyear):
        """
        Converts decimal year to datetime.
        """
        year = int(deciyear)
        rem = deciyear - year
        base = datetime(year, 1, 1)
        result = base + timedelta(
            seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem
        )
        return result

    def load_catalog(self):

        df = pd.read_csv(
            self.filename,
            sep="\s+",
            header=1,
            names=[
                "date",
                "lat",
                "lon",
                "depth",
                "cummulative_displacement_cm",
                "EVID",
                "CSID",
                "mag",
                "slip_cm",
            ]
        )

        df['time'] = [self._deciyear_to_datetime(idy) for idy in df["date"].values]
        
        # for simplicity and uniformity I apply interger family IDs to each family
        id_map = {CSID: i for i, CSID in enumerate(df.CSID.unique())} 
        df["family"] = [id_map[CSID] for CSID in df.CSID]
        
        df['NEV'] = df.groupby("family")['time'].transform(len)
        df = df.loc[df['NEV']>1]
        
        COV_map = {family_ID: self.COV(df.time.loc[df.family==family_ID]) for family_ID in df.family.unique()}
        df['COV'] = [COV_map[family_ID] for family_ID in df.family]
        
        RCm_map = {family_ID: self.characteristic_recurrence_interval(df.time.loc[df.family==family_ID]) for family_ID in df.family.unique()}
        df['RCm'] = [RCm_map[family_ID] for family_ID in df.family]
        
        return df

class SongRepeaterCatalog(EarthquakeCatalog):
    def __init__(
        self,
        file_name,
        min_time_delta_years=None,
    ):

        _catalog = pd.read_csv(
            filepath_or_buffer=file_name,
            delim_whitespace=True,
        )

        self.raw_catalog = _catalog

        _catalog["Datetime"] = pd.to_datetime(_catalog["Date"] + " " + _catalog["Time"])
        _catalog["Datetime_2"] = pd.to_datetime(
            _catalog["Date_2"] + " " + _catalog["Time_2"]
        )
        _catalog.drop(columns=["Date", "Time", "Date_2", "Time_2"], inplace=True)

        _catalog["time"] = _catalog["Datetime"]
        _catalog["lat"] = _catalog["Lat_deg"]
        _catalog["lon"] = _catalog["Lon_deg"]
        _catalog["depth"] = _catalog["Depth_km"]
        _catalog["mag"] = _catalog["Mag"]

        self.min_time_delta_years = min_time_delta_years
        if self.min_time_delta_years is not None:
            _catalog = _catalog.loc[
                np.abs(
                    (_catalog["Datetime_2"] - _catalog["Datetime"])
                    / np.timedelta64(1, "Y")
                )
                > self.min_time_delta_years
            ]
            _catalog = _catalog.reset_index(drop=True)

        super().__init__(_catalog)

        self._catalog = _catalog
        self._toggle = 0

    def get_split_ID(self):
        self.catalog["Doublet_ID"]
        return [list(s.plit("_")) for s in self.catalog["Doublet_ID"]]

    def get_nodes(self):
        ids = sum(self.split_ID(), [])  # flattens the ids
        return set(ids)

    def get_edges(self):
        return self.get_split_ID()

    def create_surrogate_catalog(
        self,
        other_catalog: Catalog,
        min_distance_km=10,  # avoid just selecting the same events or exact same geology
        max_distance_km=300,  # selected to preserve some of the same structure with Mc and similar tectonic setting
        max_magnitude_delta=0.1,  # (event_pairs.catalog.Mag - event_pairs.catalog.Mag_2).abs().mean()
        min_time_delta_days=365,  # avoid the events being related to eachother
        max_time_delta_days=10 * 365,  # to be more similar to the typical repeaters
        min_prior_time_delta=60,  # avoid mainshocks occuring before the event
        min_nearby_events=4,
    ) -> Self:

        nearby_indices = self.get_neighboring_indices(
            other_catalog, buffer_radius_km=max_distance_km
        )
        exclude = self.get_neighboring_indices(
            other_catalog, buffer_radius_km=min_distance_km
        )

        nearby_indices = [
            list(set(good) - set(bad)) for good, bad in zip(nearby_indices, exclude)
        ]

        event_info = []
        second_event_info = []

        assert min_nearby_events >= 2

        def _get_pair(
            t,
            M,
            M_ref,
            max_magnitude_delta,
            min_time_delta,
            max_time_delta,
            shadow_time,
        ):

            indices = np.arange(len(t))
            indices = indices[np.abs(M[indices] - M_ref) < max_magnitude_delta]
            indices = indices[
                np.max(t[indices]) - t[indices] > min_time_delta
            ]  # cannot be at the very end of the catalog
            indices = indices[
                [
                    not RepeaterCatalog.is_in_shadow(i, t, M, shadow_time)
                    for i in indices
                ]
            ]  # cannot follow larger event

            np.random.shuffle(indices)

            # trial and error with shuffled pairs of indices to find the valid pair
            for i1 in indices:
                for i2 in indices:
                    if (abs(t[i2] - t[i1]) > min_time_delta) and (
                        abs(t[i2] - t[i1]) < max_time_delta
                    ):
                        return i1, i2

            return None, None  # only arrives here if no adequate indices were found

        # NOTE: if min_nearby_events is too small you run the risk of collision
        # with the original dataset

        for nearby_indices_by_event, (_, row) in zip(
            nearby_indices, self.catalog.iterrows()
        ):

            candidate_events = other_catalog.catalog.iloc[nearby_indices_by_event]

            I1, I2 = _get_pair(
                candidate_events.time.values,
                candidate_events.mag.values,
                M_ref=row.mag,
                max_magnitude_delta=max_magnitude_delta,
                min_time_delta=min_time_delta_days * np.timedelta64(1, "D"),
                max_time_delta=max_time_delta_days * np.timedelta64(1, "D"),
                shadow_time=min_prior_time_delta * np.timedelta64(1, "D"),
            )

            if I1 and I2:
                if candidate_events.time.values[I1] > candidate_events.time.values[I2]:
                    I1, I2 = I2, I1
                event_info.append(candidate_events.iloc[I1])
                second_event_info.append(candidate_events.iloc[I2])

        dummy_event_pairs = copy.deepcopy(self)

        dummy_first_event_catalog = pd.DataFrame(event_info)

        dummy_second_event_catalog = pd.DataFrame(second_event_info)
        dummy_second_event_catalog.rename(
            columns={
                "depth": "Depth_km_2",
                "lat": "Lat_deg_2",
                "lon": "Lon_deg_2",
                "mag": "Mag_2",
                "time": "Datetime_2",
            },
            inplace=True,
        )

        dummy_catalog = pd.concat(
            [
                dummy_first_event_catalog.reset_index(drop=True),
                dummy_second_event_catalog.reset_index(drop=True),
            ],
            axis=1,
        )

        dummy_event_pairs.catalog = dummy_catalog
        dummy_event_pairs.catalog["Datetime"] = dummy_event_pairs.catalog["time"]
        dummy_event_pairs._catalog = copy.deepcopy(dummy_event_pairs.catalog)
        dummy_event_pairs.min_time_delta_years = min_time_delta_days / 365

        return dummy_event_pairs

    # Function to check shadow condition
    @staticmethod
    def is_in_shadow(idx, t, M, shadow_time):
        current_time = t[idx]
        current_mag = M[idx]
        shadow_indices = np.where(
            (t < current_time) & (t >= current_time - shadow_time)
        )[0]
        return any(M[shadow_indices] > current_mag)

    def toggle_catalog(self, i=None, in_place=True):
        if i is not None:
            assert (i == 0) or (i == 1)
        if i is None:
            i = int(not i)

        if in_place:
            new = self
        else:
            new = copy.deepcopy(self)

        if i == 0:
            new.catalog = copy.deepcopy(new._catalog)
        if i == 1:
            new.catalog = copy.deepcopy(new._catalog)
            new.catalog[["lat", "lon", "depth", "mag", "time"]] = new._catalog[
                ["Lat_deg_2", "Lon_deg_2", "Depth_km_2", "Mag_2", "Datetime_2"]
            ]
            new.catalog[
                ["Lat_deg_2", "Lon_deg_2", "Depth_km_2", "Mag_2", "Datetime_2"]
            ] = new._catalog[["lat", "lon", "depth", "mag", "time"]]
        new._toggle = i

        return new

    def plot_pairs(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 10))

        event1_event2 = [
            [row.Datetime, row.Datetime_2] for _, row in self.catalog.iterrows()
        ]
        rows = [[r, r] for r in range(len(self.catalog))]

        [
            ax.plot(t, r, marker=".", markersize=5, c="C0", lw=1, alpha=0.5)
            for t, r in zip(event1_event2, rows)
        ]
        if self.min_time_delta_years is not None:
            event1_event2 = [
                [
                    row.Datetime,
                    row.Datetime + self.min_time_delta_years * np.timedelta64(1, "Y"),
                ]
                for _, row in self.catalog.iterrows()
            ]
            [
                ax.plot(t, r, markersize=5, c="r", lw=1, alpha=0.5)
                for t, r in zip(event1_event2, rows)
            ]

        ax.set(yticklabels=[], xlabel="Date")


class RepeaterSequences:
    def __init__(
        self,
        query_paired_catalog,
        base_catalog,
        sequence_duration_days=30,
        sequence_start_dt_days=0,
        minimum_number_of_events=5,
        radius_for_mc_determination=100,
        radius_for_sequence=20,
        apply_same_completness=True,
    ):

        self.query_catalog = query_paired_catalog

        self.base_catalog = base_catalog
        self.sequence_duration_days = sequence_duration_days * np.timedelta64(1, "D")
        self.sequence_start_dt_days = sequence_start_dt_days * np.timedelta64(1, "D")
        self.minimum_number_of_events = minimum_number_of_events
        self.radius_for_mc_determination = radius_for_mc_determination
        self.radius_for_sequence = radius_for_sequence

        self.sequences = self.get_combined_sequence_data()

        if apply_same_completness is True:
            self.apply_same_completeness()

    def apply_same_completeness(self, remove_short_sequence=True):

        indices_where_too_short = []
        for i, pair in enumerate(self.sequences):
            seq1, seq2 = [pair["catalog_1"], pair["catalog_2"]]
            mc = np.max([seq1.mag_completeness, seq2.mag_completeness])
            seq1.mag_completeness = mc
            seq2.mag_completeness = mc
            pair["catalog_1"] = seq1
            pair["catalog_2"] = seq2

            if (
                len(seq1) < self.minimum_number_of_events
                or len(seq2) < self.minimum_number_of_events
            ):
                indices_where_too_short.append(i)

            self.sequences[i] = pair

        if remove_short_sequence is True:
            self.sequences = [
                v
                for i, v in enumerate(self.sequences)
                if i not in indices_where_too_short
            ]

    @staticmethod
    def mc_max_curvature(mag, magnitude_increment=0.1, max_curvature_correction=0.1):

        bins = (
            np.arange(np.min(mag), np.max(mag) + magnitude_increment, 0.1)
            - magnitude_increment / 2
        )
        Mc = ((bins[:-1] + bins[1:]) / 2)[
            np.argmax(np.histogram(mag, bins=bins)[0])
        ] + max_curvature_correction

        return Mc

    def get_combined_sequence_data(self):

        sequences_1 = self.get_sequences(
            self.query_catalog.toggle_catalog(0, in_place=True), self.base_catalog
        )

        sequences_2 = self.get_sequences(
            self.query_catalog.toggle_catalog(1, in_place=True), self.base_catalog
        )

        self.query_catalog.toggle_catalog(0, in_place=True)

        combined_sequences = []
        for (_, row), c1, c2 in zip(self.query_catalog, sequences_1, sequences_2):

            if min([len(c1), len(c2)]) < self.minimum_number_of_events:
                continue

            paired_data = row.to_dict()
            paired_data.update(
                dict(
                    catalog_1=c1,
                    catalog_2=c2,
                )
            )

            combined_sequences.append(paired_data)

        return combined_sequences

    def get_sequences(self, query_catalog, base_catalog):
        """Returns a list containing the pairs of events"""

        sequences = []

        local_Mc_catalog_indices = query_catalog.get_neighboring_indices(
            base_catalog, self.radius_for_mc_determination
        )
        local_sequence_catalog_indices = query_catalog.get_neighboring_indices(
            base_catalog, self.radius_for_sequence
        )

        for Imc, Isequence, (_, row) in zip(
            local_Mc_catalog_indices,
            local_sequence_catalog_indices,
            query_catalog,
        ):

            Mc_catalog = EarthquakeCatalog(base_catalog.catalog.iloc[Imc])
            Mc_catalog = Mc_catalog.get_time_slice(
                row.time + self.sequence_start_dt_days,
                row.time + self.sequence_start_dt_days + self.sequence_duration_days,
            )

            if (
                len(Mc_catalog) >= self.minimum_number_of_events
                and len(Mc_catalog) > 2  # minimum to make an Mc
            ):
                Mc = self.mc_max_curvature(Mc_catalog.catalog.mag.values)
            else:
                warnings.warn(
                    "Not enough events to determine the catalog completeness, setting Mc to 0"
                )

                Mc = 0  # Bad solution but whatever

            local_catalog = EarthquakeCatalog(base_catalog.catalog.iloc[Isequence])
            local_catalog.mag_completeness = Mc
            assert np.all(
                local_catalog.catalog.mag.values >= Mc
            )  # note setter truncated the magnitudes
            catalog = local_catalog.get_time_slice(
                row.time + self.sequence_start_dt_days,
                row.time + self.sequence_start_dt_days + self.sequence_duration_days,
            )

            sequences.append(catalog)

        return sequences
# %%

if __name__ == '__main__':
    
    repeaters = TakaAkiRepeaterCatalog()
    repeaters.catalog['number_of_events'] = repeaters.catalog.groupby("family")['date'].transform(len)
    repeaters = repeaters.slice_by('number_of_events',10)
    [repeaters.get_categorical_slice('family', int(i)).plot_time_series() for i in repeaters.catalog['family'].unique()[:10]]
    
# %%
