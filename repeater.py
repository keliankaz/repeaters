import numpy as np
import copy
from catalog import Catalog
from earthquake import EarthquakeCatalog
import pandas as pd
from typing import Optional
from typing_extensions import Self
from matplotlib import pyplot as plt

class RepeaterCatalog(EarthquakeCatalog):
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
        _catalog["Datetime_2"] = pd.to_datetime(_catalog["Date_2"] + " " + _catalog["Time_2"])
        _catalog.drop(columns=["Date", "Time", "Date_2", "Time_2"], inplace=True)
        
        _catalog['time'] = _catalog['Datetime']
        _catalog['lat'] = _catalog['Lat_deg']
        _catalog['lon'] = _catalog['Lon_deg']
        _catalog['depth'] = _catalog['Depth_km']
        _catalog['mag'] = _catalog['Mag']
        
        self.min_time_delta_years = min_time_delta_years
        if self.min_time_delta_years is not None:
            _catalog = _catalog.loc[
                np.abs((_catalog["Datetime_2"] - _catalog["Datetime"])/np.timedelta64(1,'Y')) > self.min_time_delta_years
            ]
            _catalog = _catalog.reset_index(drop=True)
        
        super().__init__(_catalog)
        
        self._catalog = _catalog
        self._toggle = 0
        
    def create_surrogate_catalog( 
        self,
        other_catalog: Catalog,
        min_distance_km = 10, # avoid just selecting the same events or exact same geology
        max_distance_km = 300, # selected to preserve some of the same structure with Mc and similar tectonic setting        
        max_magnitude_delta = 0.1, # (event_pairs.catalog.Mag - event_pairs.catalog.Mag_2).abs().mean()  
        min_time_delta_days = 365, # avoid the events being related to eachother
        max_time_delta_days = 10*365, # to be more similar to the typical repeaters
        min_prior_time_delta = 60, # avoid mainshocks occuring before the event
        min_nearby_events = 4,
    ) -> Self:
        
        nearby_indices = self.get_neighboring_indices(other_catalog, buffer_radius_km=max_distance_km)
        exclude = self.get_neighboring_indices(other_catalog, buffer_radius_km=min_distance_km)
        
        nearby_indices = [list(set(good) - set(bad)) for good, bad in zip(nearby_indices, exclude)]
        
        event_info = []
        second_event_info = []
        
        assert min_nearby_events >= 2
        
        def _get_pair(t, M, M_ref, max_magnitude_delta, min_time_delta, max_time_delta, shadow_time):
            
            # Function to check shadow condition
            def is_in_shadow(idx, t, M, shadow_time):
                current_time = t[idx]
                current_mag = M[idx]
                shadow_indices = np.where((t < current_time) & (t >= current_time - shadow_time))[0]
                return any(M[shadow_indices] > current_mag)
            
            indices = np.arange(len(t))
            indices = indices[np.abs(M[indices] - M_ref)<max_magnitude_delta]
            indices = indices[np.max(t[indices]) - t[indices] > min_time_delta]          # cannot be at the very end of the catalgo
            indices = indices[[not is_in_shadow(i, t, M, shadow_time) for i in indices]] # cannot follow larger event
            
            np.random.shuffle(indices)
            
            # trial and error with shuffled pairs of indices to find the valid pair
            for i1 in indices:
                for i2 in indices:
                    if (abs(t[i2] - t[i1])  > min_time_delta) and (abs(t[i2] - t[i1])  < max_time_delta):
                        return i1, i2
                    
            return None, None  # only arrives here if no adequate indices were found
        
        # NOTE: if min_nearby_events is too small you run the risk of collision 
        # with the original dataset
        
        for nearby_indices_by_event, (_, row) in zip(nearby_indices, self.catalog.iterrows()):
                        
            candidate_events = other_catalog.catalog.iloc[nearby_indices_by_event]
                     
            I1, I2 = _get_pair( 
                candidate_events.time.values,
                candidate_events.mag.values, 
                M_ref = row.mag,
                max_magnitude_delta = max_magnitude_delta,
                min_time_delta=min_time_delta_days*np.timedelta64(1,'D'), 
                max_time_delta=max_time_delta_days*np.timedelta64(1,'D'), 
                shadow_time=min_prior_time_delta*np.timedelta64(1,'D'),
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
                dummy_second_event_catalog.reset_index(drop=True)
            ], axis=1
        )

        dummy_event_pairs.catalog = dummy_catalog
        dummy_event_pairs.catalog["Datetime"] = dummy_event_pairs.catalog["time"]
        dummy_event_pairs._catalog = copy.deepcopy(dummy_event_pairs.catalog)
        dummy_event_pairs.min_time_delta_years = min_time_delta_days/365
        
        return dummy_event_pairs
        
    def toggle_catalog(self, i=None, in_place=True):
        if i is not None:
            assert (i == 0) or (i == 1)
        if i is None:
            i = int(not i)
        
        if in_place:
             new = self  
        else:
            new = copy.deepcopy(self)
        
        if i==0:
            new.catalog = copy.deepcopy(new._catalog)
        if i==1:
            new.catalog = copy.deepcopy(new._catalog)
            new.catalog[["lat","lon","depth","mag","time"]] = new._catalog[["Lat_deg_2","Lon_deg_2","Depth_km_2","Mag_2","Datetime_2"]]
            new.catalog[["Lat_deg_2","Lon_deg_2","Depth_km_2","Mag_2","Datetime_2"]] = new._catalog[["lat","lon","depth","mag","time"]]
        new._toggle = i
        
        return new
    
    def plot_pairs(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            _, ax = plt.subplots(figsize=(4,10))
        
        event1_event2 = [[row.Datetime, row.Datetime_2] for _, row in self.catalog.iterrows()]
        rows = [[r,r] for r in range(len(self.catalog))]
        
        [ax.plot(t, r, marker='.', markersize=5, c='C0',lw=1, alpha=0.5) for t,r in zip(event1_event2, rows)]
        if self.min_time_delta_years is not None:
            event1_event2 = [[row.Datetime, row.Datetime + self.min_time_delta_years*np.timedelta64(1,'Y')] for _, row in self.catalog.iterrows()]
            [ax.plot(t, r, markersize=5, c='r',lw=1, alpha=0.5) for t,r in zip(event1_event2, rows)]
        
        ax.set(yticklabels=[], xlabel='Date')
    
class RepeaterSequences:
    def __init__(
        self, 
        query_paired_catalog, base_catalog,
        sequence_duration_days = 30,
        sequence_start_dt_days = 0,
        minimum_number_of_events = 5,
        radius_for_mc_determination = 100,
        radius_for_sequence = 20,
        apply_same_completness = True,
    ):
        
        self.query_catalog = query_paired_catalog
        
        self.base_catalog = base_catalog
        self.sequence_duration_days = sequence_duration_days * np.timedelta64(1,'D')
        self.sequence_start_dt_days = sequence_start_dt_days * np.timedelta64(1,'D')
        self.minimum_number_of_events = minimum_number_of_events
        self.radius_for_mc_determination = radius_for_mc_determination
        self.radius_for_sequence = radius_for_sequence
        
        self.sequences = self.get_combined_sequence_data()
        
        if apply_same_completness is True:
            self.apply_same_completeness()

    def apply_same_completeness(self, remove_short_sequence=True):
        
        indices_where_too_short = []
        for i,pair in enumerate(self.sequences):
            seq1, seq2 = [pair["catalog_1"], pair["catalog_2"]]
            mc = np.max([seq1.mag_completeness,seq2.mag_completeness])
            seq1.mag_completeness = mc
            seq2.mag_completeness = mc
            pair["catalog_1"] = seq1
            pair["catalog_2"] = seq2
            
            if (
                len(seq1) < self.minimum_number_of_events or 
                len(seq2) < self.minimum_number_of_events
            ):
                indices_where_too_short.append(i)
                
            self.sequences[i] = pair
        
        if remove_short_sequence is True:
            self.sequences = [v for i,v in enumerate(self.sequences) if i not in indices_where_too_short]
    
    @staticmethod
    def mc_max_curvature(mag, magnitude_increment=0.1, max_curvature_correction=0.1):
        
        bins = np.arange(np.min(mag), np.max(mag) + magnitude_increment,0.1) - magnitude_increment/2
        Mc = ((bins[:-1] + bins[1:])/2)[
            np.argmax(np.histogram(mag, bins=bins)[0])
        ] + max_curvature_correction
        
        return Mc
    
    def get_combined_sequence_data(self):

        sequences_1 = self.get_sequences(self.query_catalog.toggle_catalog(0, in_place=False), self.base_catalog)
        sequences_2 = self.get_sequences(self.query_catalog.toggle_catalog(1, in_place=False), self.base_catalog)
        
        combined_sequences = []
        for (_, row), c1, c2 in zip(self.query_catalog, sequences_1, sequences_2):
            
            if min([len(c1), len(c2)]) < self.minimum_number_of_events:
                continue
            
            paired_data = row.to_dict()
            paired_data.update(dict(
                catalog_1 = c1,
                catalog_2 = c2,
            ))
            
            combined_sequences.append(paired_data)

        return combined_sequences
        
    def get_sequences(self, query_catalog, base_catalog):
        """Returns a list containing the pairs of events""" 
        
        sequences = []
        
        local_Mc_catalog_indices = query_catalog.get_neighboring_indices(base_catalog, self.radius_for_mc_determination)
        local_sequence_catalog_indices = query_catalog.get_neighboring_indices(base_catalog, self.radius_for_sequence)
            
        
        for Imc, Isequence, (_, row) in zip( 
            local_Mc_catalog_indices, 
            local_sequence_catalog_indices, 
            query_catalog, 
        ):
            
            Mc_catalog = EarthquakeCatalog(base_catalog.catalog.iloc[Imc])
            Mc_catalog = Mc_catalog.get_time_slice(
                row.time + self.sequence_start_dt_days, row.time + self.sequence_start_dt_days + self.sequence_duration_days
            )
            
            if len(Mc_catalog) >= self.minimum_number_of_events:
                Mc = self.mc_max_curvature(Mc_catalog.catalog.mag.values)
            else:
                Mc = 0 # Bad solution but whatever
            
            local_catalog = EarthquakeCatalog(base_catalog.catalog.iloc[Isequence])
            local_catalog.mag_completeness = Mc
            assert np.all(local_catalog.catalog.mag.values >= Mc) # note setter truncated the magnitudes
            catalog = local_catalog.get_time_slice(
                row.time + self.sequence_start_dt_days, row.time + self.sequence_start_dt_days + self.sequence_duration_days
            )
            sequences.append(catalog)
            
        return sequences