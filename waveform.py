import numpy as np
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.geodetics.base import gps2dist_azimuth
from obspy.taup import TauPyModel

model = TauPyModel(model="iasp91")
client = Client("IRIS")

def get_global_inventory(
    start_time=UTCDateTime("1990-01-01"),
    network="IU,II,IC,GT",
):
    return client.get_stations(network=network, starttime=start_time)

def remove_instrument_response(st, pre_filt, inventory):
    st.remove_response(inventory, pre_filt=pre_filt, output="VEL")
    return st

def calculate_epicentral_distance(lat1, lon1, lat2, lon2):
    dist, _, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    return dist / 1000  # Convert from meters to kilometers

def iterate_stations(inventory):
    
    stations = []
    
    for network in inventory:
        for station in network:
            station.network = network
            stations.append(station)
    
    return stations
    
def get_stations_within_radius(self, lat, lon, radius_km):
        
    stations = self.iterate_stations(self.inventory)
    
    return [
        station for station in stations
        if self.calculate_epicentral_distance(
            station.latitude, station.longitude,
            lat, lon
        ) < radius_km
    ]

def process_waveforms(tr, trim_start_dt=0, trim_end_dt=None, pipeline=None, debug=True):
    
    if trim_end_dt is None:
        trim_end_dt = tr.stats.delta * tr.stats.npts  # set to duration of the trace
        
    t0 = tr.stats.starttime
    
    trim_start = t0 + trim_start_dt
    trim_end = t0 + trim_end_dt

    if pipeline is None:
        pipeline = {
            "resample":         dict( sampling_rate = 20),
            "detrend":          dict( type = "demean"),   
            "detrend":          dict( type = "linear"),  # noqa: F601
            "taper":            dict( max_percentage = 0.1),
            "remove_response":  dict( output="VEL", pre_filt=(0.001,0.002,10,20)),
            "taper":            dict( max_percentage=0.1),  # noqa: F601
            "filter":           dict( type="bandpass", freqmin=1/100, freqmax=1),
            "trim":             dict( starttime=trim_start, endtime=trim_end),
        }
        
    if debug:  # plot all intermediate steps
        tr.plot() 
    
    for k,v in pipeline.items():
        getattr(tr,k)(**v)
        if debug:  # plot all intermediate steps
            print(k)
            tr.plot() 
        
    return tr

def cross_correlate_waveforms(tr1,tr2):

    # Perform cross-correlation
    tr1 = tr1.normalize()
    tr2 = tr2.normalize()

    correlation = np.correlate(tr1.data, tr2.data, mode='same')
    
    lag = np.argmax(correlation)
    
    return lag, correlation[lag]