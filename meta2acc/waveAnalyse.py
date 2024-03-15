from lib.wave_analyse import *
from lib.make_data import *
import h5py
import numpy as np


quake_file = "Z:/STEAD/ch2/chunk2.csv"
h5py_file = "Z:/STEAD/ch2/chunk2.hdf5"
ev_list = get_quake_name(quake_file, "df.trace_name == '109C.TA_20060723155859_EV'")
dtfl = h5py.File(h5py_file, 'r')
quake_wave_set = dtfl.get('data/' + str(ev_list[0]))

client = Client("IRIS")
inventory = client.get_stations(network=quake_wave_set.attrs['network_code'],
                                station=quake_wave_set.attrs['receiver_code'],
                                starttime=UTCDateTime(quake_wave_set.attrs['trace_start_time']),
                                endtime=UTCDateTime(quake_wave_set.attrs['trace_start_time']) + 60,
                                loc="*",
                                channel="*",
                                level="response")

# converting into displacement
st = make_stream(quake_wave_set)
st = st.remove_response(inventory=inventory, output="ACC", plot=False)

# ploting the verical component
make_plot(st[0], title='Acceleration', ylab='m/s^2')

# quake_wave = np.array(quake_wave_set)
# vel, dpm = quake_params(quake_wave)
show_quake_wave(quake_wave_set)
