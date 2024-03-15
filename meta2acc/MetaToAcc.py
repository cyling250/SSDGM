"""
This code file should run independently, so the function in it is organized together.
"""
import time
import pandas as pd
import numpy as np
import h5py
import obspy
from obspy import UTCDateTime
from obspy.clients.fdsn.client import Client


def get_quake_name(csv_file, condition=""):
    """
    Founction of quake wave selecting.Only provide the 'trace_name' for selected results.
    :param file_name: The quake file's name.
    :param condition: The searching condition with type 'str'.
    :return: Selected list with 'trace_name'
    """
    # reading the csv file into a dataframe
    df = pd.read_csv(csv_file, low_memory=False)
    print(f"total events in csv file: {len(df)}")
    # filterering the dataframe, adding the condition
    if condition == "":
        df = df[df.trace_category == "earthquake_local"]
    else:
        df = df[((df.trace_category == "earthquake_local") & eval(condition))]
    print(f'total events selected: {len(df)}')

    # making a list of trace names for the selected data
    # the keyword must be 'trace_name' because it's primary
    ev_list = df['trace_name'].to_list()

    # the 'ev_list' just save the 'trace_name' data as list
    return ev_list


def make_stream(dataset):
    """
    Obspy is very
    input: hdf5 dataset
    output: obspy stream
    """
    data = np.array(dataset)

    tr_E = obspy.Trace(data=data[:, 0])
    tr_E.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_E.stats.delta = 0.01
    tr_E.stats.channel = dataset.attrs['receiver_type'] + 'E'
    tr_E.stats.station = dataset.attrs['receiver_code']
    tr_E.stats.network = dataset.attrs['network_code']

    tr_N = obspy.Trace(data=data[:, 1])
    tr_N.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_N.stats.delta = 0.01
    tr_N.stats.channel = dataset.attrs['receiver_type'] + 'N'
    tr_N.stats.station = dataset.attrs['receiver_code']
    tr_N.stats.network = dataset.attrs['network_code']

    tr_Z = obspy.Trace(data=data[:, 2])
    tr_Z.stats.starttime = UTCDateTime(dataset.attrs['trace_start_time'])
    tr_Z.stats.delta = 0.01
    tr_Z.stats.channel = dataset.attrs['receiver_type'] + 'Z'
    tr_Z.stats.station = dataset.attrs['receiver_code']
    tr_Z.stats.network = dataset.attrs['network_code']

    stream = obspy.Stream([tr_E, tr_N, tr_Z])

    return stream


def get_inventory_bulk(file_number):
    """
    Get inventory data from IRIS station and save as txt.
    """
    # Write file name of the meta DataSet Here
    quake_file = "Z:/STEAD/ch" + str(file_number) + "/chunk" + str(file_number) + ".csv"
    h5py_file = "Z:/STEAD/ch" + str(file_number) + "/chunk" + str(file_number) + ".hdf5"
    # Acc data saving dir
    chunk_acc = "Z:/STEAD/chunk" + str(file_number) + "_acc.hdf5"
    ev_list = get_quake_name(quake_file)
    dtfl = h5py.File(h5py_file, 'r')
    acc = h5py.File(chunk_acc, "a")
    length = len(ev_list)
    step = 1000

    for i in range(8000, length, step):
        bulk = []
        start_time = time.time()
        print("--------------------epoch %d--------------------" % (i / step))

        for j in range(step):
            # Get quake_set in each epoch
            print("\rSolving bulk: %d/%d" % (j + 1, step), end="")
            quake_wave_set = dtfl.get('data/' + str(ev_list[i + j]))
            bulk.append((quake_wave_set.attrs["network_code"], quake_wave_set.attrs["receiver_code"], "*", "*",
                         UTCDateTime(quake_wave_set.attrs["trace_start_time"]),
                         UTCDateTime(quake_wave_set.attrs["trace_start_time"]) + 60))

        print("\nSolved.")
        print("Getting station data from IRIS...")
        client = Client("IRIS")
        start_time_station = time.time()
        inventory_bulk = client.get_stations_bulk(bulk, level="response")
        print("Cost %.2f s to apply stations bulk from IRIS." % (time.time() - start_time_station))
        print("Data got.")

        for j in range(step):
            print("\rAnalysing and writing data: %d/%d" % (j + 1, step), end="")
            quake_wave_set = dtfl.get('data/' + str(ev_list[i + j]))
            st = make_stream(quake_wave_set)

            try:
                st = st.remove_response(inventory_bulk, output="ACC", plot=False)
                quake_acc = np.array(st)
                # In ev_list, it's the trace_name of data.
                acc.create_dataset(ev_list[i + j], data=quake_acc, dtype="float64")

            except Exception as e:
                with open("log.txt", "a") as fp:
                    now_time = time.ctime()
                    logmsg = str(now_time) + " " + str(e) + " epoch" + str(i / step) + "(" + str(j) + ")" + "\n"
                    print(logmsg)
                    fp.write(logmsg)

        print("\nOver.")
        end_time = time.time()
        print("Used time %.2f s." % (end_time - start_time))

    return None


if __name__ == "__main__":
    get_inventory_bulk(5)
