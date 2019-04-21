import pymysql
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy as np
import threading
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from scipy import signal
from datetime import datetime


def plot_graph(title, arr):
    plt.plot(arr, 'r', label=title)
    plt.legend()
    plt.show()


def split_accelerometer_string(accelerometer_data):
    x = []
    y = []
    z = []
    timestamp = []
    data = accelerometer_data.split(',')
    for j in range(0, len(data) - 1, 4):
        x.append(float(data[j]))
        y.append(float(data[j + 1]))
        z.append(float(data[j + 2]))
        timestamp.append(data[j + 3])

    return [x, y, z, timestamp]


def get_magnitude(x, y, z):
    magnitude = []
    for i in range(1, len(x) - 1, 1):
        magnitude.append((x[i] ** 2 + y[i] ** 2 + z[i] ** 2) ** 0.5)
    return magnitude


def filter_stream(magnitude, plot_before, plot_after):
    if plot_before:
        plot_graph("Before", magnitude)
    fs = len(magnitude)
    fc = fs * 0.03  # Cut-off frequency of the filter
    w = float((fc / (fs / 2.0)))  # Normalize the frequency
    b, a = signal.butter(5, w, 'low', analog=False)
    output = signal.filtfilt(b, a, magnitude)
    if plot_after:
        plot_graph("After", output)
    return output


def interpolate(arr, length):
    arr1_interp = interp.interp1d(np.arange(len(arr)), arr)
    arr1_interpolated = arr1_interp(np.linspace(0, len(arr) - 1, length))
    return arr1_interpolated


def get_cost_dtw(signal1, signal2):
    cost, path = fastdtw(signal1, signal2, dist=euclidean)
    return cost


def optimize_stream(testing_stream):
    # Split String
    stream_split = split_accelerometer_string(testing_stream)
    x, y, z, timestamps = stream_split[0], stream_split[1], stream_split[2], stream_split[3]

    # Get Magnitude
    magnitude = get_magnitude(x, y, z)

    # Filter Magnitude
    filtered_stream = filter_stream(magnitude, False, False)

    return filtered_stream, timestamps


def calculate_average(arr, points_average_length):
    averages = []
    for i in range(0, len(arr), points_average_length):
        values_sum = 0
        if i + points_average_length < len(arr):
            for j in range(i, i + points_average_length):
                values_sum = values_sum + arr[j]
            avg = values_sum / points_average_length
            averages.append(avg)
    return averages


def reformat_date(date):
    splitter = date.split(" ")
    date1 = '20' + splitter[0]
    date2 = splitter[1].split(":")[0] + ":" + splitter[1].split(":")[1] + ":" + splitter[1].split(":")[2] + "." + \
            splitter[1].split(":")[3]
    format_date = date1 + " " + date2
    return format_date


def get_difference_in_milli_seconds(date1, date2):
    date_time_format = '%Y-%m-%d %H:%M:%S.%f'
    date1 = reformat_date(date1)
    date2 = reformat_date(date2)
    diff = datetime.strptime(date1, date_time_format) - datetime.strptime(date2, date_time_format)
    return abs(diff.total_seconds() * 1000)


def cut_stream(filtered_stream, peak_threshold, points_average_length, timestamps):
    averages = calculate_average(filtered_stream, points_average_length)
    stream_cuts = []
    stream_durations = []
    i = 0
    end_stream_i = 0
    last_point = averages[0]
    while i in range(len(averages)):
        if averages[i] > peak_threshold or averages[i] - last_point >= 2.2:
            peak_index = i
            if i + 1 < len(averages):
                while averages[i + 1] > averages[i]:
                    peak_index = i + 1
                    if len(averages) == i + 2:
                        break
                    else:
                        i = i + 1
            start_stream_i = get_start_index(averages, peak_index, points_average_length)
            end_stream_i, end_average_i = get_end_index(averages, filtered_stream, peak_index, points_average_length)
            last_point = averages[end_average_i]

            window = cut_range(start_stream_i, end_stream_i, filtered_stream)
            time = get_difference_in_milli_seconds(timestamps[start_stream_i], timestamps[end_stream_i])
            stream_durations.append(time)
            stream_cuts.append(window)
            i = end_average_i + 1
        else:
            i = i + 1
    if end_stream_i == 0:
        stream_reminder = filtered_stream
        duration_reminder = timestamps
    else:
        stream_reminder = cut_range(end_stream_i, len(filtered_stream - 1), filtered_stream)
        duration_reminder = cut_range(end_stream_i, len(filtered_stream - 1), timestamps)
    return stream_cuts, stream_reminder, stream_durations, duration_reminder


def get_start_index(averages, peak_index, points_average_length):
    start_index_average = peak_index
    for j in range(peak_index, 0, -1):
        if averages[j - 1] < averages[j]:
            start_index_average = j - 1
        else:
            break

    start_index = start_index_average * points_average_length + points_average_length / 2
    return int(start_index)


def get_end_index(averages, stream, peak_index, points_average_length):
    end_index_average = peak_index
    for j in range(peak_index, len(averages)):
        if j + 1 < len(averages):
            if averages[j + 1] < averages[j]:
                end_index_average = j + 1
            else:
                break

    if end_index_average == peak_index:
        end_index_average = len(averages) - 1
        end_index = len(stream) - 1
        return end_index, end_index_average
    else:
        end_index = end_index_average * points_average_length + points_average_length / 2
        return int(end_index), end_index_average


def cut_range(start_window_index, end_window_index, output):
    window = []
    for i in range(start_window_index, end_window_index):
        window.append(output[i])
    return window


def update_stream_to_processed(stream_data_id):
    sql = "UPDATE `stream_data` SET`is_processed`= '1' WHERE `id`='" + str(stream_data_id) + "'"
    conn = pymysql.connect(host='localhost', user='root', password='', db='dtw_test8')
    link = conn.cursor()
    result = link.execute(sql)
    conn.commit()
    return result


def get_stream_data(stream_id):
    sql = "SELECT * FROM `stream_data` WHERE `stream_id` = '" + str(stream_id) + "' AND `is_processed` = 0"
    conn = pymysql.connect(host='localhost', user='root', password='', db='dtw_test8')
    link = conn.cursor()
    link.execute(sql)
    testing_streams = link.fetchall()
    stream = ""
    for i in range(len(testing_streams)):
        stream = stream + testing_streams[i][2]
        update_stream_to_processed(testing_streams[i][0])
    return stream


def submit_stream_cuts(stream_id, stream_cuts, stream_durations):
    sql = "INSERT INTO `result`(`stream_id`, `stroke_magnitudes`, `stroke_duration`, `stroke_result`) VALUES ('" + str(
        stream_id) + "','" + str(stream_cuts[0]) + "','" + str(stream_durations[0]) + "','Pending')"
    for i in range(1, len(stream_cuts)):
        sql = sql + " , ('" + str(stream_id) + "','" + str(stream_cuts[i]) + "','" + str(
            stream_durations[i]) + "','Pending')"
    conn = pymysql.connect(host='localhost', user='root', password='', db='dtw_test8')
    link = conn.cursor()
    result = link.execute(sql)
    conn.commit()
    return result

def submit_stroke_result(result, stream):
    stroke_types=["Correct Stroke", "Wrong Hand Entry", "Wrong High Elbow", "Wrong Recovery", "unKnown"]
    print(stroke_types[result-1])
    sql = "UPDATE `result` SET `stroke_result`='"+str(stroke_types[result-1])+"' WHERE `stroke_magnitudes` = '"+str(stream)+"'"
    conn = pymysql.connect(host='localhost', user='root', password='', db='dtw_test8')
    link = conn.cursor()
    result = link.execute(sql)
    conn.commit()
    return result

def get_stroke_templates():
    sql = "SELECT * FROM `best_templates` WHERE 1"
    conn = pymysql.connect(host='localhost', user='root', password='', db='dtw_test8')
    link = conn.cursor()
    result2 = link.execute(sql)
    result2= link.fetchall()
    conn.commit()
    templates = []
    templates_classes = []
    print(result2)
    for i in range(len(result2)):
        templates.append(result2[i][3].split(','))
        templates[i] = [float(i) for i in templates[i][:-1]]
        templates_classes.append(result2[i][2])

    return templates, templates_classes


def classifiy_stroke(templates, test, templates_classes):
    costs = []
    for i in range(len(templates)):
        templates[i] = interpolate(templates[i], 250)
        test = interpolate(test, 250)
        cost = get_cost_dtw(templates[i], test)
        costs.append(cost)
    print(min(costs))
    if min(costs) >= 1000:
        return 5
    return templates_classes[np.argmin(costs)]


def engine(stream_reminder, total_streams, stream_id, duration_reminder, templates, templates_classes):
    stream = get_stream_data(stream_id)
    if stream != "":
        optimized_stream, timestamps = optimize_stream(stream)
        if len(stream_reminder) != 0:
            optimized_stream = np.concatenate((stream_reminder, optimized_stream), axis=None)
            timestamps = np.concatenate((duration_reminder, timestamps), axis=None)
        stream_cuts, stream_reminder, stream_durations, duration_reminder = cut_stream(optimized_stream, 13, 125,
                                                                                       timestamps)
        if len(stream_cuts) > 0:
            submit_stream_cuts(stream_id, stream_cuts, stream_durations)
            total_streams = total_streams + stream_cuts
            print(len(total_streams))
            for i in range(len(stream_cuts)):
                stream_cut_result = classifiy_stroke(templates,stream_cuts[i], templates_classes)
                submit_stroke_result(stream_cut_result,stream_cuts[i])


    print(len(total_streams))
    threading.Timer(2.0, engine, [stream_reminder, total_streams, stream_id, duration_reminder, templates, templates_classes]).start()


reminder_stream = []
streams_total = []
reminder_duration = []

templates, templates_classes = get_stroke_templates()
for i in range(len(templates)):
    templates[i] = filter_stream(templates[i], False, False)


sql = "SELECT MAX(`id`) FROM `stream`"
conn = pymysql.connect(host='localhost', user='root', password='', db='dtw_test8')
link = conn.cursor()
maxid = link.execute(sql)
maxid= link.fetchall()
conn.commit()
engine(reminder_stream, streams_total, maxid[0][0], reminder_duration,templates, templates_classes)