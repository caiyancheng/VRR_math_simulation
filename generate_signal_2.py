# 相比1，2确保了一定会让某个帧率完成一个波
import numpy as np
import matplotlib.pyplot as plt

def x2y(time_index, frame_time, persistence, peak_value):
    if time_index % frame_time < persistence * frame_time:
        return peak_value[0]
    else:
        return peak_value[1]

def one_frame(begin_time, frame_rate, persistence, peak_value, sampling_rate):
    frame_time = 1 / frame_rate
    time_array = np.arange(begin_time, begin_time + frame_time, 1 / sampling_rate)
    frame_rate_array = np.full(time_array.shape, frame_rate)
    white_points = round(time_array.shape[0] * persistence)
    luminance_array = np.concatenate((np.full(white_points, peak_value[0]),
                                      np.full(time_array.shape[0] - white_points, peak_value[1])))
    time_index = time_array[-1] + 1 / sampling_rate
    return time_array, frame_rate_array, luminance_array, time_index

def signal_vrr(refresh_rate_list, time_list, persistence, peak_value,
               sampling_rate=1000, repeat_times=1, force_begin_end_equal=False):
    '''
    :param refresh_rate_list: A list of length 3 representing changes in frequency
    :param time_list: A list of length 5 representing the duration of each frequency
    :param persistence: The proportion of time the light is present in the entire frame, ranging from 0 to 1
    :param peak_value: A list of length 2 representing the maximum and minimum values of light
    :param sampling_rate: How many points exist per second
    :param repeat_time: Repeat How Many Times
    :param force_begin_end_equal: Force Complete One Frame and The Begin should equal to the End
    :return: x_array - time; y_array - luminance; r_array - refresh rate
    '''
    x_array = np.array([])
    l_array = np.array([])
    r_array = np.array([])
    Prepare_to_break = False
    time_index = 0
    while time_index < sum(time_list):
        if time_index < sum(time_list[:1]):
            refresh_rate = refresh_rate_list[0]
        elif time_index < sum(time_list[:2]):
            refresh_rate = refresh_rate_list[0] + (refresh_rate_list[1] - refresh_rate_list[0]) / time_list[1] * (time_index - time_list[0])
        elif time_index < sum(time_list[:3]):
            refresh_rate = refresh_rate_list[1]
        elif time_index < sum(time_list[:4]):
            refresh_rate = refresh_rate_list[1] + (refresh_rate_list[2] - refresh_rate_list[1]) / time_list[3] * (time_index - sum(time_list[:3]))
        elif time_index < sum(time_list[:5]):
            refresh_rate = refresh_rate_list[2]
        else:
            break
        time_array, frame_rate_array, luminance_array, time_index = one_frame(begin_time=time_index, frame_rate=refresh_rate,
                                                                              persistence=persistence, peak_value=peak_value,
                                                                              sampling_rate=sampling_rate)
        x_array = np.append(x_array, time_array)
        l_array = np.append(l_array, luminance_array)
        r_array = np.append(r_array, frame_rate_array)

    cycle_duration = time_index - 1 / sampling_rate
    full_x = np.concatenate([x_array + i * cycle_duration for i in range(repeat_times)])
    full_l = np.tile(l_array, repeat_times)
    full_r = np.tile(r_array, repeat_times)
    if force_begin_end_equal:
        full_l[-1] = full_l[0]

    return full_x, full_l, full_r

if __name__ == "__main__":
    refresh_rate_list = [30, 120, 30]
    time_list = [0.1,0,0.1,0,0]
    # time_list = [0, 0.1, 0, 0.1, 0]
    persistence = 0.5
    peak_value = [1,0]
    sampling_rate = 1000
    repeat_times = 4
    x_array, y_array, r_array = signal_vrr(refresh_rate_list=refresh_rate_list,
                                  time_list=time_list,
                                  persistence=persistence,
                                  peak_value=peak_value,
                                  sampling_rate=sampling_rate,
                                  repeat_times=repeat_times,
                                  force_begin_end_equal=True)
    plt.figure(figsize=(10,4))

    plt.subplot(2, 1, 1)
    plt.plot(x_array, y_array)
    # plt.xlabel('Time (s)')
    plt.ylabel('Luminance ($cd / m^2$)')
    plt.grid(True)
    # plt.title('Luminance vs Time with VRR')

    plt.subplot(2, 1, 2)
    plt.plot(x_array, r_array)
    plt.xlabel('Time (s)')
    plt.ylabel('Refresh Rate (Hz')
    # plt.title('Luminance vs Time with VRR')
    plt.show()
    # plt.savefig(r'E:\Py_codes\VRR_math_simulation/try.png')