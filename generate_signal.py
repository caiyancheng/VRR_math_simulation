import numpy as np
import matplotlib.pyplot as plt

def x2y(time_index, frame_time, persistence, peak_value):
    if time_index % frame_time < persistence * frame_time:
        return peak_value[0]
    else:
        return peak_value[1]

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
    # cycle_duration = sum(time_list)
    x_list = []
    y_list = []
    r_list = []
    last_time = 0
    last_luminance = 1
    Prepare_to_break = False
    if force_begin_end_equal:
        Time_range = sum(time_list) + 1
    else:
        Time_range = sum(time_list)
    for time_index in np.arange(0, Time_range, 1 / sampling_rate):
        x_list.append(time_index)
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
        elif force_begin_end_equal:
            refresh_rate = refresh_rate_list[2]
            Prepare_to_break = True
        else:
            break
        r_list.append(refresh_rate)
        now_luminance = x2y(time_index=time_index - last_time,
                            frame_time=1 / refresh_rate,
                            persistence=persistence,
                            peak_value=peak_value)
        if last_luminance == 0 and now_luminance == 1:
            last_time = time_index
        last_luminance = now_luminance
        y_list.append(now_luminance)
        if force_begin_end_equal and Prepare_to_break:
            if now_luminance == y_list[0]:
                break
    cycle_duration = time_index
    single_cycle_x = np.array(x_list)
    single_cycle_y = np.array(y_list)
    single_cycle_r = np.array(r_list)
    full_x = np.concatenate([single_cycle_x + i * cycle_duration for i in range(repeat_times)])
    full_y = np.tile(single_cycle_y, repeat_times)
    full_r = np.tile(single_cycle_r, repeat_times)
    if force_begin_end_equal:
        full_y[-1] = full_y[0]

    return full_x, full_y, full_r

if __name__ == "__main__":
    refresh_rate_list = [30, 120, 30]
    # time_list = [0.1,0,0.1,0,0]
    time_list = [0, 0.1, 0, 0.1, 0]
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