import matplotlib.pyplot as plt
import numpy as np
from TCSF.TCSF import TCSF
from generate_signal_2 import signal_vrr
from mpl_toolkits.mplot3d import Axes3D


def Calculte_JND_from_persistence(phase_time, persistence):
    refresh_rate_list = [30, 120, 30]
    # time_list = [phase_time, 0, phase_time, 0, 0]
    time_list = [0, phase_time, 0, phase_time, 0]
    peak_value = [1,0]
    sampling_rate = 1000
    repeat_times = 10

    vrr_time_array, vrr_luminance_array, _ = signal_vrr(refresh_rate_list=refresh_rate_list,
                                time_list=time_list,
                                persistence=persistence,
                                peak_value=peak_value,
                                sampling_rate=sampling_rate,
                                repeat_times=repeat_times,
                                force_begin_end_equal=True)

    w_s = sampling_rate
    N_s = vrr_time_array.shape[0]
    K_DFT = np.abs(np.fft.fft(vrr_luminance_array)) / N_s
    x_freq = np.arange(0, N_s) * w_s / N_s
    K_mean = vrr_luminance_array.mean()
    X_JND_frequency = []
    Y_JND = []
    for frequency_index in range(N_s):
        frequency = x_freq[frequency_index]
        if frequency == 0:
            continue
        if frequency > 240:
            break
        S = TCSF(frequency)
        C_R = 2 * K_DFT[frequency_index] / K_mean
        JND = S * C_R
        X_JND_frequency.append(frequency)
        Y_JND.append(JND)
    Y_JND = np.array(Y_JND)

    JND_max = Y_JND.max()
    JND_sum = Y_JND.sum()
    JND_combine = np.linalg.norm(Y_JND)
    return JND_max, JND_sum, JND_combine

if __name__ == "__main__":
    phase_time_range = np.arange(0.01, 1, 0.01)
    persistence_range = np.arange(0.01, 1, 0.01)
    phase_time_X_meshgrid, persistence_Y_meshgrid = np.meshgrid(phase_time_range, persistence_range)
    JND_combine_array = np.zeros(phase_time_X_meshgrid.shape)
    for persistence_index in range(JND_combine_array.shape[0]):
        persistence = persistence_range[persistence_index]
        for phase_time_index in range(JND_combine_array.shape[1]):
            phase_time = phase_time_range[phase_time_index]
            JND_max, JND_sum, JND_combine = Calculte_JND_from_persistence(phase_time=phase_time, persistence=persistence)
            JND_combine_array[persistence_index][phase_time_index] = JND_combine

    # # plt.figure(figsize=(20, 10))
    # plt.subplot(3, 1, 1)
    # plt.plot(phase_time_list, JND_max_list)
    # # plt.xlabel('persistence')
    # plt.ylabel('JND Max')
    # plt.subplot(3, 1, 2)
    # plt.plot(phase_time_list, JND_sum_list)
    # # plt.xlabel('persistence')
    # plt.ylabel('JND Sum')
    # plt.subplot(3, 1, 3)
    # plt.plot(phase_time_list, JND_combine_list)
    # plt.xlabel('phase time')
    # plt.ylabel('JND Norm')
    # plt.show()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(phase_time_X_meshgrid, persistence_Y_meshgrid, JND_combine_array, cmap='rainbow')
    ax.set_xlabel('phase_time')
    ax.set_ylabel('persistence')
    ax.set_zlabel('JND Norm')
    ax.view_init(elev=30, azim=30)
    plt.show()