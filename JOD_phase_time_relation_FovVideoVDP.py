import matplotlib.pyplot as plt
import numpy as np
from stCSF_FovVideoVDP.stCSF_frequency import stCSF_frequency
from generate_signal_2 import signal_vrr

stCSF = stCSF_frequency()
stCSF.calculate_fft(sample_interval=0.001)

def Calculte_JOD_from_persistence(phase_time):
    refresh_rate_list = [30, 120, 30]
    # time_list = [phase_time, 0, phase_time, 0, 0]
    time_list = [0, phase_time, 0, phase_time, 0]
    peak_value = [1,0]
    sampling_rate = 10000
    repeat_times = 10
    persistence = 0.5

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
    X_JOD_frequency = []
    Y_JOD_sustained = []
    Y_JOD_transient = []
    for frequency_index in range(N_s):
        frequency = x_freq[frequency_index]
        if frequency == 0:
            continue
        if frequency > 240:
            break
        S_sustained, S_transient = stCSF.get_stCSF(frequency)
        C_R = 2 * K_DFT[frequency_index] / K_mean
        JOD_sustained = S_sustained * C_R
        JOD_transient = S_transient * C_R
        X_JOD_frequency.append(frequency)
        Y_JOD_sustained.append(JOD_sustained)
        Y_JOD_transient.append(JOD_transient)
    Y_JOD_sustained = np.array(Y_JOD_sustained)
    Y_JOD_transient = np.array(Y_JOD_transient)
    JOD_sustained_max = Y_JOD_sustained.max()
    JOD_transient_max = Y_JOD_transient.max()
    JOD_sustained_sum = Y_JOD_sustained.sum()
    JOD_transient_sum = Y_JOD_transient.sum()
    JOD_sustained_norm = np.linalg.norm(Y_JOD_sustained)
    JOD_transient_norm = np.linalg.norm(Y_JOD_transient)
    return JOD_sustained_max, JOD_transient_max, JOD_sustained_sum, JOD_transient_sum, JOD_sustained_norm, JOD_transient_norm

if __name__ == "__main__":
    phase_time_range = np.arange(0.01, 1, 0.01)
    phase_time_list = []
    JOD_sustained_max_list = []
    JOD_transient_max_list = []
    JOD_sustained_sum_list = []
    JOD_transient_sum_list = []
    JOD_sustained_norm_list = []
    JOD_transient_norm_list = []

    for p in phase_time_range:
        phase_time_list.append(p)
        JOD_sustained_max, JOD_transient_max, JOD_sustained_sum, \
            JOD_transient_sum, JOD_sustained_norm, JOD_transient_norm = Calculte_JOD_from_persistence(phase_time=p)
        JOD_sustained_max_list.append(JOD_sustained_max)
        JOD_transient_max_list.append(JOD_transient_max)
        JOD_sustained_sum_list.append(JOD_sustained_sum)
        JOD_transient_sum_list.append(JOD_transient_sum)
        JOD_sustained_norm_list.append(JOD_sustained_norm)
        JOD_transient_norm_list.append(JOD_transient_norm)

    # plt.figure(figsize=(20, 10))
    plt.subplot(3, 1, 1)
    plt.plot(phase_time_list, JOD_sustained_max_list, label='Sustained')
    plt.plot(phase_time_list, JOD_transient_max_list, label='Transient')
    # plt.xlabel('persistence')
    plt.ylabel('JOD Max')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(phase_time_list, JOD_sustained_sum_list, label='Sustained')
    plt.plot(phase_time_list, JOD_transient_sum_list, label='Transient')
    # plt.xlabel('persistence')
    plt.ylabel('JOD Sum')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(phase_time_list, JOD_sustained_norm_list, label='Sustained')
    plt.plot(phase_time_list, JOD_transient_norm_list, label='Transient')
    plt.xlabel('persistence')
    plt.ylabel('JOD Norm')
    plt.legend()
    plt.show()