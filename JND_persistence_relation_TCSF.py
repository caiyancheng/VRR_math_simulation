import matplotlib.pyplot as plt
import numpy as np
from TCSF.TCSF import TCSF
from generate_signal import signal_vrr


def Calculte_JND_from_persistence(persistence):
    refresh_rate_list = [30, 240, 30]
    time_list = [0,1,0,1,0]
    peak_value = [1,0]
    sampling_rate = 10000

    vrr_time_array, vrr_luminance_array, _ = signal_vrr(refresh_rate_list=refresh_rate_list,
                                time_list=time_list,
                                persistence=persistence,
                                peak_value=peak_value,
                                sampling_rate=sampling_rate)

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

persistence_range = np.arange(0.01, 1, 0.01)
p_list = []
JND_max_list = []
JND_sum_list = []
JND_combine_list = []

for p in persistence_range:
    p_list.append(p)
    JND_max, JND_sum, JND_combine = Calculte_JND_from_persistence(persistence=p)
    JND_max_list.append(JND_max)
    JND_sum_list.append(JND_sum)
    JND_combine_list.append(JND_combine)

# plt.figure(figsize=(20, 10))
plt.subplot(3, 1, 1)
plt.plot(p_list, JND_max_list)
# plt.xlabel('persistence')
plt.ylabel('JND Max')
plt.subplot(3, 1, 2)
plt.plot(p_list, JND_sum_list)
# plt.xlabel('persistence')
plt.ylabel('JND Sum')
plt.subplot(3, 1, 3)
plt.plot(p_list, JND_combine_list)
plt.xlabel('persistence')
plt.ylabel('JND Norm')
plt.show()