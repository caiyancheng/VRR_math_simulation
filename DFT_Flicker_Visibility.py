import matplotlib.pyplot as plt
import numpy as np
from TCSF.TCSF import TCSF
from generate_signal_2 import signal_vrr


def plot_pict(x_array, y_array, x_label, y_label, title, fig_size=False, save=False, save_fig_name='no name'):
    if fig_size:
        plt.figure(figsize=fig_size)
    plt.plot(x_array, y_array)
    plt.title(label=title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save:
        plt.savefig(f'{save_fig_name}.png')
        plt.close()
    else:
        plt.show()

refresh_rate_list = [30, 120, 30]
time_list = [0.5, 0, 0.5, 0, 0]
persistence = 0.5
peak_value = [1,0]
sampling_rate = 10000
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
x_freq_sub = x_freq[x_freq <= 240] #只看240帧以下
N_s_sub = x_freq_sub.shape[0]

plot_pict(x_array=x_freq[1:N_s_sub], y_array=K_DFT[1:N_s_sub],
          x_label='Frequency', y_label='Luminance ($cd / m^2$)',
          title='spectrum overall', save=False, save_fig_name='spectrum_overall_sub_np_0')

K_mean = vrr_luminance_array.mean()

# Calculate JND:
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

X_JND_frequency = np.array(X_JND_frequency)
Y_JND = np.array(Y_JND)

JND_max = Y_JND.max()
JND_sum = Y_JND.sum()
JND_combine = np.linalg.norm(Y_JND)

plot_pict(x_array=X_JND_frequency, y_array=Y_JND, x_label='Frequency', y_label='JND',
            title='JND_Frequency', save=False, fig_size=(6,3), save_fig_name='JND_Frequency')

print('JND Max', JND_max)
print('JND Sum', JND_sum)
print('JND combine', JND_combine)