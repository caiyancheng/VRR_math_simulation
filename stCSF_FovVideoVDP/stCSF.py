import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def stCSF_sustained_time(time):
    R_s = 0.00573 * math.exp(-(math.log(time + 1e-4) - math.log(0.06))**2/(2*0.5**2))
    return R_s

def stCSF_transient_time(time):
    R_t = 0.0621 * ((-stCSF_sustained_time(time)*(math.log(time + 1e-4)-math.log(0.06)))/(0.5**2*(time + 1e-4)))
    return R_t

if __name__ == "__main__":
    x_t = np.arange(0,1,0.001)
    y_sustained_response_t = np.zeros(x_t.shape)
    y_transient_response_t = np.zeros(x_t.shape)
    for i in range(len(x_t)):
        y_sustained_response_t[i] = stCSF_sustained_time(x_t[i])
        y_transient_response_t[i] = stCSF_transient_time(x_t[i])
    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plt.plot(x_t, y_sustained_response_t)
    plt.plot(x_t, y_transient_response_t)
    plt.xlim([0, 0.4])
    plt.title('stCSF_time_response')
    plt.xlabel('Time (s)')
    plt.ylabel('Response')

    plt.subplot(1, 2, 2)
    y_sustained_response_f = np.abs(np.fft.rfft(y_sustained_response_t))
    y_transient_response_f = np.abs(np.fft.rfft(y_transient_response_t))
    x_f = np.fft.rfftfreq(len(x_t), 0.001)
    plt.plot(x_f, y_sustained_response_f)
    plt.plot(x_f, y_transient_response_f)
    plt.xlim([0,80])
    plt.title('stCSF_frequency_modulation')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Modulation')
    plt.show()
    # plt.savefig('TCSF.png')
    # plt.close()