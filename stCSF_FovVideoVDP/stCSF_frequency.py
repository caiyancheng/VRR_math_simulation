import numpy as np
import math
import matplotlib.pyplot as plt
def stCSF_sustained_time(time):
    R_s = 0.00573 * math.exp(-(math.log(time + 1e-4) - math.log(0.06))**2/(2*0.5**2))
    return R_s

def stCSF_transient_time(time):
    R_t = 0.0621 * ((-stCSF_sustained_time(time)*(math.log(time + 1e-4)-math.log(0.06)))/(0.5**2*(time + 1e-4)))
    return R_t

class stCSF_frequency:
    def calculate_fft(self, sample_interval):
        x_t = np.arange(0, 100, sample_interval)
        y_sustained_response_t = np.zeros(x_t.shape)
        y_transient_response_t = np.zeros(x_t.shape)
        for i in range(len(x_t)):
            y_sustained_response_t[i] = stCSF_sustained_time(x_t[i])
            y_transient_response_t[i] = stCSF_transient_time(x_t[i])
        self.y_sustained_response_f = np.abs(np.fft.rfft(y_sustained_response_t))
        self.y_transient_response_f = np.abs(np.fft.rfft(y_transient_response_t))
        self.x_f = np.fft.rfftfreq(len(x_t), sample_interval)

    def draw_fft(self):
        plt.figure()
        plt.plot(self.x_f, self.y_sustained_response_f)
        plt.plot(self.x_f, self.y_transient_response_f)
        plt.title('stCSF_frequency_modulation')
        plt.xlim(0,80)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Modulation')
        plt.show()

    def get_stCSF(self, fequency):
        if fequency < 0:
            raise ValueError('Frequency smaller than 0 Hz')
        elif fequency > 300:
            raise ValueError('Frequency larger than 300 Hz')

        differences = np.abs(fequency - self.x_f)
        f_close_index = np.argmin(differences)
        return self.y_sustained_response_f[f_close_index], self.y_transient_response_f[f_close_index]



if __name__ == "__main__":
    x_f = np.arange(0,80,0.1)
    y_sustained_f = np.zeros(x_f.shape)
    y_transient_f = np.zeros(x_f.shape)
    stCSF = stCSF_frequency()
    stCSF.calculate_fft(sample_interval=0.001)
    # stCSF.draw_fft()
    for i in range(len(x_f)):
        y_sustained_f[i], y_transient_f[i] = stCSF.get_stCSF(fequency=x_f[i])
    # plt.figure(figsize=(10,4))
    plt.plot(x_f, y_sustained_f)
    plt.plot(x_f, y_transient_f)
    # plt.xlim([0,80])
    plt.title('stCSF_frequency_modulation')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Modulation')
    plt.show()
    # plt.savefig('TCSF.png')
    # plt.close()