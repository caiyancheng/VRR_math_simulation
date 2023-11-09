import numpy as np
import math
import matplotlib.pyplot as plt

def TCSF_2d(frequency, Td):
    t = np.log10(Td)
    s = (3 * frequency + 23.8 * (-3 + t) - 62.1 * t) / ((23.8 - 10.5) * (-3 + t) + (-62.1 + 45.3) * t)
    S = 10 ** s
    return S

def TCSF(frequency):
    S = np.abs(148.7 * ((1 + 2j * math.pi * frequency * 0.00267) ** (-15) - 0.882 * (
                1 + 2j * math.pi * frequency * 1.834 * 0.00267) ** (-16)))
    return S

def TCSF_new(frequency, Td, single=True):
    K = 40  # Hz threshold value
    prop = 0.44012363507380736 #173.10083837467033 / S_max
    if not single: #np.array
        frequency_1 = frequency[frequency > K].reshape(frequency.shape[0], -1)
        frequency_2 = frequency[frequency <= K].reshape(frequency.shape[0], -1)
        Td_1 = Td[:, frequency_2.shape[1]:]
        Td_2 = Td[:, :frequency_2.shape[1]]
        S_1 = TCSF_2d(frequency_1, Td_1)
        S_2 = TCSF(frequency_2) * TCSF_2d(K, Td_2) / TCSF(K)
        S = np.concatenate([S_2, S_1], axis=-1)
        # S_max = S.max()
        S = S * prop#173.10083837467033 / S_max
        return S
    else: #single number
        if frequency > K:
            S = TCSF_2d(frequency, Td)
        else:
            S = TCSF(frequency) * TCSF_2d(K, Td) / TCSF(K)
        S = S * prop
        return S



if __name__ == "__main__":
    x_frequency = np.linspace(1, 60, 1000)
    y_Td = np.linspace(1, 1000, 1000)
    # y_Td = np.full(1, 1000)
    X, Y = np.meshgrid(x_frequency, y_Td)
    z_S = TCSF_new(X,Y,single=False)

    fig = plt.figure()
    ax3 = plt.axes(projection='3d')
    ax3.plot_surface(X, Y, z_S, cmap='rainbow')
    plt.title('TCSF_2d')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Retinal Illuminance (Td)')
    ax3.set_zlabel('Contrast Sensitivity')
    ax3.view_init(elev=20, azim=110)
    plt.savefig('TCSF_2d.png')
    plt.close()