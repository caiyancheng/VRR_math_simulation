import numpy as np
import math
import matplotlib.pyplot as plt

def TCSF(frequency):
    S = np.abs(148.7 * ((1 + 2j * math.pi * frequency * 0.00267) ** (-15) - 0.882 * (
                1 + 2j * math.pi * frequency * 1.834 * 0.00267) ** (-16)))
    return S

if __name__ == "__main__":
    x = np.linspace(0,120,1000)
    y = np.zeros(x.shape)
    for i in range(len(x)):
        y[i] = TCSF(x[i])
    plt.figure(figsize=(6,6))
    plt.plot(x, y)
    plt.title('TCSF')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Contrast Sensitivity')
    plt.savefig('TCSF.png')
    plt.close()