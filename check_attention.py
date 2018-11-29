import numpy as np
import scipy.io
import matplotlib.pyplot as plt

mods = ['roff_3', 'roff_5']

def import_from_mat(data, size):
    features = []
    labels = []
    for mod in mods:
        real = np.array(data[mod].real[:size])
        imag = np.array(data[mod].imag[:size])
        signal = np.concatenate([real, imag], axis=1)
        features.append(signal)
        labels.append(mods.index(mod) * np.ones([size, 1]))

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


data = scipy.io.loadmat(
    "E:/qpsk_butter_3_4_5_6_snr10.dat",
)

f, l = import_from_mat(data, 1000)

print()

