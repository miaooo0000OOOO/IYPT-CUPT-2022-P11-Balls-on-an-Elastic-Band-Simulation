import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
from matplotlib import cm

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

sx = np.loadtxt('sx.csv',delimiter=',')
t = np.loadtxt('t.csv',delimiter=',')
T = 1e-2
Fs = 1/T
L = len(t)

# # STFT处理绘制声谱图
# X = librosa.stft(sx,n_fft=250)
# Xdb = librosa.amplitude_to_db(abs(X))  # X--二维数组数据

# plt.figure(figsize=(14, 5))
# librosa.display.specshow(Xdb, sr=Fs, x_axis='time', y_axis='hz',fmax=10)
# plt.colorbar()
# plt.title('STFT transform processing audio signal')
# plt.show()

rate=Fs
Block_size=1000
freqs, times, Sxx = signal.spectrogram(sx, fs=Fs, window='hanning',
                                      nperseg=Block_size, noverlap=0.5*Block_size,
                                      detrend=False, scaling='spectrum')
plt.figure()
plt.pcolormesh(times, freqs, 20 * np.log10(Sxx/1e-06), cmap='inferno')
# plt.clim(70,150)
plt.ylim(0,3)
plt.colorbar()
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]');

X=times
Y=freqs
Z=20*np.log10(Sxx/1e-06)
X,Y=np.meshgrid(X,Y)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma,
#                        linewidth=0, antialiased=False)
surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma,
                       linewidth=0, antialiased=False)
ax.set_ylim(0,3)
ax.set_xlabel('time')
ax.set_ylabel('freq')
ax.set_zlabel('db')
fig.colorbar(surf)

plt.show()