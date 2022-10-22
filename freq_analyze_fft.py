import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

sx = np.loadtxt('sx.csv',delimiter=',')
t = np.loadtxt('t.csv',delimiter=',')
T = 1e-2
Fs = 1/T
L = len(t)

Y = fft(sx)
p2 = np.abs(Y)   # 双侧频谱
p1 = p2[:int(L/2)]
f = np.arange(int(L/2))*Fs/L;
plt.plot(f,2*p1/L) 
plt.title('sx(t)的单面幅度谱')
plt.xlabel('f (Hz)')
plt.ylabel('|P1(f)|')
plt.show()