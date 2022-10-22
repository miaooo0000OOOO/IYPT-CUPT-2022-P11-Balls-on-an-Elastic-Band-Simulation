import numpy as np
import matplotlib.pyplot as plt

Em = np.loadtxt('E_move.csv', delimiter=',')
Er = np.loadtxt('E_rotate.csv', delimiter=',')
E = np.loadtxt('E_total.csv', delimiter=',')

plt.figure(figsize=(16, 9))

ax1 = plt.subplot(2, 2, 1,) # 两行一列，位置是1的子图
plt.plot(Em)
plt.ylabel('E_move')

ax2 = plt.subplot(2, 2, 2)
plt.plot(Er)
plt.ylabel('E_rotate')

ax3 = plt.subplot(2,2,3)
plt.plot(E)
plt.ylabel('E_total')

plt.savefig('Energy_plot.jpg')
plt.show()
