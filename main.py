import numpy as np
from matplotlib import pyplot as plt
import math 

class Env:
    DT = 1e-3
    G = 1
    def __init__(self):
        # 单位都是国际制单位
        self.sx, self.sy, self.r, self.m, self.vx, self.vy = [], [], [], [], [], []
        self.color = []
        self.t = 0
        self.n = 0
        self.L = 0.001     # 扭转圈数为0时，橡皮筋的原长
        self.nTurns = 0 # 橡皮筋的扭转圈数
        # self.k = (5e7)*(math.pi*(3/1000)**2)/(10/100)      # 橡皮筋的劲度系数 k = Y*S/L Y杨氏模量 S横截面积 L长度
        self.k = 20
        self.mu = 1      # 摩擦系数
        self.g = 9.8       #重力加速度
        
        self.posList = []
        self.tList = []

    def add_ball(self, s, v, r, m ,color=100):
        self.sx.append(s[0])
        self.sy.append(s[1])
        self.vx.append(v[0])
        self.vy.append(v[1])
        self.m.append(m)
        self.r.append(r)
        self.color.append(color)
        self.n += 1

    def get_ball(self, index):
        # [s, v, r, m ,color]
        return [[self.sx[index], self.sy[index]], [self.vx[index], self.vy[index]], self.r[index], self.m[index], self.color[index]]
    
    def get_dis(self, i, j):
        return ((self.sx[i]-self.sx[j])**2 + (self.sy[i]-self.sy[j])**2)**0.5

    def pop_ball(self, index):
        res = self.get_ball(index)
        self.sx.pop(index)
        self.sy.pop(index)
        self.vx.pop(index)
        self.vy.pop(index)
        self.m.pop(index)
        self.r.pop(index)
        self.color.pop(index)
        return res
    
    def set_ball(self, index, s, v, r, m, color):
        self.sx[index], self.sy[index] = s
        self.vx[index], self.vy[index] = v
        self.r[index] = r
        self.m[index] = m
        self.color[index] = color
    
    def ball_to_str(self, index):
        return "pos: [%.2f, %.2f]\nv: [%.2f, %.2f]\nr: %.2f\nm: %.2f" % (self.sx[index], self.sy[index], self.vx[index], self.vy[index], self.r[index], self.m[index])

    def step(self):
        DT = self.DT
        for i in range(self.n):
            F = [0, 0]
            for j in range(self.n):
                if i==j:
                    continue
                r = self.get_dis(i, j) # 球心距
                ds_ij = [self.sx[j]-self.sx[i], self.sy[j]-self.sy[i]] # s_i - s_j
                dL = max(0, r-self.r[i]-self.r[j]-self.L)   # 橡皮筋形变量
                F_length = self.k*dL       # 橡皮筋拉力大小 遵循胡克定律
                dFx = ds_ij[0]/r*F_length  # 力在x轴的分量
                dFy = ds_ij[1]/r*F_length  # 力在y轴的分量
                F[0] += dFx
                F[1] += dFy
                if self.vx[i]**2+self.vy[i]**2 >= 0.0001:   # 如果小球有速度
                    v_length = (self.vx[i]**2+self.vy[i]**2)**0.5
                    f_length = self.mu*self.m[i]*self.g
                    if f_length > F_length:
                        f_length = F_length
                    dFx = -self.vx[i]/v_length*f_length
                    dFy = -self.vy[i]/v_length*f_length
                    F[0] += dFx
                    F[1] += dFy
                if self.r[i]+self.r[j] >= r:
                    print(f'球{i}与{j}相撞')
                    raise
            a = [F[0]/self.m[i], F[1]/self.m[i]]
            self.vx[i] += DT * a[0]
            self.vy[i] += DT * a[1]
        for i in range(self.n):
            self.sx[i] += DT * self.vx[i]
            self.sy[i] += DT * self.vy[i]
        self.t += DT
    
    def render(self, track=True):
        DT = self.DT
        if not track:
            plt.clf()
        plt.scatter(self.sx, self.sy,c=self.color)
        plt.pause(1e-2)
    
    def realtime_plot(self):
        self.posList.append(self.sx[0])
        self.tList.append(self.t)
        # plt.plot(self.tList, self.posList, color='r')
        # plt.pause(1e-3)
    
e = Env()
p1, p2 = [0,0], [0.07,0]
v1, v2 = [0,0.4], [0,-0.4]
m1, m2 = 16/1000, 16/1000
r1, r2 = 0,0
e.add_ball(p1, v1, r1, m1, color=10)
e.add_ball(p2, v2, r2, m2, color=20)
while e.t < 3.5:
    e.step()
    print(e.ball_to_str(0))
    print(e.ball_to_str(1))
    print('distance: %.2f cm' % (e.get_dis(0,1)*100))
    # e.realtime_plot()
    e.render()
# np.savetxt('sx.csv', e.posList, '%.5f', delimiter=',')
# np.savetxt('t.csv', e.tLi1