import numpy as np
from matplotlib import pyplot as plt
import cv2

video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640,480))

def get_len(v):
    return (np.sum(v**2))**0.5

DT = 1e-3*2
sizeL = ([5]*100)[:]
sizeL[0] = 100
E_total_list = []
E_move_list = []
E_rotate_list = []
steps = 0
class Env:
    def __init__(self):
        self.s, self.v, self.r, self.m = np.array([],np.float128), np.array([],np.float128), np.array([],np.float128), np.array([],np.float128)
        self.color = []
        self.t = 0                  # 时间
        self.n = 0                  # 球数
        # self.L = 7/100              # 扭转圈数为0时，橡皮筋的原长
        self.theta = 30*2*np.pi      # 橡皮筋扭转弧度
        self.k = 5                 # 橡皮筋的劲度系数
        self.kn = 2.5e-6           # 橡皮筋扭转系数
        self.mu = 0.05              # 动摩擦系数
        self.g = 9.8                #重力加速度
    
    def getL(self):
        '''
        橡皮筋原长L是关于扭转弧度theta的函数
        使用一次函数拟合
        '''
        # return -0.000398*abs(self.theta)+0.085
        return 0/100

    def add_ball(self, s, v, r, m, color):
        self.s = np.append(self.s, s).reshape([-1,2])
        self.v = np.append(self.v, v).reshape([-1,2])
        self.m = np.append(self.m, m)
        self.r = np.append(self.r, r)
        self.color.append(color)
        self.n += 1

    def confirm(self):
        '''
        之后小球个数不能变化
        '''
        self.sList = np.array([],np.float128).reshape([-1,self.n,2])

    def step(self):
        '''
        半隐式欧拉法的物理引擎
        小球受橡皮筋拉力，橡皮筋扭力，桌面阻力
        '''
        global steps
        E_total = E_move = E_rotate = 0
        for i in range(self.n):
            F = np.array([0, 0], np.float128)   #作用在球i上的合力
            for j in range(self.n):
                if i==j:
                    continue
                # 计算常用值
                d = get_len(self.s[i]-self.s[j])    # 两球球心距
                ds_ij = self.s[j]-self.s[i]         # 球i指向球j的向量
                v_length = get_len(self.v[i])       # 速度的模长
                # 计算橡皮筋的拉力
                F_length = self.k*max(0, d-self.r[i]-self.r[j]-self.getL())
                # print("F:"+str(F_length))
                dF = ds_ij/d*F_length
                F += dF
                # 计算摩擦力
                if  v_length >= 0.0001:
                    f_length = self.mu*self.m[i]*self.g
                    f_length = min(f_length,F_length)
                    dF = -self.v[i]/v_length*f_length
                    F += dF
                # 计算橡皮筋扭转力矩从而算出扭力
                Fn_length = self.kn*self.theta/self.r[i]
                # 使向量绕原点逆时针转90度
                M_turn90 = np.array([[0,-1],
                                     [1, 0]])
                Fn = -np.dot(M_turn90,ds_ij)/d*Fn_length
                F += Fn
                # 碰撞告警
                if self.r[i]+self.r[j] >= d and False:
                    print(f'球{i}与{j}相撞')
                    raise
            # 计算加速度
            a = F/self.m[i]
            # 计算速度
            self.v[i] += a*DT
            # 计算橡皮筋扭转弧度
            ds = self.v[i]*DT
            ds_par = ds_ij/d**2*np.dot(ds, ds_ij)       # ds平行于ds_ij
            ds_ver = ds - ds_par                        # ds垂直于ds_ij
            sgn = -np.cross(ds_ver, ds_ij)/abs(np.cross(ds_ver, ds_ij))
            self.theta += sgn*get_len(ds_ver)/self.r[i]
            # 计算能量
            # 平动动能
            E_move += 0.5*self.m[i]*v_length**2
            # 转动动能
            I = 2/5*self.m[i]*self.r[i]**2  # 转动惯量
            omega = get_len(ds_ver)/self.r[i]/DT    # 角速度
            E_rotate += 0.5*I*omega**2
            E_total = E_move + E_rotate
        steps += 1
            
            

        for i in range(self.n):
            # 计算位移
            self.s[i] += self.v[i]*DT
        if steps%5==0:
            self.sList = np.append(self.sList,self.s.reshape([-1,self.n,2]), axis=0)

            if len(self.sList) >= 100:
                self.sList = np.delete(self.sList, 0, axis=0)
            
            E_move_list.append(E_move)
            E_rotate_list.append(E_rotate)
            E_total_list.append(E_total)
        self.t += DT
    
    def render(self):
        plt.clf()
        plt.plot([self.sList[-1,0,0], self.sList[-1,1,0]], [self.sList[-1,0,1], self.sList[-1,1,1]], '-.',color='brown')
        for i in range(self.n):
            plt.scatter(self.sList[:,i,0], self.sList[:,i,1], c=self.color[i], s=sizeL[:len(self.sList)][::-1])
        C = [3.5 ,0]
        DC = 10
        plt.xlim((C[0]-DC)/100, (C[0]+DC)/100)
        plt.ylim((C[1]-DC)/100, (C[1]+DC)/100)
        if True:
            self.write_video()
        plt.pause(1e-5)
    
    def write_video(self):
        plt.savefig('prosessing.jpg')
        frame = cv2.imread('prosessing.jpg')
        video.write(frame)
    
e = Env()

p1, p2 = [0,0], [4/100,0]
v1, v2 = [0,0/100], [0,-0/100]
m1, m2 = 16/1000, 16/1000
r1, r2 = 1.4/100, 1.4/100
e.add_ball(p1, v1, r1, m1, color='r')
e.add_ball(p2, v2, r2, m2, color='b')
e.confirm()
tList, vList = [], []
try:
    while e.t < 10:
        e.step()
        if steps%10==0:
            e.render()
            print('distance: %.2f cm\ttime: %.2f s\t|v1|: %.2fm/s\ttheta/2pi: %.2f\tL(|n|): %.2f cm' % \
            (get_len(e.s[0]-e.s[1])*100, e.t, get_len(e.v[0]), e.theta/6.283, e.getL()*100))
            tList.append(e.t)
            vList.append(get_len(e.v[0]))
finally:
    video.release()
    np.savetxt('E_total.csv', np.array(E_total_list))
    np.savetxt('E_move.csv', np.array(E_move_list))
    np.savetxt('E_rotate.csv', np.array(E_rotate_list))

# plt.plot(tList, vList)
# plt.savefig('vt.jpg')
# plt.show()
'''
m cm 
1:100
'''