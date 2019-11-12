import numpy as np
# import math
# x1=np.array([[1,2,3,4,5],
#              [2,3,4,4,5],
#              [3,3,3,3,3]
#              ])
# print(x1[1])
# print(x1[1].shape)
# x1=np.mat(x1)
# print(x1[1])
# print(x1[1].shape)
# x=list(np.arange(-3,3,0.1))
# w=[[1],[1],[1]]
# w=np.mat(w)
# w1=np.ones(3)
# y=w[0]+w[1]*x+w[2]
# print(y.shape)
# x=np.arange(-3,3,0.1)
# print(x.shape)
# y=w[0]+w[1]*x+w[2]
# print(y.shape)
# import codecs
# s='你好'
# f=codecs.open('test1.txt','w+',encoding='gbk')
# f.write(s)
# f=codecs.open('test1.txt','r+',encoding='gbk')
# z=f.read()
# print(z)
# print(type(z))
# s='阿巴斯'
# s=s.encode('utf-8')
# print(type(s))
class A:
    w1=0
    def __init__(self):
        self.name="123"
    def f1(self):
        print(self.name)
b=A()
a=A()
a.name="2"
a.w1=10
print(a.w1)
#A.w1=9
print(b.w1)
w=np.mat([[1,2],
   [0,3],
   [3,2]])
print(w[:,0].A)
print(np.nonzero(w[:,0].A)[0])
w=np.mat([[1],[-1],[2],[0]])
print(w>0)
w=np.mat([[1],[2]])
if w[0]>0 and w[1]<3:
    print(1)
w=np.mat([[1,2],[3,4],[5,6]])
print(w[1,:])
b=np.mat([[0.23488]])
E=2.23879
w=np.mat([[2.3963298432]])
print(-(E+w))
print(-(E+float(w)))
i=1
print(type("ss"+str(1)))