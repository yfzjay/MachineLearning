import numpy as np
x1=np.array([[1,2,3,4,5],
             [2,3,4,4,5],
             [3,3,3,3,3]
             ])
print(x1[1])
print(x1[1].shape)
x1=np.mat(x1)
print(x1[1])
print(x1[1].shape)
x=list(np.arange(-3,3,0.1))
w=[[1],[1],[1]]
w=np.mat(w)
w1=np.ones(3)
y=w[0]+w[1]*x+w[2]
print(y.shape)
x=np.arange(-3,3,0.1)
print(x.shape)
y=w[0]+w[1]*x+w[2]
print(y.shape)