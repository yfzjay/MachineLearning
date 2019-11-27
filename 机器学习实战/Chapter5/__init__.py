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
x=np.arange(-3,3,0.1)
w=[[1,2,3]]
w=np.mat(w)
print(w.shape)
#w1=np.ones(3)
index=np.where(w>1)
print(index)
