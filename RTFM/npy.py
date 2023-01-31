import numpy as np
import matplotlib.pyplot as plt
print(np.load('aba2.npy').shape)
'''

z = np.zeros(29)
z[9:] = 1
z = np.repeat(z, 16)
np.save('gt_aba2.npy',z)

'''

y = (np.load('pred.npy'))
y*=1000
print(y.shape)
x = np.arange(0,len(y)).T

plt.plot(x,y)
plt.xlabel("Frame Number")
plt.ylabel("Score")
plt.title("Abnormal Activity Detection")
plt.show()
