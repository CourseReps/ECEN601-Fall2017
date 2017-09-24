import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return 1 - (np.exp(-a*t))

def g(t):
    return 1 - (np.exp(-2*t) * np.cos(np.pi*t))

t1 = np.arange(0.0, 2.00, 0.01)

bestA = 0.0
minL = 100.0

# it helps to know where to look
for i in range(4600, 4605):
    a = i/1000.0
    L2 = 0
    for t in t1:
        L2 = L2 + (f(t)-g(t))**2
    
    if L2 < minL:
        minL = L2
        bestA = a

RMSE = np.sqrt(minL/len(t1))
minL = np.sqrt(minL)

print('Minimizing a:'+repr(bestA)+'\nL2 norm:'+repr(minL)+
        '\nResidual mean squared error:'+repr(RMSE))


a = bestA

plt.figure(1)
plt.plot(t1, f(t1), 'r', label='f(t)')
plt.plot(t1, g(t1), 'b', label='g(t)')
plt.legend()
plt.show()
