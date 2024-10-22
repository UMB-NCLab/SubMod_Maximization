from matplotlib import pyplot as plt
import numpy as np

exp = np.loadtxt(f'f_value_exp.txt')
ring = np.loadtxt(f'f_value_ring.txt')
random = np.loadtxt('f_value_random.txt')
star = np.loadtxt('f_value_star.txt')
plt.plot(exp, label='exp')
plt.plot(ring, label="ring")
plt.plot(random, label="random")
plt.plot(star, label="star")
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Facility location')
plt.legend()

plt.savefig(f'plot.png')