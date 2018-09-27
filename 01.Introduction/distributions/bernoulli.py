import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

np.random.seed(50)

nobs = 10
theta = 0.7
Y = np.random.binomial(1, theta, nobs)

fig = plt.figure(figsize=(10,5))
gs = gridspec.GridSpec(1, 2, width_ratios=[5, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

ax1.plot(range(nobs), Y, 'x')
ax2.hist(-Y, bins=2)

ax1.yaxis.set(ticks=(0,1), ticklabels=('Failure', 'Success'))
ax2.xaxis.set(ticks=(-1,0), ticklabels=('Success', 'Failure'));

title = 'Bernoulli Trial Outcomes where theta = ', theta
ax1.set(title=title, xlabel='Trial', ylim=(-0.2, 1.2))
ax2.set(ylabel='Frequency')

fig.tight_layout()

plt.show()