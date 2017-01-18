# How this problem work ? 
# You have a set of points with labels red and green and you want to 
# classify a new test point with the correct label
# for that purpose you'll measure the distance between the test point
# and all the train points and your predict label will be the lowest
# distance.

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

# if you're in a notebook you can use
# %matplotlib inline
X_train = np.array([[1,1], [2,2.5],[3,1.2],[5.5,6.3],[6,9],[7,6]])
Y_train = ['red','red','red','blue','blue','blue'] 
X_test = np.array([3,4])

def dist(x,y):
	return np.sqrt(np.sum((x-y)**2))	

num = len(X_train)
distance = np.zeros(num)
for i in range(num):
	distance[i] = dist(X_train[i],X_test)

min_index = np.argmin(distance)
# Making the prediction
print('prediction = ', Y_train[min_index])
# Ploting the data
fig, axs = plt.subplots(1,2, sharex = True)
fig.subplots_adjust(left=0.08, right=0.98, wspace=0.3)

ax = axs[0]
ax.scatter(X_train[:,0], X_train[:,1], s = 170, color = Y_train[:])
ax.scatter(X_test[0], X_test[1],s = 170,color = 'green') 
ax.set_title('without prediction')
ax = axs[1]
ax.scatter(X_train[:,0], X_train[:,1], s = 170, color = Y_train[:])
ax.scatter(X_test[0], X_test[1],s = 170,color = Y_train[min_index])
ax.set_title('with prediction')
plt.show()
