# In this problem a set of images from	the sklearn dataset is loaded
#
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
digits = datasets.load_digits()

'''
# Plot the image
plt.figure()
plt.imshow(digits.images[0], cmap= plt.cm.gray_r,interpolation = 'nearest')
plt.show()
# Plot the label
print(digits.target[0])
'''

# You need to get just some images from the dataset to make the training
# because you want images that your program never seen to measure the 
# accuracy
X_train = digits.images[0:1000]
Y_train = digits.target[0:1000]
# Now you can choose some test image
X_test = digits.images[345]

# And here we can make the neighbor classifier that was used on point-classification.py

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
plt.figure()
plt.imshow(X_test, cmap=plt.cm.gray_r,interpolation = 'nearest')
plt.show()

# Now let's analyse the accuracy of the classifier
ok = 0
errors = 0
# choosing 100 images
for j in range(1200,1300):
	X_test = digits.images[j]
	for i in range(num):
		distance[i] = dist(X_train[i], X_test)
	min_index = np.argmin(distance)
	if Y_train[min_index] != digits.target[j]:
		errors += 1
	else:
		ok +=1
print('Accuracy = ', ok,'%')
print('Number of errors = ',errors)
		
		
