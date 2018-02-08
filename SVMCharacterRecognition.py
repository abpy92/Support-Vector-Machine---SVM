import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn import svm


#Shape of digits.data 1797x64
digits = load_digits()

#Shape of digits 1797 x 64
#print(digits.target.shape)


#n_samples = len(digits.images)
#print(digits.images.shape)
#data = digits.images.reshape((n_samples, -1))
#print(data.shape)

clf = svm.SVC(gamma=0.001, C=100)

#learning
#load up the examples in x,y
#shape of digits.data 1796x64 and digits.target 1796x
#digits.target = digits.target[:-1].reshape((len(digits.target[:-1]),-1))
X,y = digits.data[:-10], digits.target[:-10]

#X,y = digits.data[:-10], digits.target[:-10]

clf.fit(X,y)

#print('Prediction :',clf.predict(digits.data[-5]))

print(clf.predict(digits.data[[1]]))

plt.imshow(digits.images[1], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
