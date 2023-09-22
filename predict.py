import cv2
from keras.models import load_model
import numpy as np
sign = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','S','T','U','V','W','X','Y']
model = load_model('model/model.h5')

img = cv2.imread('S_13.jpg')
img = cv2.resize(img, (100,100))
img = np.array(img)
img = img.reshape(100,100,3)
img = img.astype('float32')
img = img/255
X = []
X.append(img)
X = np.asarray(X)
predict = model.predict(X)
predict = np.argmax(predict)
print(sign[predict])
