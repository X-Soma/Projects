import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion = keras.datasets.fashion_mnist
(xtrain, ytrain), (xtest, ytest) = fashion.load_data()

# A sample of image from the data set

imgIndex = 6 #Change the numbers to see a different plt
image = xtrain[imgIndex]
print("image Label :", ytrain[imgIndex])
plt.imshow(image)
# Shape of both the training data

print(xtrain.shape)
print(xtest.shape)


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
print(model.summary())


xvalid, train = xtrain[:5000] / 255.0, xtrain[5000:] / 255.0
yvalid, ytrain = ytrain[:5000], ytrain[5000:]

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
history = model.fit(train, ytrain, epochs= 30,validation_data=(xvalid, yvalid))

# New predictions
new = xtest[:5]
predictions = model.predict(new)
print(predictions)

# Predicted classes 
classes = np.argmax(predictions, axis=1)
print(classes)