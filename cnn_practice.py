import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle
from tensorflow.keras import layers, models
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard
from time import time
from keras.models import load_model
np.set_printoptions(precision=5,suppress=True)
# x_train = x_train[...,None]
# x_test = x_test[...,None]

# # Model / data parameters
# num_classes = 10
# input_shape = (28, 28, 1)

# # Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
print(x_test[0])
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(x_train.shape)




# model = keras.Sequential(
#     [
#         keras.Input(shape=input_shape),
#         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2, 2)),
#         layers.Flatten(),
#         layers.Dropout(0.5),
#         layers.Dense(num_classes, activation="softmax"),
#     ]
# )


from PIL import Image

model = load_model('model.h5')

img = Image.open("realdebug.png").convert("L")
x = np.array(img)
print(x)
x = x.astype("float32") / 255
print(x)

x = np.expand_dims(x, -1)
print(x.shape)



test_data = x[None,...]
print(test_data.shape)
print(model.predict(test_data))
res =(model.predict(test_data) > 0.5).astype("int32")
# res = model.predict(test_data)

# tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
# model.compile(optimizer='adam', 
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])


# model.fit(x_train, y_train, epochs=30, callbacks=[tensorboard])
# model.save("model.h5")
# print("==========================================")
# print(model.evaluate(x_test, y_test))
