import pickle
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.layers import Dropout
import numpy as np
from keras import backend as K
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
import datetime
import time
from keras.models import load_model


data_path = "expert_data/Hopper-v2.pkl"

file = open(data_path, 'rb')

# dump information to that file
data = pickle.load(file)
file.close()

print("type of data is: ", type(data))

print("keys are: ", data.keys())

demonstrated_actions = data['actions']
demonstrated_obs = data['observations']

# logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/scalars/" + str(time.time())
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

print("type of actions is: ", type(demonstrated_actions))
print("type of obs are: ", type(demonstrated_obs))

print("shape of actions is: ", demonstrated_actions.shape)
print("shape of observations is: ", demonstrated_obs.shape)
print("example state-action pair is: ", (demonstrated_actions[0], demonstrated_obs[0]))

print("range of observations is: ", (np.max(demonstrated_obs), np.min(demonstrated_obs)))
print("range of actions is: ", (np.max(demonstrated_actions), np.min(demonstrated_actions)))

observation_dim = demonstrated_obs[0].shape[0]
print("observation dimension is: ", observation_dim)

demonstrated_actions = demonstrated_actions.reshape(demonstrated_actions.shape[0], demonstrated_actions.shape[2])
X_train, X_test, y_train, y_test = train_test_split(demonstrated_obs, demonstrated_actions,
                                                    test_size=0.2, random_state=42)

print("shape of x_train is: ", X_train.shape)
print("shape of y_train is: ", y_train.shape)

model = Sequential()
# model_path = 'bc.h5'
# model = load_model(model_path)

model.add(Dense(8, input_dim=observation_dim))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.3))
model.add(Dense(4, activation='linear'))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.3))
model.add(Dense(3, activation='linear'))
sgd = SGD(lr=0.001, momentum=0.8)
model.compile(loss='mse', optimizer=sgd)
print(model.summary())

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=100, batch_size=32, callbacks=[tensorboard_callback])

val_loss = history.history['val_loss']
train_loss = history.history['loss']

model.save('bc.h5')


plt.plot(val_loss, label="validation")
plt.plot(train_loss, label="train")
plt.legend()

plt.show()