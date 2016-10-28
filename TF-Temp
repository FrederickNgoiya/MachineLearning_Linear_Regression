import keras
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.core import Dropout , Activation

seed = 7
np.random.seed(seed)
df = pd.read_csv('input.csv', sep=',')
X = df.values
df1 = pd.read_csv('output.csv', sep=',')
Y = df1.values

#create model
model = Sequential()
model.add(Dense(100, input_dim=X.shape[1], init='uniform', activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(100, init='uniform', activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, init='uniform', activation='linear'))

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model.compile(loss='mse', optimizer='adam')
history = model.fit(X, Y, nb_epoch=3000, batch_size=20, validation_split=0.25,shuffle=True)
#predicted = model.predict(df1.values)

#validation_set
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim(0.,100)
plt.show()




Y_predict = model.predict(X)
plt.plot(Y, label='real')
plt.plot(Y_predict, label='predicted')
plt.legend()
plt.show()



