from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from numpy import array

# return training data
def get_train():
    seq = [[0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
    seq = array(seq)
    X, y = seq[:, 0:2], seq[:, 2]
    print X
    X = X.reshape((len(X), 2, 1))
    return X, y

# # define model
# model = Sequential()
# model.add(LSTM(10, input_shape=(2, 1)))
# model.add(Dense(1, activation='linear'))
# # compile model
# model.compile(loss='mse', optimizer='adam')



X, y = get_train()

print X
print y


# model.fit(X, y, epochs=300, shuffle=False, verbose=0)
# # save model to single file
# model.save('lstm_model.h5')

# load model from single file
model = load_model('lstm_model.h5')
# make predictions

yhat = model.predict(X, verbose=0)

print(yhat)

score = model.evaluate(X, y, verbose=0)

print score