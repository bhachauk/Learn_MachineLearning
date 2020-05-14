import pandas as pd
import keras
import tensorflow as tf
from matplotlib import pyplot as plt


df = pd.read_csv('/media/bhanuchanderu/nova/nyc_taxi.csv')
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime']).dt.weekday
train_data = df[['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',  'dropoff_latitude',
                 'passenger_count']]
target_data = df[['fare_amount']]

print("Train :")
print(train_data.head())

print("Test :")
print(target_data.head())

ip_shape = train_data.shape[1]


def simple_model():
    return keras.Sequential([keras.layers.Dense(units=1, input_shape=[ip_shape], kernel_initializer='normal')])


def two_layer_model():
    return keras.Sequential([keras.layers.Dense(units=3, input_shape=[ip_shape], kernel_initializer='normal'),
                             keras.layers.Dense(units=1)])


def three_layer_model():
    return keras.Sequential([keras.layers.Dense(units=3, input_shape=[ip_shape], kernel_initializer='normal'),
                             keras.layers.Dense(units=2, kernel_initializer='normal'),
                             keras.layers.Dense(units=1)])


models = [
    ('simple', simple_model()),
    ('two_layer', two_layer_model()),
    ('three_layer', three_layer_model())
]


class myCallback (tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 10 == 0:
            print("\n Epoch : {} , Loss : {}".format(epoch, str(logs)))


for name, model in models:
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='mean_squared_error')
    print('--------------------------')
    print('Model : ', name)
    history = model.fit(train_data, target_data, epochs=100, verbose=0, validation_split=0.1)
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.title(name)
    plt.show()
