import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(20000)

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')


def getFormat (train):



    train['Ticket'].fillna(0, inplace=True)

    # train['isSpecial'] = train.Name.str.contains('(', regex=False)

    train['Ticket_Type']=train.Ticket.apply(lambda x : x.isalnum())

    train['Ticket']= train.Ticket.str.extract('(\d+)')

    train['title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)

    train.Fare.fillna(0, inplace=True)

    train.Age.fillna(0, inplace=True)

    train['Cabin_First'] = train.Cabin.str[0]

    train['Cabin_Number'] = train.Cabin.str.replace('[^0-9]', '')

    train['Cabin'].fillna('UNKNOWN-CABIN', inplace=True)

    train['Cabin'] = train.Cabin.str.replace('[^a-zA-Z]', '')

    train['Cabin_Unique'] = train['Cabin'].apply( lambda x : len(set((str(x)))))

    train['Cabins'] = train['Cabin'].apply( lambda x : ''.join(set((str(x)))))

    train['Cabin'] = train['Cabin'].apply (lambda x : len(str(x)))

    trainML = train[['Pclass', 'Sex', 'SibSp', 'Parch','PassengerId',
                     'Fare', 'Embarked', 'Ticket_Type', 'title',
                     'Cabin', 'Cabin_First', 'Cabin_Unique', 'Ticket',
                     'Cabins', 'Cabin_Number']]

    trainML.fillna('Unknown', inplace=True)

    nominal_cols = ['Sex','SibSp', 'Pclass', 'title', 'Embarked', 'Cabin_First', 'Cabins']

    # X = pd.get_dummies(trainML[nominal_cols])

    X = pd.DataFrame()

    trainML[nominal_cols] = trainML[nominal_cols].astype('category')

    cat_columns = trainML.select_dtypes(['category']).columns

    X[cat_columns] = trainML[cat_columns].apply(lambda x: x.cat.codes)

    return X


X = getFormat(train=train)

Y = train['Survived']

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=0.2, random_state=9)



# create the model
model = Sequential()
model.add(Dense(8, input_dim=len(list(X)), init='uniform', activation='sigmoid'))
model.add(Dense(6, init='uniform', activation='sigmoid'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

print model.summary()


# compile the model
opt = ['rmsprop', 'adam']

for op in opt:
  model.compile(loss='binary_crossentropy', optimizer=op, metrics=['accuracy'])


  history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), nb_epoch=200, batch_size= train.shape[0]/100, verbose=0)

  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

  # Plot training & validation loss values
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.show()

  # evaluate the model
  scores = model.evaluate(X_validation, Y_validation)
  print "Optimizer : %s"% op + " Accuracy: %.2f%%" % (scores[1]*100)
