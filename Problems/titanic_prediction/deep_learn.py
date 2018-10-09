import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt


train = pd.read_csv('/home/bhanuchander/course/Learn_MachineLearning/data/csv/titanic/train.csv')

train = train.set_index('PassengerId')

print train.shape

print train.Survived.value_counts(normalize=True)

train['Name_len']=train.Name.str.len()

train['Ticket_First']=train.Ticket.str[0]

train['FamilyCount']=train.SibSp+train.Parch

train['Cabin_First']=train.Cabin.str[0]

train['title'] = train.Name.str.extract('\, ([A-Z][^ ]*\.)',expand=False)

train.Fare.fillna(train.Fare.mean(),inplace=True)

trainML = train[['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket',
       'Fare', 'Embarked', 'Name_len', 'Ticket_First', 'FamilyCount',
       'title']]

nominal_cols = ['Embarked', 'Sex', 'Parch', 'Ticket_First', 'title']

trainML[nominal_cols] = trainML[nominal_cols].astype('category')

cat_columns = trainML.select_dtypes(['category']).columns

trainML[cat_columns] = trainML[cat_columns].apply(lambda x: x.cat.codes)

trainML.fillna(0, inplace=True)

X=trainML[['Age', 'SibSp', 'Parch',
       'Fare', 'Sex', 'Pclass','title', 'Name_len','Embarked', 'FamilyCount']] # Taking all the numerical values

Y = trainML['Survived'].values

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


  history = model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), nb_epoch=200, batch_size= trainML.shape[0]/100, verbose=0)

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
