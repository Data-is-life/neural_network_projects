import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier


df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values

lbl_ncdr_x_1 = LabelEncoder()
X[:, 1] = lbl_ncdr_x_1.fit_transform(X[:, 1])
lbl_ncdr_x_2 = LabelEncoder()
X[:, 2] = lbl_ncdr_x_2.fit_transform(X[:, 2])
hot_ncdr = OneHotEncoder(categories='auto')
X = hot_ncdr.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

std_sclr = StandardScaler()
X_train = std_sclr.fit_transform(X_train)
X_test = std_sclr.fit_transform(X_test)

def dropout_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform',
                         activation='relu', input_dim=11))
    classifier.add(Dropout(rate=0.1))
    classifier.add(
        Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(
        Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(
        optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=dropout_classifier)

parameters = {'batch_size':[16,24,32], 'nb_epoch': [100,250,500], 
              'optimizer': ['Adamax', 'rmsprop', 'sgd']}

grid_search = GridSearchCV(classifier, parameters,
                           scoring = 'accuracy', cv=10, n_jobs=-1)

grid_search = grid_search.fit(X_train, y_train)
