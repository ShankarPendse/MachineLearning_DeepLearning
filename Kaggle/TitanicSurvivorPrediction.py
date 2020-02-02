# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
test = pd.read_csv("../input/titanic/test.csv")
train = pd.read_csv("../input/titanic/train.csv")

train.head()

train.isnull().sum()

test.isnull().sum()

train.describe(include=['O'])

# **As we can see above, Ticket and Cabin are mostly shared, so we can actually drop these two features from the data set**

dataset = [train,test]
for data in dataset:
    data.drop(['Ticket','Cabin'], axis = 1, inplace = True)

# **Age, Embarked and Fare have null values in train and test set, lets fill the null values with median value for Age, mode value for Embarked and Median value for Fare**

for data in dataset:
    data['Age'].fillna(data['Age'].median(), inplace = True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)
    data['Fare'].fillna(data['Fare'].median(), inplace = True)

train.isnull().sum()

test.isnull().sum()

# **Now, we have managed to get rid of Null values**

# **Lets extract the titles(Mr,Mrs,Ms,..) from the Name column and create a new attribute called Title**

for data in dataset:
    data['Title'] = data['Name'].str.split(',',expand = True)[1].str.split('.',expand = True)[0]

train['Title'].value_counts()

test['Title'].value_counts()

train_title_counts = train['Title'].value_counts() < 10
test_title_counts = test['Title'].value_counts() < 10

# **Apart from Mr, Mrs, Ms there are some special titles like, Rev, Major, col, capt.. Lets replace all those special titles with Misc**
Misc_count = 10

train['Title'] = train['Title'].apply(lambda x: 'Misc' if train_title_counts.loc[x] == True else x)
test['Title'] = test['Title'].apply(lambda x: 'Misc' if test_title_counts.loc[x] == True else x)

# **PassengerId, Name and Cabin will not be of much use in determining the survival category, so we will drop them from our train and test data set**

train['Title'].value_counts()

test['Title'].value_counts()

train.columns

test.columns

# **We can create two new features from SibSp and Parch features, namely: Family_size and Is_alone. So lets create them**

for data in dataset:
    data['Family_size'] = data['SibSp'] + data['Parch'] + 1
    data['Is_alone'] = 1
    data['Is_alone'].loc[data['Family_size'] > 1] = 0

train['Is_alone'].value_counts()

test['Is_alone'].value_counts()

train['Family_size'].value_counts()

test['Family_size'].value_counts()

# **We can now convert the Fare and Age into searate Bins**

for data in dataset:
    data['Fare_bin'] = pd.qcut(data['Fare'], 4)
    data['Age_bin'] = pd.cut(data['Age'].astype(int), 5 )

train.head()

test.head()

train.drop(['PassengerId','Name'], axis = 1, inplace = True)
test.drop(['Name'], axis = 1, inplace = True)

train.head()

test.head()

for data in dataset:
    data.drop(['Age','Fare'], axis = 1, inplace = True)

train.head()

test.head()

cat_attr = ['Sex','Title','Embarked','Fare_bin','Age_bin']

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
for data in dataset:
    for attr in cat_attr:
        data[attr] = encoder.fit_transform(data[attr])

train.head()

test.head()

# **Lets find the correlation of Survival with rest of the attributes(independent variables)**

attr = ['Pclass','Sex','Embarked','Family_size','Is_alone','Fare_bin','Age_bin','SibSp','Parch','Title']
for col in attr:
    print(train[['Survived',col]].groupby(col).mean())
    print()

# %% [code]
train.corr()

from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(train.drop(['Survived','Family_size'],axis = 1),train['Survived'],test_size = 0.2, shuffle = True,stratify = train['Survived'],random_state = 42)

X_train.shape

X_train.columns

X_val.shape

X_val.columns

# **Let us try out different classification algorithms and get the accuracy**
# **Importing required libraries from sklearn** 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

DTC = DecisionTreeClassifier()
RFC = RandomForestClassifier()
NBC = GaussianNB()
SVMC = SVC()
LRC = LogisticRegression()
XGBC = XGBClassifier()

MLAs = [DTC,RFC,NBC,SVMC,LRC,XGBC]

train.columns

test.columns

X_train.columns

X_val.columns

test.columns

for algo in MLAs:
    algo.fit(X_train,y_train)
    print("Algorithm: ",algo)
    print("Training accuracy: ", algo.score(X_train,y_train))
    print("validation accuracy: ", algo.score(X_val,y_val))
    print("**************************************\n")

# **Let us work on RandomForest Classifier a bit more and try to fine tune it**
RFC = RandomForestClassifier(n_estimators = 300, criterion = 'gini', max_depth = 4, random_state = 42)

RFC.fit(X_train,y_train)
print("Training accuracy: ", RFC.score(X_train,y_train))
print("validation accuracy: ", RFC.score(X_val,y_val))

///test.head()

X_train.head()

# **As we are not using Family_size feature in our training step, we will drop this attribute from our test set as well**

test.drop(['Family_size'], axis = 1, inplace = True)

# **Lets compare the attributes used for training the model and those present in the test set **

X_train.columns

test.columns

predictions = RFC.predict(test.drop(['PassengerId','Survived'],axis = 1))

len(predictions)

test.shape

submit['PassengerId'] = test['PassengerId']

submit['Survived'] = predictions

test.head()

test['Survived'] = predictions

test.head()

del submit

# %% [code]
submit = test[['PassengerId','Survived']]

submit.shape

submit['Survived'].value_counts(normalize=True)

submit.sample(10)

submit.to_csv("../working/submit.csv", index=False) ///
