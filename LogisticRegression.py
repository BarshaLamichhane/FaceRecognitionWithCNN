
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split


### collecting data
desired_width = 1000
titanic_data = pd.read_csv('../../Documents/train.csv')
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)


pd.set_option('display.max_columns',titanic_data.shape[1]) ## for printing all columns
pd.set_option('display.max_rows',15) # None will also print all rows; here I am printing 10 only# pd.set_option('display.max_rows', df.shape[0]+1) ## for printing all rows
print(titanic_data)

print("# of passengers in original data:" +str(len(titanic_data.index)))

## analyzing data
##i.e establishing the relationship between variables

# sns.countplot(x="Survived", data=titanic_data)
# plt.show()
# sns.countplot(x="Survived", hue="Sex", data=titanic_data)
# plt.show()
sns.countplot(x="Survived",hue="Pclass",data=titanic_data)
#plt.show()
titanic_data["Age"].plot.hist()
#plt.show()
titanic_data["Fare"].plot.hist(bins=20, figsize=(10,5))
#plt.show()
print(titanic_data.info())
sns.countplot(x="SibSp", data=titanic_data)
plt.ion()
plt.show()
plt.pause(1)
plt.close()

#######Data Wrangling
print(titanic_data.isnull())
print(titanic_data.isnull().sum())
sns.heatmap(titanic_data.isnull(), yticklabels=False )
plt.show()



sns.boxplot(x="Pclass", y="Age", data=titanic_data)


plt.ion()
plt.show()
plt.pause(1)
plt.close()

titanic_data.drop("Cabin", axis=1, inplace=True)
titanic_data.dropna(inplace=True)
print(titanic_data.info())
sns.heatmap(titanic_data.isnull(), yticklabels=False)
plt.ion()
plt.show()
plt.pause(1)
plt.close()



sex = pd.get_dummies(titanic_data['Sex'], drop_first=True) # dropping first so that if male=1 then male else female
embark = pd.get_dummies(titanic_data['Embarked'], drop_first=True)
Pcl = pd.get_dummies(titanic_data['Pclass'], drop_first=True)
print(Pcl)
titanic_data = pd.concat([titanic_data, sex, embark, Pcl], axis=1)

print(titanic_data.head(5))
titanic_data.drop(["Sex", "Embarked", "Pclass", "Name", "Ticket", "PassengerId"], axis=1, inplace=True)

print(titanic_data.head(5))


###### Train and Test Data

####Train Data

X = titanic_data.drop("Survived", axis=1)
y = titanic_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(max_iter=1000)
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import *

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test,predictions))







