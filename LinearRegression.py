import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

desired_width = 1000

df = pd.read_csv('../../Documents/homeprices.csv')
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
# print(type(df.area))
print(df)

# for i in range(0, len(df.area)):
#     sum = sum+df.area[i]
#
# print("sum of all areas are:", str(sum))
# arrangeInArray = np.array(df.area[0:])
# reshapeArray = np.reshape(arrangeInArray, (1, -1))
# print(reshapeArray)

regm = linear_model.LinearRegression()
regm.fit(df[['area']], df.price)  # fitting means training linear regression model using the data frame
plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, regm.predict(df[['area']]), color='blue')  ## predicted plot
plt.ion()
plt.show()
plt.pause(0.1)
plt.close()
#
# print(reg.predict(df[['area']]))  # here 3300 is dependent variable
print('Mean Absolute Error:', metrics.mean_absolute_error(df.price, regm.predict(df[['area']])))
print('Mean Squared Error:', metrics.mean_squared_error(df.price, regm.predict(df[['area']])))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(df.price, regm.predict(df[['area']]))))
dfff = pd.DataFrame({'Actual': df.price, 'predicted': regm.predict(df[['area']])})
print(dfff)

reg = linear_model.LinearRegression()  # create object for linear regression
x = df[['area']]
y = df[['price']]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
reg.fit(X_train, y_train)  # fitting means training linear regression model using the data frame
y_predict = reg.predict(X_test)

# reg.fit(df[['area']], df.price)  # fitting means training linear regression model using the data frame

# print(reg.predict(df[['area']]))  # here 3300 is dependent variable
plt.xlabel('area(sqr ft)')
plt.ylabel('price(US$)')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, df.price, color='black')  ## actual plot
# plt.plot(df.area, reg.predict(df[['area']]), color='blue') ## predicted plot
# plt.plot(df.area, reg.predict(x), color='blue') ## predicted plot
# error or accuracy calculation


plt.ion()
plt.show()
plt.pause(0.1)
plt.close()


plt.scatter(X_test, y_test, color='grey', marker='+')
plt.plot(X_test, y_predict, color='blue')
plt.ion()
plt.show()
plt.pause(0.1)
plt.close()

plt.scatter(df.area, df.price, color='grey', marker='+')
plt.plot(X_test, y_predict, color='blue')
plt.ion()
plt.show()
plt.pause(0.1)
plt.close()
print(X_test)
print(X_train)
print("predicted datasets are:", y_predict)
print("test y_set are:", y_test)
y_test = np.array(y_test[0:])
print("array converted y is:", y_test)
print("flattenede array is:", y_test.reshape(1, -1))
dff = pd.DataFrame({'Actual': y_test.flatten(), 'predicted': y_predict.flatten()})
print(dff)

print("accuracy is:", metrics.f1_score(y_test,y_predict))

### avualting metrices for linear regression
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))

### at first it calculates m, finds b and x is given and will predict y
# m = reg.coef_
# b = reg.intercept_
# x = 5000
# print(reg.coef_) #this prints slope at particular value of y and x; slope is also known as gradient
# print(reg.intercept_) # prints y intercept which is b
# y = m*x+b
# print(y)
#######################################################
