import pip as a
import numpy as np

import sys

import pandas as pd
desired_width = 1000

# print("ppppp", pd.DataFrame)

dataset = pd.read_csv('../../../Documents/weatherHistory.csv')
dataset.fillna(0, inplace=True)
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)


pd.set_option('display.max_columns',dataset.shape[1]) ## for printing all columns
pd.set_option('display.max_rows',None) # None will also print all rows; here I am printing 10 only# pd.set_option('display.max_rows', df.shape[0]+1) ## for printing all rows
print(dataset)
print("max temperature is", dataset['Temperature'].max())
print(dataset['EST'][dataset['Events'] == 'Rain'])

#print("Mean of wind Speed is: ", meanOfWindSpeed)



meanOfWindSpeed = dataset['WindSpeedMPH'].mean()
print("Mean of wind Speed is: ", meanOfWindSpeed)

# weatherData ={
#     'day':['sun','Mon','Tues','Wed'],
#     'temperatureInDegressCelsius':[32,35,28,24],
#     'windspeed':[6,7,2,7],
#     'event': ['Rain','Sunny','Snow','hello']
# }
#
# df = pd.DataFrame(weatherData)
# print(df)

print("shape of dataset is",dataset.shape) # here first means no of rows and second means no of column

print("first five rows are: \n", dataset.head(5))

print("columns are: ",dataset.columns)
print(dataset.describe())
