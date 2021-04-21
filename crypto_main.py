#Libraries

import numpy as np
import pandas as pd
import datetime as dt

from sklearn.model_selection import train_test_split

#pdata
df= pd.read_csv('doge.csv')

#set index=date
df= df.set_index(pd.DatetimeIndex(df['Date'].values))

#Get close price
df = df[['Close']]

#Variable to store number of days into the predicted future
prediction_days = 1

#Column to store predicted price
df['Prediction'] = df[['Close']].shift(-prediction_days)

#Create independent dataset
X = np.array(df.drop(['Prediction'],1))

#Remove n+1 rows of data , n=prediction prediction_days
X = X[:len(df)- prediction_days - 1]

#Creating dependent dataset 'Y'
y = np.array(df['Prediction'])

#All values of y except for last n+1 rows
y = y[:- prediction_days -1]

#Split data into training and testing dataset (80-20)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#ML model
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators = 2, random_state = 587)
forest.fit(x_train, y_train)

#Get validation data
#Varible to store all except last n rows of dataset
temp_df= df[:-prediction_days]

#Variable to store independent price Values
x_val = temp_df.tail(1)['Close'][0]

prediction = forest.predict([[x_val]])

#Print the price of Dogecoin for next n days
print('The predicted price of Dogecoin was', prediction)

#Actual Values
print('The actual price of Dogecoin was', temp_df.tail(1)['Prediction'][0])

#Accuracy
print("Prediction Accuracy:",forest.score(x_test, y_test))
