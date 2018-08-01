# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 13:09:23 2018

@author: adabhoiwala
"""

#---------------- Time Series Forecasting------------------------
#--- importing libraries-------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6

from statsmodels.tsa.stattools import adfuller

from pandas import Series


#--- importing the dataset-------------

data = pd.read_csv('C:/Users/adabhoiwala/Documents/AafrinFiles/seattleWeather_1948-2017.csv')

data_1 = list(data_1)

data_series = pd.Series(data_1, index=pd.to_datetime(data['DATE'],format='%Y-%m-%d'))

#---- checking datatypes-------
data_series.index
data_series.dtypes

#--- resampling into quarters---------------
data_q = data_series.resample('Q').mean()

#----Plotting the data---------
plt.plot(data_q)

#----- Function Checking Stationary of a Time Series (Constant Mean, Constant Variance )----------

def test_stationary(timeseries):
    #---- Determinig rolling statistics-------
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()
    
    #------Plotting above rolling statistics-------
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(timeseries, color='red', label='Rolling Mean')
    std = plt.plot(timeseries, color='green', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean and Standard Deviation')
    plt.show()
    

#--- The above test shows that the data has seasonality by visualizing it-----


 #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


test_stationary(data_q)

#---- from the above statistics, it shows that time series is 95% confidence that
# it is stationary, as test-statistics is less than critical-value

#---- to make it 99% stationary, we will do some transformation------
#--- first difference--------------
ts_first_difference = data_q.PRCP - data_q.PRCP.shift(1) 
ts_first_difference_no_missing = ts_first_difference.dropna()

#---- sesonal difference-----
ts_seasonal_difference = data_q - data_q.shift(12) 
ts_seasonal_difference_no_missing = ts_seasonal_difference.dropna()

print(ts_seasonal_difference_no_missing.Index)

#---- log difference--------
ts_log = np.log(data_q.PRCP)


#---- going forward with the seasonal difference------
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(ts_seasonal_difference_no_missing)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_seasonal_difference_no_missing, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()


#--- Checking stationarity with plots( trend, seasonal, and residual)----
ts_log_decompose = trend
ts_log_decompose.dropna(inplace=True)
test_stationary(ts_log_decompose)


# we can move forward with forecasting this timeseries........

#--- plotting for ACF---------
 from statsmodels.graphics.tsaplots import plot_acf
 plot_acf(ts_seasonal_difference_no_missing, lags=100)


#----- plotting for PACF----- 

from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(ts_seasonal_difference_no_missing, lags=20)


#---- Modeling Time Series-----
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(ts_seasonal_difference_no_missing, order=(0,1,0))

#--- fitting the model------

model_fit = model.fit()

model_fit.summary()


plt.plot(model_fit.fittedvalues, color='red')

#--- checking residual to check the accuracy of this model----
from pandas import DataFrame
residuals = DataFrame(model_fit.resid)
residuals.plot()

#--- checking the normality of the residuals---
residuals.plot(kind='kde')

residuals.describe()


#--- Prediction---------
 prediction = model_fit.predict(260, 266, dynamic= True)
    

#-- -predicting next day forecast----20
forecast = model_fit.forecast()[0]

plt.plot(forecast)


#--- load original data set----

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

#---- inverting the seasonal difference and converting the  value back to the original scal

forecast = inverse_difference(ts_seasonal_difference_no_missing, forecast, 1)
print(forecast)

#--- it will rain the next day based on the PRCP value------
