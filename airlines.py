# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 09:09:36 2020

@author: admin
"""

import pandas as pd
data=pd.read_excel('E:\\assignment\\forecasting partial\\Airlines.xlsx')
data.head()
data['Month']=pd.to_datetime(data['Month'])
new_data=data.set_index(['Month'])
new_data.plot()

##now checking the statioanarity of the data by using 2 tests

##1.determing the rolling statistics
r_mean=new_data.rolling(window=12).mean()
r_std=new_data.rolling(window=12).std()

##plottinng rolling statistics
import matplotlib.pyplot as plt
original_data=plt.plot(new_data,color='blue',label='original data')
mean=plt.plot(r_mean,color='red',label='r_mean')
std=plt.plot(r_std,color='black',label='r_std')
plt.legend(loc='best')
plt.title('r_mean and r_std')
plt.show()
plt.plot()
## from the plot we can clearly visualize that or mean as well as standard deviation is not constant.
##hence our data is not stationary
##2.again lets see using Dickey-fuller test
from statsmodels.tsa.stattools import adfuller
print('Dickey_fuller test')
df_test= adfuller(new_data['Passengers'],autolag='AIC')
dfoutput= pd.Series(df_test[0:4],index=['test statistics','p-value','lags used','no od observation used'])
for key,value in df_test[4].items():
        dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

## as we can observe that the p-value is large and test statistics is also not smaller than the critical value hence rejecting null hypothesis.
## so data is not stationary


## so now estimating and eliminating trend
## taking logarithem transformation
import numpy as np
log_newdata= np.log(new_data)
log_newdata.plot()
## here we can visualize we are not able to totally eliminate the trend so now lets apply moving average technique to do for the same
moving_avg= log_newdata.rolling(window=12).mean()
plt.plot(log_newdata)
plt.plot(moving_avg)

##we can see still the rolling meaan is not stationary.
##now getting the difference between the moving avg and the actual no of passengers,
difference= log_newdata-moving_avg
difference.head(14)
## now dropping the NA valuues
difference.dropna(inplace=True)
difference.head()


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
     moving_avg=timeseries.rolling(window=12).mean()
     moving_std=timeseries.rolling(window=12).std()

    #Plot rolling statistics:
     original_data = plt.plot(timeseries, color='blue',label='orginal_data')
     mean = plt.plot(moving_avg, color='red', label='r_mean')
     std = plt.plot(moving_std, color='black', label = 'r_std')
     plt.legend(loc='best')
     plt.title('rolling Mean & standard Deviation')
     plt.show(block=False)
     
     
     print('Dickey_fuller test')
     df_test= adfuller(timeseries['Passengers'],autolag='AIC')
     dfoutput= pd.Series(df_test[0:4],index=['test statistics','p-value','lags used','no of observation used'])
     for key,value in df_test[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
    
    
test_stationarity(difference)
dfoutput


## to calculate trend inside thhe timeseries lets go for an exponential weighted moving avg  transformation
expweighted_avg= log_newdata.ewm(halflife=12).mean()
plt.plot(log_newdata)
plt.plot(expweighted_avg, color= 'red')
##as we can see it has a upward trend and also  has seasonality so now differentiating exponential from the logarithemic
difference1= log_newdata-expweighted_avg
test_stationarity(difference1)

##now we can visualize that it is stationary


##now shifting the value into timeseries for forecasting
shift_lognewdata= log_newdata-log_newdata.shift()
plt.plot(shift_lognewdata)
##dropping the nan values
shift_lognewdata.dropna(inplace=True)
test_stationarity(shift_lognewdata)


##We can see that the mean and std variations have small variations with time
##Also, the Dickey-Fuller test statistic is less than the 10% critical value, thus the TS is stationary with 90% confidence and rejecting the null hypothesis

##now going for the decomposation
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(log_newdata)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(511)
plt.plot(log_newdata, label='Original')
plt.legend(loc='best')
plt.subplot(512)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(513)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(514)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

#now we can see that the trend, seasonality as well as residual or irregularity are separated out from data.
#now lets check wheather the residual or the noise are stationarity or not

residual_data= residual
residual_data.dropna(inplace=True)
test_stationarity(residual_data)

## from the Dickey fuller test it can seen that the noise is less than the critical value. so its stationary
##now we move to forecasting a timeseries


##ploting ACF and PACF for getting the value of p and q in ARIMA

from statsmodels.tsa.stattools import acf, pacf

acf = acf(shift_lognewdata, nlags=30)
pacf = pacf(shift_lognewdata, nlags=30, method='ols')

##acf plot 
plt.subplot(111) 
plt.plot(acf)
plt.axhline(y=0,linestyle='--',color='red')
plt.axhline(y=-1.96/np.sqrt(len(shift_lognewdata)),color='gray')
plt.axhline(y=1.96/np.sqrt(len(shift_lognewdata)),color='gray')
plt.title('ACF')
help(plt.axhline)
##pacf plot
plt.subplot(111)
plt.plot(pacf)
plt.axhline(y=0,linestyle='--',color='r')
plt.axhline(y=-1.96/np.sqrt(len(shift_lognewdata)),color='gray')
plt.axhline(y=1.96/np.sqrt(len(shift_lognewdata)),color='gray')
plt.title('PACF')
plt.tight_layout()

##hence we can take p=2 and q=2 
##loading ARIMA plot

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(log_newdata, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(shift_lognewdata)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-shift_lognewdata['Passengers'])**2))

## so by comparing AR, MA and ARIMA model we can see ARIMA gives less RSS values. so it is bettter.
##so now taking back the data to original  scale

##saving the fitted value into a series format
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues)
predictions_ARIMA_diff.head()

##now calculating thhe cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()

##now predicting the values
predictions_ARIMA_log = pd.Series(log_newdata['Passengers'].ix[0], index=log_newdata.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

##now finally taking exponential so that data will come to the actual format where we started
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(new_data)
plt.plot(predictions_ARIMA)

##so if we want to predict the values for the next 2 years
results_ARIMA.forecast(steps=24)
