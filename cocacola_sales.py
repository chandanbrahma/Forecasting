# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 09:09:36 2020

@author: admin
"""

import pandas as pd
data=pd.read_excel('E:\\submitted assignment\\forecasting\\Cocacola.xlsx')
data.head()
data.plot()


##rolling statistics for stationarity test
r_mean=data.rolling(window=4).mean()
r_std=data.rolling(window=4).std()

##plottinng rolling statistics
import matplotlib.pyplot as plt
original_data=plt.plot(data['Sales'],color='blue',label='original data')
mean=plt.plot(r_mean,color='red',label='r_mean')
std=plt.plot(r_std,color='black',label='r_std')
plt.legend(loc='best')
plt.title('r_mean and r_std')
plt.show()
plt.plot()
## as data is not stationary lets check p value and critical value for the same using Dickey fuller test

from statsmodels.tsa.stattools import adfuller
print('Dickey_fuller test')
df_test= adfuller(data['Sales'],autolag='AIC')
dfoutput= pd.Series(df_test[0:4],index=['test statistics','p-value','lags used','no od observation used'])
for key,value in df_test[4].items():
        dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

##p value is 0.996 and t statistics is 1.309. So we data is not at all stationary

## now estimating and eliminating trend
## taking logarithem transformation
import numpy as np
log_data= np.log(data['Sales'])
log_data.plot()

###still the has trend . lets apply movinng avgg techhnique for the same

moving_avg= log_data.rolling(window=4).mean()
plt.plot(log_data)
plt.plot(moving_avg)

##we can see still the rolling mean is not stationary.
##now getting the difference between the moving avg and the actual no of passengers,
difference= log_data-moving_avg
difference.head(10)
## now dropping the NA valuues
difference.dropna(inplace=True)
difference.head()

##building a model to check stationirty
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
     moving_avg=timeseries.rolling(window=4).mean()
     moving_std=timeseries.rolling(window=4).std()

    #Plot rolling statistics:
     original_data = plt.plot(timeseries, color='blue',label='orginal_data')
     mean = plt.plot(moving_avg, color='red', label='r_mean')
     std = plt.plot(moving_std, color='black', label = 'r_std')
     plt.legend(loc='best')
     plt.title('rolling Mean & standard Deviation')
     plt.show(block=False)
     
     
     print('Dickey_fuller test')
     df_test= adfuller(timeseries)
     dfoutput= pd.Series(df_test[0:4],index=['test statistics','p-value','lags used','no of observation used'])
     for key,value in df_test[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
    
    
test_stationarity(difference)
dfoutput


##exponential weighted moving avg  transformation for trend
expweighted_avg= log_data.ewm(halflife=4).mean()
plt.plot(log_data)
plt.plot(expweighted_avg, color= 'red')
##as we can see it has a upward trend and also  has seasonality so now differentiating exponential from the logarithemic
difference1= log_data-expweighted_avg
test_stationarity(difference1)

##now we can visualize that it is stationary with 90% confidence level and also with a p value of 0.076 which is prety good


##now shifting the value into timeseries for forecasting
shift_logdata= log_data-log_data.shift()
plt.plot(shift_logdata)
##dropping the nan values
shift_logdata.dropna(inplace=True)
test_stationarity(shift_logdata)


## our data is stationary with 90% confidence level with a better P value of 0.02
##forecasting
##ploting ACF and PACF for getting the value of p and q in ARIMA

from statsmodels.tsa.stattools import acf, pacf

acf = acf(shift_logdata, nlags=20)
pacf = pacf(shift_logdata, nlags=20, method='ols')

##acf plot 
plt.subplot(111) 
plt.plot(acf)
plt.axhline(y=0,linestyle='--',color='r')
plt.axhline(y=-1.96/np.sqrt(len(shift_logdata)),color='gray')
plt.axhline(y=1.96/np.sqrt(len(shift_logdata)),color='gray')
plt.title('ACF')

##pacf plot
plt.subplot(111)
plt.plot(pacf)
plt.axhline(y=0,linestyle='--',color='r')
plt.axhline(y=-1.96/np.sqrt(len(shift_logdata)),color='gray')
plt.axhline(y=1.96/np.sqrt(len(shift_logdata)),color='gray')
plt.title('PACF')
plt.tight_layout()

##hence we can take p=1 and q=1 
##loading ARIMA plot

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(log_data, order=(1, 1, 1))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(shift_logdata)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-shift_logdata)**2))

## so by comparing AR, MA and ARIMA model we can see ARIMA gives less RSS values 0.611. so it is bettter.

##now taking back the data to original  scale


predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues)
predictions_ARIMA_diff.head()

##now calculating the cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()

##now predicting the values
predictions_ARIMA_log = pd.Series(log_data.ix[0], index=log_data.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()

##now finally taking exponential so that data will come to the actual format where we started
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(data['Sales'])
plt.plot(predictions_ARIMA)

##so if we want to predict the values for the next 5 years
results_ARIMA.forecast(steps=20)
