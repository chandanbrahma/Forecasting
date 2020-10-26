
## importing the dataset
import pandas as pd

data=pd.read_csv('E:\\assignment\\forecasting partial\\PlasticSales.csv')

data.head()
data.describe()
data['Month']=pd.to_datetime(data['Month'])

new_data=data.set_index(['Month'])

new_data.plot()

##from the plot we can visualize that the data have seasonality as well as trend.
##now checking the stationarity of the data by using 2 tests i.e rolling statistics and dickeyfuller test

##Test1.determing the rolling statistics
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

## from the plot we can clearly visualize that  mean as well as standard deviation is not constant.
##hence our data is not stationary

##Test2.again lets see using Dickey-fuller test
from statsmodels.tsa.stattools import adfuller
print('Dickey_fuller test')
df_test= adfuller(new_data['Sales'],autolag='AIC')
dfoutput= pd.Series(df_test[0:4],index=['test statistics','p-value','lags used','no od observation used'])
for key,value in df_test[4].items():
        dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

## In the dickey-fuller test we know that we take the null hypothesis as Ho= The data is not stationary 
## as we can observe that the p-value is large i.e not smaller than the critical value hence accepting null hypothesis.
## so data is not stationary

##Futher we know that main reason for the stationary is trend and seasonality.If we eliminate them, data will become stationary
## now estimating and eliminating trend
## taking logarithem transformation(approach1)
import numpy as np
log_newdata= np.log(new_data)
log_newdata.plot()

## here we can visualize that the logarothemic transform itself not able to eliminate the trend so now lets go for other approaches
##apply moving average technique to do for the same(approach2)
moving_avg= log_newdata.rolling(window=12).mean()
plt.plot(log_newdata)
plt.plot(moving_avg)

##we can visualize that still the rolling mean is not stationary.

##approach 3
##now lets combine both the approaches and try wheather their differences will make any sence interms  of getting our result,
difference= log_newdata-moving_avg
difference.head(14)

## We got consecutive 11 NAN values,lets us drop those values
difference.dropna(inplace=True)
difference.head()


## from now on instead of tryting different approaches and applying thr rolling statistics on eeach and every approach,
# lets us take an uniform timeseries and apply both the tests on them and create a user denined function so that,
# next time onwords we can directly apply on the approaches
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
     df_test= adfuller(timeseries['Sales'],autolag='AIC')
     dfoutput= pd.Series(df_test[0:4],index=['test statistics','p-value','lags used','no of observation used'])
     for key,value in df_test[4].items():
        dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
    
dfoutput



## now applying on approach 3
test_stationarity(difference)

## so we got a stationary time series with a very less critical value.
## but one maajor issue is that in our previous approach, we need to strictly define the time series as in this case we take the yeaarly average.
## So in other practical situations where a particular number cant be defined, weighted average is taken.
##for that purpose exponential weighted average appoach is mostly used
##lets apply the same to our dataset and try to get the result.
 
##approach 4
expweighted_avg= log_newdata.ewm(halflife=12).mean()
plt.plot(log_newdata)
plt.plot(expweighted_avg, color= 'red')

##as we can see it has a upward trend and also  has seasonality so now differentiating exponential from the logarithemic and eliminating trend
difference1= log_newdata-expweighted_avg
test_stationarity(difference1)

##So we can visualize that it is stationary as well a very less p value, also better from the previous approach as here we do not have any missing values

## In the avove approaches we removed the trends from the data, but elimination on only trend is not enough to make a data stationary,
## We also have to elimintate the seasonality, So lets go for elimination of trend and seasonality

##Approach 5(By using differencing and decompoosition)

## Eliminating trend and seasonality by Differencing i.e by taking difference of a particular approach at an instance with respect to its previous instanace
shift_lognewdata= log_newdata-log_newdata.shift()
plt.plot(shift_lognewdata)
##dropping the nan values
shift_lognewdata.dropna(inplace=True)
test_stationarity(shift_lognewdata)


##We can see that the mean and std variations have small variations with time
## But P value is not that much significant so lets go for the othher approach i.e decomposition

##In decomposation both the trend and seasonality are removed from the data and the rest are returned.
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

print('Dickey_fuller test')
df_test= adfuller(residual_data,autolag='AIC')
dfoutput= pd.Series(df_test[0:4],index=['test statistics','p-value','lags used','no of observation used'])
for key,value in df_test[4].items():
    dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

## from the Dickey fuller test it can seen that The P value is less than the critical value. so its stationary rejecting the null hypothesis



##now we move to forecasting as we got a stationary data
## for forcasting the timeseries we will use the ARIMA model i.e  Auto-Regressive Integrated Moving Averages

##ploting ACF and PACF for getting the value of p and q to apply ARIMA

from statsmodels.tsa.stattools import acf, pacf

acf = acf(shift_lognewdata, nlags=20)
pacf = pacf(shift_lognewdata, nlags=20, method='ols')

##acf plot 
plt.subplot(111) 
plt.plot(acf)
plt.axhline(y=0,linestyle='--',color='red')
plt.axhline(y=-1.96/np.sqrt(len(shift_lognewdata)),color='gray')
plt.axhline(y=1.96/np.sqrt(len(shift_lognewdata)),color='gray')
plt.title('ACF')

##pacf plot
plt.subplot(111)
plt.plot(pacf)
plt.axhline(y=0,linestyle='--',color='r')
plt.axhline(y=-1.96/np.sqrt(len(shift_lognewdata)),color='gray')
plt.axhline(y=1.96/np.sqrt(len(shift_lognewdata)),color='gray')
plt.title('PACF')
plt.tight_layout()

##hence we can take p=1 and q=2 
##loading ARIMA plot

from statsmodels.tsa.arima_model import ARIMA
model = ARIMA(log_newdata, order=(0, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(shift_lognewdata)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-shift_lognewdata['Sales'])**2))

## so by comparing AR, MA and ARIMA model we can see AR gives less RSS values. so it is bettter.
##so now taking back the data to original  scale

##saving the fitted value into a series format
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues)
predictions_ARIMA_diff.head()

##now calculating the cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_diff_cumsum.head()

##now predicting the values
predictions_ARIMA_log = pd.Series(log_newdata.iloc[0], index=log_newdata.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.dropna(inplace=True)
predictions_ARIMA_log.head()

##now finally taking exponential so that data will come to the actual format where we started
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(new_data)
plt.plot(predictions_ARIMA)

##so if we want to predict the values for the next 2 years
results_ARIMA.forecast(steps=24)
