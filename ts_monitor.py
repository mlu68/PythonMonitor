
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
import itertools

import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

get_ipython().magic('matplotlib inline')
plt.style.use('bmh')


# # Generate time series

# In[2]:

from statsmodels.tsa.arima_process import arma_generate_sample

nsample = 100
x1 = np.linspace(0, 100, nsample)

arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])
arparams = np.r_[1, -arparams]
maparam = np.r_[1, maparams]

y = arma_generate_sample(arparams, maparams, nsample)
fig, ax = plt.subplots(figsize=(15,8))
ax.plot(x1, y, label="Data") 


# # Decomposition plot

# In[3]:

serie =  pd.DataFrame(y)
serie.index = pd.DatetimeIndex(freq='w', start=0, periods=nsample)

decomposition = sm.tsa.seasonal_decompose(serie, model='additive')

fig = decomposition.plot()
fig.set_figwidth(12)
fig.set_figheight(8)
fig.suptitle('Decomposition')
plt.show()


# # ARIMA modeling

# In[4]:

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, d and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


# ## Select model

# In[5]:

import sys
warnings.filterwarnings("ignore") # specify to ignore warning messages

best_aic = np.inf
best_pdq = None
best_seasonal_pdq = None
temp_model = None

for param in pdq:

    for param_seasonal in seasonal_pdq:
        
        try:
            temp_model = sm.tsa.statespace.SARIMAX(serie,
                                             order = param,
                                             seasonal_order = param_seasonal,
                                             enforce_stationarity=True,
                                             enforce_invertibility=True)

            results = temp_model.fit()

            #print("SARIMAX{}x{}12 - AIC:{}".format(param, param_seasonal, results.aic))
            if results.aic < best_aic:
                best_aic = results.aic
                best_pdq = param
                best_seasonal_pdq = param_seasonal
        except:
            #print("Unexpected error:", sys.exc_info()[0])
            continue
print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))


# ## Model

# In[6]:

best_model = sm.tsa.statespace.SARIMAX(serie,
                                      order=best_pdq,
                                      seasonal_order=best_seasonal_pdq,
                                      enforce_stationarity=True,
                                      enforce_invertibility=True)

best_results = best_model.fit()

print(best_results.summary().tables[0])
print(best_results.summary().tables[1])


# In[7]:

predicted = best_results.get_prediction(start=0, dynamic=False, full_results=True)


# In[8]:

plt.figure(figsize=(20,15))
plt.plot(predicted.predicted_mean, color = 'b', label = 'Smoothed')
plt.plot(serie, color = 'r', label = 'Original data')
plt.legend(loc='upper left')


# ## Errors

# In[9]:

best_results.plot_diagnostics(lags=30, figsize=(16,12))


# In[10]:

errors = pd.DataFrame(columns=['SERIE','PREDICTED'])
errors['SERIE'] = np.reshape(serie[0], len(serie[0]))
errors['PREDICTED'] = predicted.predicted_mean
errors['ERRORS'] = errors['SERIE'] - errors['PREDICTED']


# # Control

# ## Control limits

# In[13]:

dev = errors['ERRORS'].std()


# In[59]:

nsgimas = 2

up = errors['PREDICTED'] + dev * nsgimas
low = errors['PREDICTED'] - dev * nsgimas


# In[60]:

ooc = np.where((errors['SERIE'] > up) | (errors['SERIE'] < low))
print(ooc)


# In[61]:

plt.figure(figsize=(25,15))

errors['SERIE'].iloc[ooc]

plt.figure(figsize=(15,12))
plt.plot(up, color = 'r', label = 'upper')
plt.plot(low, color = 'r', label = 'lower')
plt.plot(errors['SERIE'])

plt.plot(errors.index[ooc], errors['SERIE'].iloc[ooc], 'ro', markersize  = 10, label = "out of control")
plt.legend(loc='upper left', fontsize = 'x-large')

