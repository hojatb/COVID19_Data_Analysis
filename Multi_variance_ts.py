# -*- coding: utf-8 -*-
"""
Created on Wed May  6 07:18:06 2020

@author: hojat behrooz
"""
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)  

from statsmodels.tsa.vector_ar.var_model import VAR 
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np
import pandas as pd
import os
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from  input1 import input_data , input_single ,average_corr ,read_weights,\
                    delete_zeros , moving_average,Umoving_average,corr_matrix,\
                    input_consider,dif_m,wm,mean_t,ma_m,Uma_m,dif, dif_m,\
                    integ_m,integ_level,logn_f,exp_f,normalize,adf_test, make_weights
    
datatype=['US_DEATH_CASES',  
'US_CONFIRMED_CASES',
'GLOBAL_DEATH_CASES',
'GLOBAL_CONFRIRMED_CASES']
US_DEATH=0  
US_CONFIRMED =1  
GLOBAL_DEATH=2
GLOBAL_CONFRIRMED=3
np.seterr(divide='ignore')

                   
      
epi= 1.0/10000 # precious data                            
Alpha=.05

dataset_type,flatten,name,Y,last_date,first_date =input_data()
# making a data frame of input data with date index and column name of regions
#delete dataset with finall value less than 1/10,000th total cases
valuable=(np.array(Y)[:,-1:]>epi*np.sum(np.array(Y)[:,-1:])).flatten()
most_valuable=np.array(sorted(zip(np.array(Y)[:,-1:].flatten(),name),reverse=True))
y=[Y[i] for i in range(len(Y)) if valuable[i]]
namep=[name[i] for i in range(len(name)) if valuable[i]]
d=np.array(y,dtype='float64')
normal_table=d[:,-1:]
norm=pd.DataFrame({'NAME':namep,'NORMAL':normal_table[:,0]})
norm= norm.set_index('NAME')

d=d.transpose()
normal_table=normal_table.transpose()

# flatten based on the moving average and flatten parameter
#d=ma_m(d, flatten)
#normalize on maximum of each dataset multiply by 100,000
#d=(d/normal_table)*100000.
#diffrential of the data set up to 7 times
d_m=np.zeros((7,d.shape[0],d.shape[1]),dtype=float)
d_m[0]=d

for i in range(1,7):
    d_m[i]=dif_m(d_m[i-1])
    
#checking stationarity
lenc=len(namep)
p_m=np.zeros((7,lenc),dtype='float')
p0=np.zeros(lenc)

print("stationary test for diffrent defrentials(%d) regions"%(lenc))
for j in range (7):
   # print(">>>>>>>>\n",end=" ",flush=True,sep="")
    for i in range(lenc):
       # print("%3d\r"%(i),end=" ",flush=True)
        p_m[j,i]=adf_test(d_m[j,:,i],Alpha)
    print(">>d%d statinary needs:%d"%(j,np.sum(np.array(p_m[j]>Alpha))),flush=True)

difrant=int(input_single("Enter level of diffrentiation?1-7", 7, 1))
l=pd.DataFrame({'NAME': namep, 'P0': p_m[0], 'P1': p_m[1],"p2":p_m[2],"P3":p_m[3]
                ,"p4":p_m[4],"P5":p_m[5],"p6":p_m[6]}) 
#selecting the diffretiation depth base on the result
data=d_m[difrant]
p=p_m[difrant]
data_source=pd.DataFrame(d_m[1],columns=namep)
data = pd.DataFrame(data, columns=namep)
dti = pd.date_range(start=first_date, periods=len(data), freq='D')
data=data.set_index(dti)
data_source=data_source.set_index(dti)
print ("the regions deleted from list because of their P_value more than:",Alpha)
print([(namep[i],np.round(p[i],2)) for i in range(len(namep)) if p[i]>Alpha])
namep= [namep[i] for i in range(len(namep)) if p[i]<Alpha]
data=data[namep]
data_source=data_source[namep]
data_copy=data.copy()
#creating the train and validation set
lenc=len(data)
predict_days=input_single("please enter  prediction days 0-14(0 for exit):",14,0)
consider= input_consider(namep)
#weights=read_weights(namep,dataset_type) 
weights=make_weights(namep,dataset_type)
if(predict_days==0): sys.exit(0)
print("lag_test=",end=" ")
qq=[]
test_len=7
rrange=lenc-test_len-6
#if(lenc<90):rrange=lenc-10
for q in range(1,rrange):
    train = data[:-test_len]
    valid = data[-test_len:]    

    lag=len(valid)
    model = VAR(endog=train)
#ic{'aic', 'fpe', 'hqic', 'bic', None}        
    model_fit = model.fit(q)
    lag_order = model_fit.k_ar
    # make prediction on validation
    prediction = model_fit.forecast(train[-lag_order:].values, lag)                
    # back ward integration to main data set
    data[-lag:]=prediction     
    data_predicted=integ_level(data.values,difrant-1)  
#    data_predicted=(data_predicted/100000)*norm.loc[namep]['NORMAL'].values    
#    data_predicted=Uma_m(data_predicted,flatten)

    data_predicted[data_predicted<0]=0                   
   
    data_predicted=pd.DataFrame(index=dti,data=data_predicted,columns=namep)
    mae=np.zeros(len(namep))
    WME=np.zeros(len(namep))        
    for j,i in enumerate(namep):
        mae[j]=mean_absolute_error(data_predicted[i][-test_len:],                                    data_source[i][-test_len:])*weights[j]
    WMES=np.sum(mae)
    qq.append(WMES)
    print("[%d>>%.1f]"%(q,WMES),end=" ")

pd.DataFrame(qq).plot()
#qqq=[qq[i]+np.log(10*i+1) for i in range(len(qq))]
best_lag=np.argmin(qq)
print ("\nminimum WMAE product weight:%.2f on lag:%d"%(qq[best_lag],best_lag))
#https://www.kaggle.com/c/covid19-global-forecasting-week-5/overview/evaluation
# for quantile to be predicted, e.g., one of [0.50] the score formula
# simplify as score =(1/2N)* Sum(wf* abs(delta(y)))
if(dataset_type in [US_DEATH,GLOBAL_DEATH]):print("Fatalities score =",10*qq[best_lag]/(2*len(namep)))
else: print("score =",qq[best_lag]/(2*len(namep)))
################################################################################        
l2=pd.DataFrame({'NAME': namep, 
                  'MAE': mae,'CASES':data_source.max(),'WME':WMES})
    
combin=pd.DataFrame({'REAL':data_source[namep[consider]],
                      'PREDICT':data_predicted[namep[consider]]})
combin=combin[combin.REAL>0]
ax1=combin.plot(label='predict and real data for %s with WMAE:%.3f'
                %(namep[consider],WME[consider])) 
ax1.grid(True)
last_date=last_date.replace('/', '_')
combin.to_csv('results/Model_test_%2d_DAYS_%s_FOR_%s_ON_%s.csv'\
              %(lag,datatype[dataset_type],namep[consider],last_date))
l2.to_csv('results/Model_test_err_%2d_DAYS_%s_FOR_%s.csv'\
              %(lag,datatype[dataset_type],last_date),index=False)    
lag=int(predict_days)
data=data_copy
model = VAR(endog=data)   
model_fit = model.fit(best_lag)
lag_order = model_fit.k_ar
# make prediction on validation
prediction = model_fit.forecast(data[-lag_order:].values, int(lag))
data_ex=np.zeros(shape=(data.shape[0]+lag,data.shape[1]))
data_ex[0:data.shape[0],:]=data
data_ex[-lag:,:]=prediction

############### return the converted function to original
data_predicted=integ_level(data_ex,difrant-1) 
#data_predicted=(data_predicted/100000)*norm.loc[namep]['NORMAL'].values
#data_predicted=Uma_m(data_predicted,flatten)

data_predicted[data_predicted<0]=0 
pred = pd.DataFrame(index=range(0,len(data_ex)),data=data_predicted,columns=namep) 
dti = pd.date_range(start=first_date, periods=len(data_ex), freq='D')
pred=pred.set_index(dti)
max1d=pred.max()
max=pred.sum()      
for i, e in enumerate(pred[namep[consider]]):
    if(e>(max1d[consider]/100.0)):break
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_title("%s %s Prediction for %d Days"%(namep[consider],datatype[dataset_type],lag))
ax1.set_xlabel('DATE (day)')
ax1.set_ylabel('Number of Daily Cases', color=color)
ax1.plot(pred[namep[consider]][i:], color=color,
         label='Simple Daily %s %s'%(namep[consider],datatype[dataset_type]))
ax1.plot(pred[namep[consider]][-lag:],  lw=20, c='yellow', zorder=-1)
# Change major ticks to show every 20.
ax1.tick_params(axis='y', labelcolor=color)    
ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.yaxis.set_major_locator(MultipleLocator(int(10**int(np.log10(np.abs(max1d[consider])+1)+1)
                                                /10.)))
ax1.xaxis.set_tick_params(rotation=90)
ax1.yaxis.set_minor_locator(AutoMinorLocator(4))
#    ax1.xaxis.set_minor_locator(AutoMinorLocator(1))
ax1.grid(which='major', color='#CCCCCC', linestyle='--')    
 
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Number of cumlative cases', color=color)  # we already handled the x-label with ax1
ax2.plot(pred[namep[consider]].cumsum()[i:], color=color,
         label='Cumlative Daily %s %s'%(namep[consider],datatype[dataset_type]))
ax2.plot(pred[namep[consider]].cumsum()[-lag:],  lw=20, c='yellow', zorder=-2)

ax2.tick_params(axis='y', labelcolor=color)
ax2.xaxis.set_major_locator(MultipleLocator(1))
ax2.yaxis.set_major_locator(MultipleLocator(int(10**int(np.log10(np.abs(max[consider])+1)+1)
                                                /10.)))
#    ax1.xaxis.set_minor_locator(AutoMinorLocator(1))
ax2.xaxis.set_tick_params(rotation=90)
# Change minor ticks to show every 5. (20/4 = 5)
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
ax1.legend(loc=3, borderaxespad=1.).set_zorder(2)
ax2.legend(loc=4, borderaxespad=1.).set_zorder(2)

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax2.grid(which='major', color='#CCCCCC', linestyle='--')    
plt.show()
last_date=last_date.replace('/', '_')
pred.to_csv('results/predict_%2d_DAYS_%s_FOR_%s.csv'%(lag,datatype[dataset_type],last_date))




