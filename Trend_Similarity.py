# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:48:43 2020

@author: hojat behrooz
"""
# fit a polynomial to a vector of the data
from scipy.stats import pearsonr

import os
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as mdates
from  input1 import input_data , input_single ,average_corr ,\
                    delete_zeros , moving_average,corr_matrix,input_consider
from minepy import MINE  
#recomendation of alpha for Maximal Information Coefficient    
# according to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5893960/              
def alphac(n):
    if(n < 25):return(0.85)
    if(n < 50):return(0.8)
    if(n < 250):return(0.75)
    if(n < 500):return(	0.65)
    if(n < 1000):return(0.6)
    if(n < 2500):return(0.55)
    if(n < 5000):return(0.5)
    if(n < 10000):return(0.45)
    if(n < 40000):return(0.4)
    return(.4)
    

US_DEATH=0  
US_CONFIRMED =1  
GLOBAL_DEATH=2
GLOBAL_CONFRIRMED=3
datatype=['US_DEATH_CASES',  
'US_CONFIRMED_CASES',
'GLOBAL_DEATH_CASES',
'GLOBAL_CONFRIRMED_CASES']
# data sorces imported from https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data   
url_confirmed_global = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_death_global = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
url_death_US ='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
url_confirmed_US ='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
 
#minimum amount that ignore after normalization data this only consider for 
# starting and ending of the dataset.
epsilon = 0.005
#minimum percent of the total cases for one region to be considered precious data it 
# is compare he maximmum cases over the total case in all region and if this is bigger than
# epi it means this data set is precious
epi = 0.0001


NAME=[]
# throught the inpu_data() severla data asked from user as:
#     a catagory for study choosed and return as dataset_type it is between 
# US_DEATH=0  
# US_CONFIRMED =1  
# GLOBAL_DEATH=2
# GLOBAL_CONFRIRMED=3
# based on selected catagory the proper data reade from correspanding dataset cvs file and 
# from the database a list of the name of territiries created (NAME). the first date and 
# the last date of existed data also defined. the whole daily data returns in Y
# the list of the territories view to the user and asks for selecting one of them to considered
# for further more study. the row of the user input returns as considered
# then from user ask a flatten number between 1-7. this number use for calculating the 
# moving averge. after that asking from user to input a number between 0-1 . this number
# use for finding the regions with similarity factors more than that. this item returns as 
# dimilarity. 


#to ignore warinig message rise in coefficent function calculation
np.seterr(divide='ignore', invalid='ignore')

dataset_type,flatten,NAMET,Y,last_date,first_date =input_data()
 

list_trimed =[]
list_max1 =[]
list_max1a =[]
list_maxT =[]
list_complete =[]
list_precious=[]
list_not_trimed=[]
list_starts=[]
list_ends=[]
NAME=[]
raw_data=[]
data= np.array(Y)
num_region = len(data)
# calculating the total number of casea in whole domain from last row 
#sumup=np.sum(data[-1:,-1:])
sumup=np.sum(data[:,-1:])
row=0
while( row<num_region):
    complete_form = 0
    precious=0
    y = data[row]
    y_diff=y.astype('float64')
    ysum=y_diff[-1:].sum()
#calculate diffrential data
    for ii in range(1,len(y_diff)):
       y_diff[ii] = y[ii]-y[ii-1] 
       if y_diff[ii] < 0 :y_diff[ii]=0
    y_diff_origin =y_diff
# normalize between 0-1 and calculate moving average for flatten days
    if(len(y_diff)>0 and y_diff.sum()>0):           
        y_diff = moving_average(y_diff.astype('float64'),flatten)
        amax=max(y_diff)
        if(amax!=0):
            y_diff=y_diff/amax
#        y_diff=[0 if el<epsilon else el for el in y_diff]
#        for i, val in enumerate(y_diff):
#             if(amax!=0):y_diff[i] = val / amax           
# # make the value equal to zero for the value is less than minimum of epsilon
#             if y_diff[i] < epsilon:
#                 y_diff[i]=0
#delete zeros from strting and end of the dataset. 
        y_diff,start,end  = delete_zeros(y_diff,epsilon)
        max_p=np.argmax(y_diff)
        ll=len(y_diff)
# check if the maximum value of data set is between 1/3 and 4/5 len it it means it passed the peak
        if(max_p>(ll/3.) and max_p<(ll*4/5.) and float(y_diff[ll-1])<1.):
            complete_form = 1
# if it passed the pick and at the end its value is 20% of the max it menas it is on complete the forms
            if(float(y_diff[ll-1])<.2):
                complete_form=2
# if the totall number of case for this particular region is less than epi% of the total case of total regions is not precious 
        if(ysum*1.0/sumup>epi):
            precious = 1
    else:
        y_diff=np.zeros(shape=(3))
        amax=epi
    if(precious):   
        NAME.append(NAMET[row])
        raw_data.append(data[row])
        list_not_trimed.append(y_diff_origin)
        list_trimed.append(y_diff)
        list_starts.append(start)
        list_ends.append(end)
        list_max1a.append(amax)                 # maximum 1 day for moving average
        list_max1.append(max(y_diff_origin))    # maximum of one  day
        list_maxT.append(max(y))                # maximum of total days total
        list_complete.append(complete_form)  
        list_precious.append(precious)
    row+=1
    
max1= np.sum(list_max1) 
maxT= np.sum(list_maxT) 
normal_max1 = list_max1/max1
normal_maxT = list_maxT/maxT
# Making a Matrix of similarity each row to each column not reversible
size= len(list_trimed)
#matrix of the correlation cofficent of regions data
sd_matrix = np.zeros(shape=(size,size))
p_matrix =np.zeros(shape=(size,size))
#matrix of the Maximal Information Coefficient
#https://www.freecodecamp.org/news/how-machines-make-predictions-finding-correlations-in-complex-data-dfd9f0d87889/

mic_matrix= np.zeros(shape=(size,size))
# matrix of sorted correlation for each row
sort_sd =  np.zeros(shape=(size,size))
# matrix of index of each elements that show the sorted matrix elements related to which index
sort_sd_index =  np.zeros(shape=(size,size))
sort_p=np.zeros(shape=(size,size))
sort_p_index=np.zeros(shape=(size,size))

#this part calculate the correlation coefficent factor matrix for data set 
# and save it in a sd_matrix. a sorted version of this matrix that sort the it  
#based on the columns produces and save in sort_sd matrix to find out the original index
#of the sorted one also produce and save on sort_sd_index

for i in range(0,size):
    for j in range(0,size):
        leni = len(list_trimed[i])
        lenj = len(list_trimed[j])
# if length data set i is less than j then j trimmed to be equl lenghth to i
# and via versa
        if(leni<2 or lenj<2):
            corr=0       
        elif(leni<=lenj):
            corr, p_value = pearsonr(list_trimed[i],list_trimed[j][:leni])
            mine = MINE(alpha=alphac(len(list_trimed[i])), c=15)
            mine.compute_score(list_trimed[i],list_trimed[j][:leni])
            mic = np.around(mine.mic(), 1)           
        else :
            corr, p_value = pearsonr(list_trimed[i][:lenj],list_trimed[j])
            mine = MINE(alpha=alphac(len(list_trimed[i][:lenj])), c=15)
            mine.compute_score(list_trimed[i][:lenj],list_trimed[j])
            mic = np.around(mine.mic(), 1)              
        if(corr!=corr):
            sd_matrix[i,j]=0
        else:
            sd_matrix[i,j]=np.around(corr,1)
        p_matrix[i,j]=p_value
        mic_matrix[i,j]=mic
#        if(mic_matrix[i,j]>.9 and abs(sd_matrix[i,j])<.1 ):
#            print(NAME[i],NAME[j],"have non liner relation with Mic:",
#                  mic_matrix[i,j],",CORR:",sd_matrix[i,j])
        if(abs(sd_matrix[i,j])>.5 and p_matrix[i,j]>0.05):
            print(NAME[i],NAME[j],"have H0 null not prove",
                  p_matrix[i,j],",CORR:",sd_matrix[i,j])
for i in range(0,size):
        sort_sd_index[i,:]=np.argsort(sd_matrix[i,:])[::-1]
        sort_sd[i,:]=np.sort(sd_matrix[i,:])[::-1]
        sort_p_index[i,:]=np.argsort(p_matrix[i,:])[::-1]
        sort_p[i,:]=np.sort(p_matrix[i,:])[::-1]        
#plot itterations of similar ones to consider:
#calculate weighted average correlation factor for each territory
average_corr(mic_matrix,sd_matrix,NAME,dataset_type, \
             list_max1,list_max1a,list_maxT,last_date,list_complete,list_precious,\
             list_starts,list_ends,list_trimed)
         
corr_matrix(sd_matrix,list_precious,NAME,dataset_type,"SD")
corr_matrix(mic_matrix,list_precious,NAME,dataset_type,"MIC")




consider= input_consider(NAME)
list_compare=[]
list_compare_index=[]
cstart=int(list_starts[consider])
cend=int(list_ends[i]) 
consider_data=raw_data[consider][cstart:cend]
lenc=len(consider_data)
maxd=0
for i in range(len(list_trimed)):
    cstart=int(list_starts[i])
    cend=int(list_ends[i])    
    comp_data=raw_data[i][cstart:cend]    
    lencomp=len(comp_data)
    if (lencomp>maxd):maxd=lencomp
    if (lenc<=lencomp and i!=consider):
        list_compare.append(comp_data[:lenc])
        list_compare_index.append(i)
X=np.zeros(shape=(lenc,len(list_compare_index)))
X_pre=np.zeros(shape=(maxd,len(list_compare_index)))
X_pre[:,:]=np.nan
for i in range(len(list_compare_index)):
    X[:,i]=list_compare[i]
    ind=list_compare_index[i]
    cstart=int(list_starts[ind])
    cend=int(list_ends[ind])    
    comp_data=raw_data[ind][cstart:cend]    
    X_pre[:len(comp_data),i]=comp_data

y=consider_data


#test_XGboost(list_compare,list_trimed[conside])







dot=[a * b for a, b in zip(list_complete, list_precious)]
pr_comp=dot.count(2)
pr_pass=dot.count(1)+pr_comp
print("------------%SUMMARY%-----------------------------")
print("Total study regions:    ",size) 
print("precious samples:       ",list_precious.count(1))
print("passed peak:            ",list_complete.count(1)+list_complete.count(2))
print("precious and passed peak",pr_pass)
print("reached near end:       ",list_complete.count(2))
print("precious and near end   ",pr_comp)
print("for predicting future trend, it needs to select a comparison region")
print("for that select a similarity factor less than %1.2f"%(sort_sd[consider,1]))
print("\nselected region: %s with Maximum case:%d in day: %d"%(NAME[consider],
                                                               list_maxT[consider],len(list_trimed[consider])))

similarity=input_single("please enter a vlue between 0 and %1.2f:"%(sort_sd[consider,1]),
             sort_sd[consider,1],0)
print("\nmost similar trends to %s is:"%(NAME[consider]))

# finding the region with similarity more that similarity factor for consider
#region and put it in a list list_tmp if the length of the region is higher than consider
#one. it is becuase it looks for the region that has more maturity than the consider one
#then ask the user to select one if there is any, to show the trend of the consider in
#future based on the selected region trends

list_tmp=[]
count=1
for ii in range(size):
    i2=int(sort_sd_index[consider,ii])
    if(sort_sd[consider,ii]<similarity):break
    if(len(list_trimed[i2])>len(list_trimed[consider])):
        list_tmp.append(i2)
        print("%d-[%s] similarity:%1.2f, Max_Cases:%6d,days:%d"%(count,NAME[i2],
              sort_sd[consider,ii],list_maxT[i2],len(list_trimed[i2])))
        count+=1


if not os.path.exists('fig'):
    os.makedirs('fig')
c_name= NAME[consider]

if(count==1):
    print("there isn't any territory with similarity more than: ",similarity,
          "and longer days to show trends")
    print("I'm sorry but I only show you the similar ones and not its trend")
else:    
    comp= int(input_single("select from above list to show the trend based on that:",count,1))
    comp=int(list_tmp[comp-1])
    print(NAME[comp])
    fig2 ,ax2 = plt.subplots()
    lenbase = len(list_trimed[consider])
    lencomp= len(list_trimed[comp])
    if(lenbase<lencomp):
        max_base=float(list_max1[consider])
        max_comp=float(list_max1[comp])
        start=int(list_starts[consider])
        end=int(list_ends[consider])
        cstart=int(list_starts[comp])
        cend=int(list_ends[comp])
        y=np.array(list_not_trimed[comp][cstart:cend])
        y= moving_average(y.astype('float64'),flatten)
        y1=np.array(list_not_trimed[consider][start:end])
        y1= moving_average(y1.astype('float64'),flatten)   
        max_comp=max(y)
        max_base= max(y1)
        y=y*max_base/max_comp 
        for i in range(1,len(y)):
            y[i]+=y[i-1]
        now = datetime.strptime(first_date , '%m/%d/%y') + \
                                dt.timedelta(days=int(start))
        then = now + dt.timedelta(days=int(len(y)))
        days = mdates.drange(now,then,dt.timedelta(days=1))   
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax2.plot(days,y,linewidth=6,linestyle=(0, (5, 2, 1, 2)), \
                 dash_capstyle='round', \
        label='if following trend of %s with %1.2f similarity'%(NAME[comp],\
                                                sd_matrix[consider,comp]))
        plt.gcf().autofmt_xdate()
        
        c_name= NAME[consider]
        now = datetime.strptime(first_date , '%m/%d/%y') +dt.timedelta(days=start)
        then = datetime.strptime(last_date , '%m/%d/%y') +dt.timedelta(days=1)        
        days = mdates.drange(now,then ,dt.timedelta(days=1))   
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        for i in range(1,len(y1)):
            y1[i]+=y1[i-1]
        ax2.plot(days,y1, label='%s real data max cases:%d' \
                 %(c_name,list_maxT[consider]))    
        plt.gcf().autofmt_xdate()
        fig2.set_size_inches((24, 15), forward=False)
        ax2.grid(True)
        ax2.set_title('Date:%s Prediction of comming days cases based on trend of: %s for %s of %s' \
                      %(last_date,NAME[comp],datatype[dataset_type],NAME[consider]),fontsize=15)
        ax2.legend(loc='upper left',fontsize=16)
        ax2.grid(True)
        ax2.set_xlabel('days from starting of first case for each territory')
        ax2.set_ylabel('TOTAL CASES')
        last_date=last_date.replace('/', '_')

        fig2.savefig("fig/%s_%s_%s_trend_%s.png"%(datatype[dataset_type],last_date,NAME[consider],NAME[comp]),dpi=300)

y=list_trimed[consider]
x=list(range(0,len(y)))
fig1 ,ax1 = plt.subplots()
ax1.plot(x, y,linewidth=6,linestyle=(0, (5, 2, 1, 2)), dash_capstyle='round', 
         label='{0:<20s} -days:{1:3d} Max:{2:6d}'.format(c_name,len(y),int(list_maxT[consider])))
most=6
ax1.set_title('%2d most simillar trend to: %s for %s on date:%s'%(most,c_name,
                datatype[dataset_type],last_date),fontsize=16)
for ii in range(size):
    rr1=int(sort_sd_index[consider,ii])
# consider the precious data series and similarity more than similarity percent 
    ymax=0   
    if(list_precious[rr1] and most>0 and rr1!=consider):
        most-=1
        y=list_trimed[rr1]
        x=list(range(0,len(y)))
        c_name= NAME[rr1]   
        ymax= [ymax if ymax>np.max(y) else np.max(y)]
        ax1.plot(x, y, label=('{0:<20s} -days:{1:>3d} Max:{2:>6d} Sim:{3:>1.2f}'.format(c_name, \
                                                 len(y),int(list_maxT[rr1]),
                                                 sd_matrix[consider,rr1])))
#ax1.set_ylim(0,ymax)
ax1.legend(loc='upper left',fontsize=16)
ax1.grid(True)
ax1.set_xlabel('passed days from first case reported')
ax1.set_ylabel('DAILY CASES normalized (0 for strting day, 1 is maximum of dialy cases')

    


    
last_date=last_date.replace('/', '_')

fig1.set_size_inches((24, 15), forward=False)

fig1.savefig("fig/%s_%s_%s_similarity.png"%(datatype[dataset_type],last_date,NAME[consider]),dpi=300)
plt.show()

        