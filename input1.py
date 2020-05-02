# Importing the dataset based on the needed type 
#  4 set of data is possible GLOBAL_CONFIRMED,GLOBAL_DEATH,US_CONFIRMED,US_DEATH
import numpy as np
import pandas as pd
import os 
import sys 
US_DEATH=0  
US_CONFIRMED =1  
GLOBAL_DEATH=2
GLOBAL_CONFRIRMED=3
dataset_typem=['CCORR/US_DEATH.csv','CCORR/US_CONFIRMED.csv','CCORR/GLOBAL_DEATH.csv','CCORR/GLOBAL_CONFRIRMED.csv']
# source datasets are imported form the source github
url_confirmed_global = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url_death_global = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
url_death_US ='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'
url_confirmed_US ='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
look='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv'
#this function get a list1 a set of the numbers and in n as int positive number a input
# it returns a moving average of n items of list1 and return the result
# the diffrent is the lenght of list1 and return list are same
# for this purpos it makes an rotational average of n items it means it repeat the list1 in 
# continuation of it , then calculate the moving average until the lenIlist1)
# then the result has the same len as input  
def moving_average(list1,n):
    nlist=np.zeros(len(list1))
    if(len(list1)<=n):
        for i in range(len(list1)):
            nlist[i]=np.average(list1)
    else:
        for i in range (len(list1)-n+1):
            nlist[i+n-1]=np.average(list1[i:i+n])
    return nlist   
# this function get a list as input and delete the trial zeros from start and ends of the
# list and return the result with the starting and ending point of the first nonzero in main
# list .if the whole list is zero then return null
def delete_zeros(y_diff,epsilon):
    if(np.sum(y_diff)==0 or len(y_diff)==0):
        return [],0,0
    elif(len(y_diff)==1):
        return y_diff,0,1

    start=0
    end=len(y_diff)
    row=0
    while(row<end):
        if(y_diff[row]>epsilon):
            break
        row+=1
    start=row
    row=end-1
    while(row>0):
        if(y_diff[row]>epsilon):
            break
        row-=1
    end=row+1
    
            
    # for row in range(1,len(y_diff)):
    #     if(np.sum(y_diff[:row])!=0):
    #         start=row-1
    #         break
    
    # for row in range(start,len(y_diff)):
    #     if(np.sum(y_diff[row:])==0):
    #         end=row
    #         break

    return y_diff[start:end],start,end  
def read_csv_corr(dataset_type):
    url=dataset_typem[dataset_type]
    if( not os.path.isfile(url)):
        return 0
    dataset = pd.read_csv(url)   
    return dataset

# this function get a correlation coefficent factor matrix and list of the name of the each
# region corresponding to matrix with the type of data set a list of maximum of one day 
# of each region and last date of processing matrix
# then calculating a waited average for every columns of matrix with corresponding one day 
# maximum as a weight then put it as a SIMILAR_TO and same average for rows and put it as
# SIMILAR_FROM together with name and ISO3 code of region how find from lookup table readed
# in and the last date in a cvs file named based on the dataset_type in a director named 
# CCORR. if the directory not exist, created 
def list_days(n):
    days=""
    for i in range(n):
        days=days+",Day"+str(i)
    return(days)
#write the correlation and mic matrix on csv files
def corr_matrix(sd_matrix,list_precious,NAME,dataset_type,TYPE):
    if not os.path.exists('CCORR'):
        os.makedirs('CCORR')
    name=dataset_typem[dataset_type][:-4]
    name=name+"_"+TYPE+".csv"
    with open(name, "w") as f: 
        f.write(" ")
        for i,n in enumerate(NAME):
                f.write(",")
                f.write(n.replace(',',';'))
        f.write("\n")
        for i,n in enumerate(NAME):
                f.write(n.replace(',',';'))
                for j ,s in enumerate(sd_matrix[i,:]):
                    f.write(",") 
                    f.write(str(s))
                f.write("\n")

                
              
#Write the MIC and CORR Matrix average data in csv file              
    
def average_corr(mic_matrix,sd_matrix,NAME,dataset_type, \
             list_max1,list_max1a,list_maxT,last_date,list_complete,list_precious,\
             list_starts,list_ends,list_trimed):
    tbl= pd.read_csv(look)
    list_max1_normal=list_max1/np.sum(list_max1) 
    ww=[len(x) for x in list_trimed]
    if not os.path.exists('CCORR'):
        os.makedirs('CCORR')
    with open(dataset_typem[dataset_type], "w") as f:  
        f.write('REGION,ISO3,CORR_WEIGHTED,MIC_WEIGHTED,LAST_DATE,START_DAY,END_DAY,DAYS,MAX_1_DAY,MAX_1_DAY_AV,MAX_TOTAL,COMPLETE_FORM,PRECIOUS')
        f.write(str(list_days(np.max(ww))))
        f.write("\n")
        for row in range(len(NAME)):
            iso3=' '
            tmp=tbl[tbl['Country_Region']==NAME[row] ]['iso3']
            if(len(tmp)!=0):
                tmp= tmp.to_numpy()
                iso3 = str(tmp[0])
            else:
                tmp=tbl[tbl['Province_State']==NAME[row] ]['iso3']
                if(len(tmp)!=0):
                    tmp= tmp.to_numpy()
                    if(iso3==' '):
                        iso3 = str(tmp[0])
                    else: 
                        print(NAME[row], "there is not any ISO3 for!!!",iso3)
            f.write((NAME[row].replace(',',';')))
            f.write(',')   
            f.write(iso3)
            f.write(',')      
            waverage=np.average(sd_matrix[:,row],weights=list_max1_normal) #*list_max1[row]/max1)
            mic_waverage=np.average(mic_matrix[:,row],weights=list_max1_normal) #*list_max1[row]/max1)
            f.write(str(waverage))
            f.write(',')       

            f.write(str(mic_waverage))
            f.write(',')                   
            f.write(str(last_date))
            f.write(',')                   
            f.write(str(list_starts[row]))
            f.write(',')                   
            f.write(str(list_ends[row]))
            f.write(',')                   
            f.write(str(ww[row]))            
            f.write(',')                   
            f.write(str(list_max1[row]))           
            f.write(',')          
            f.write(str(list_max1a[row]))           
            f.write(',')
            f.write(str(list_maxT[row]))           
            f.write(',')  
            f.write(str(list_complete[row]))            
            f.write(',')                   
            f.write(str(list_precious[row]))   
            for item in list_trimed[row]:
                f.write(',')
                f.write(str(item))
            f.write('\n')    

#get an prompt as input and print it then ask for an input from user between max,min
# if the input is inbetween return it, else ask again
def input_single(prompt,max,min):
    cont =1
    while (cont):
        print(prompt)
        t=input ("please enter:")
        t=float(t)
        if (t>=min and t<=max):
            print("selected >>",t)
            break
        else: print ("invalid entry!")
    return t
# geting a series of number sepreted by comma from user and return a list of it
def input_series(prompt,max,min):
    cont =1
    while (cont):
        print(prompt,"between:",min,max)
        print("it must be in format of: 1,17,14,23,42,78")
        t=input ("please enter:")
        if(any(c not in "0123456789," for c in t)):
            print ("invalid entry!")
        else:
           
            t=[int(x) for x in t.split(sep=',')]
            if(np.max(t)<max and np.min(t)>min):
                return t        
            else: print ("invalid entry!")
#select data set and ask for a flatten degree 1-7 for calculating moving average
#then a level of similarity between the CONSIDER trend and the others to show 0-1
#then the CONSIDER REGION for showing the result from a list of all regions
def input_data():
    cont =1
    while (cont):
        print("0.US DEATH\n1.US CONFIRMED\n2.GLOBAL DEATH\n3.GLOBAL CONFIRMED\n4.EXIT")
        dataset_type=int(input ("please select a teritory:"))
        if (dataset_type>-1 and dataset_type<4):
            print("selected dataset",dataset_type)
            break
        elif(dataset_type==4):
            cont=0
            print("program exit")
            break
        else: print ("invalid teretory!")
    # flatten moving average number
    while (cont):
        flatten=int(input ("please enter a moving average degree(1-7):"))
        if (flatten in range(1,8)):
            print("selected flatten",flatten)
            break
        elif(flatten==0):
            cont = 0
            print("program exit")
        else: print ("invalid flatten range!")
    # while (cont):
    #     similarity=float(input ("please enter a percent of similarity between 0-1:"))
    #     if (similarity>0 and similarity<=1):
    #         print("selected similarity",similarity)
    #         break
    #     elif(similarity==0):
    #         cont = 0
    #         print("program exit")
    #     else: print ("invalid similarity range!")
        
    if (dataset_type ==GLOBAL_CONFRIRMED):url = url_confirmed_global
    elif (dataset_type ==GLOBAL_DEATH):   url = url_death_global   
    elif(dataset_type ==US_DEATH): url = url_death_US      
    elif(dataset_type ==US_CONFIRMED): url= url_confirmed_US    
    else: 
        sys.exit(0)
    dataset = pd.read_csv(url)
    header=dataset.columns
#edit the mistake about Diamond Princess and delete MS Zaandam!!!
# to set a value to a dataframe is best to find inde first then set the value
    if (dataset_type ==GLOBAL_CONFRIRMED or dataset_type ==GLOBAL_DEATH):
        dataset=dataset[dataset['Country/Region']!='MS Zaandam']
        if(len(dataset[dataset['Country/Region']=='Diamond Princess'])!=0):
            tmp=dataset[dataset['Country/Region']=='Diamond Princess'].index[0]
            dataset.loc[tmp,'Province/State']='Diamond Princess'
            dataset.loc[tmp,'Country/Region']='Canada'
    data= dataset.to_numpy()
    if (dataset_type ==GLOBAL_CONFRIRMED or dataset_type ==GLOBAL_DEATH):   
        last_date= header[-1:]
        first_date= header[4]
        NAME=[]
        Y=data[:,4:]
        qlist=[]
        for  ii, country in enumerate(np.unique(data[:,1])):
            cname=str(country)
            
            subdata = data[data[:,1]==country]
            for jj in range(len(subdata)):
                if(subdata[jj,0]!=subdata[jj,0]): #subcountry is null
                    cnt =1
                    break
                else:cnt=0
            if cnt==0 :
                qlist.append(sum(subdata[:,4:]))
                NAME.append(cname)
            else:
                for row in range(len(subdata)):
                    qlist.append(subdata[row,4:])
                    if(subdata[row,0]!=subdata[row,0]):
                        NAME.append(subdata[row,1])
                    else:
                        NAME.append(subdata[row,0]) 
        Y=qlist
#        Y=np.append(Y,[np.sum(Y,axis=0)],axis=0)
#        NAME.append("_World")
        
    elif(dataset_type ==US_DEATH):
        last_date= header[-1:]
        first_date= header[12]
        Y=data[:,12:]
        header = header.to_numpy()
        states=np.unique(dataset['Province_State'])
        qlist = np.zeros((len(states),len(header)-12))
        for row , st in enumerate(states):
            q=dataset[dataset.Province_State==st]
            q=q.to_numpy()
            q=q[:,12:].sum(axis=0)
            qlist[row]=q
        NAME=[x for x in states]
        Y=qlist
#        Y=np.append(Y,[np.sum(Y,axis=0)],axis=0)
#        NAME.append("_USA")        
           
    elif(dataset_type ==US_CONFIRMED):
        last_date= header[-1:]
        first_date= header[11]
        Y=data[:,11:]
        header = header.to_numpy()
        states=np.unique(dataset['Province_State'])
        qlist = np.zeros((len(states),len(header)-11))
        for row , st in enumerate(states):
            q=dataset[dataset.Province_State==st]
            q=q.to_numpy()
            q=q[:,11:].sum(axis=0)
            qlist[row]=q
        NAME=[x for x in states]
        Y=qlist
#        Y=np.append(Y,[np.sum(Y,axis=0)],axis=0)
#        NAME.append("_USA") 
    # row number of the terretory to be consider
    # while (cont):
    #     for ii in range (len(NAME)):
    #         print(ii,">",NAME[ii],) 
    #     consider=int(input ("\nplease enter a region number:"))
    #     if (consider in range(len(NAME))):
    #         print("\nselected",consider,">>",NAME[consider])
    #         break
       
    #     else: print ("invalid region range!")  
    #  #number of itration to be compare to main terretory
    # last_date=last_date[0]
    # print( "data is from",first_date," until", last_date)  
    last_date=last_date[0] 
    return dataset_type,flatten,NAME,Y,last_date,first_date  

def input_consider(NAME):
    while (1):
        for ii in range (len(NAME)):
            print(ii,">",NAME[ii],) 
        consider=int(input ("\nplease enter a region number:"))
        if (consider in range(len(NAME))):
            print("\nselected",consider,">>",NAME[consider])
            break
           
        else: print ("invalid region range!")  
         #number of itration to be compare to main terretory
    return(consider)