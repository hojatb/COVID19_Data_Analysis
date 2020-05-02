# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:48:43 2020

@author: hojat behrooz
"""
datatype=['US_DEATH_CASES',  
'US_CONFIRMED_CASES',
'GLOBAL_DEATH_CASES',
'GLOBAL_CONFRIRMED_CASES']
# fit a polynomial to a vector of the data
#import numpy as np
import os
import sys
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from  input1 import read_csv_corr, input_single
#natural shape file of the globe imported from https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/
shapefile=r'gis\ne_110m_admin_0_countries.shp'
#natural shape file of the the USA STATES imported from https://catalog.data.gov/dataset/tiger-line-shapefile-2017-nation-u-s-current-state-and-equivalent-national
shapefile_states =r'gis\tl_2017_us_state.shp'

look='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/UID_ISO_FIPS_LookUp_Table.csv'

Title =['USA_DEATH','US_CONFIRMED','GLOBAL_DEATH','GLOBAL_CONFIRMED']
US_DEATH=0  
US_CONFIRMED =1  
GLOBAL_DEATH=2
GLOBAL_CONFIRMED=3
dataset_typem=['US_DEATH.csv','US_CONFIRMED.csv','GLOBAL_DEATH.csv','GLOBAL_CONFRIRMED.csv']


print("0.US DEATH\n1.US CONFIRMED\n2.GLOBAL DEATH\n3.GLOBAL CONFIRMED\n4.EXIT")
inp= int(input_single("please enter a catagory:",4,0))
if inp>3 : sys.exit()

dataset = read_csv_corr(inp)
if (len(dataset)<2):
    print ("the correlation matrix for this catagory not found,please first run the similarity program for this catagory")
    sys.exit()
if inp<2:
    world1=  gpd.read_file(shapefile_states)
else:
    world1 = gpd.read_file(shapefile)
    
dataset.sort_values(by=['CORR_WEIGHTED'],inplace=True, ascending=False)
fig1 ,ax1 = plt.subplots()
fig2 , ax2 = plt.subplots()
fig3 , ax3 = plt.subplots()
fig , ax = plt.subplots()
minx=min(dataset['CORR_WEIGHTED'])
maxx=max(dataset['CORR_WEIGHTED'])
delta=maxx-minx
d1= dataset.iloc[:15,:]
d1.plot(ax=ax1,x ='REGION', y='CORR_WEIGHTED', kind = 'barh')
ax1.set_title('15 Most SIMILAR region to others  IN TREND OF %s CASES'%(Title[inp]))
d2= dataset.iloc[-15:,:]
d2.plot(ax=ax2,x ='REGION', y='CORR_WEIGHTED', kind = 'barh')
ax2.set_title('15 Least SIMILAR region to others  IN TREND OF %s CASES'%(Title[inp]))
dataset.plot(ax=ax3,x ='REGION', y='CORR_WEIGHTED', kind = 'bar')
ax3.set_title('SIMILAR region to others  IN TREND OF %s CASES'%(Title[inp]))
# some map data cleaning
if(inp>1):
    world1= world1[world1['SOVEREIGNT']!='Antarctica']    
    mix=pd.merge(dataset,world1,left_on='ISO3',right_on='ADM0_A3', how='left')
    mix =mix[pd.notnull(mix['featurecla'])]
    mix=mix[['REGION','ISO3','MIC_WEIGHTED','CORR_WEIGHTED','geometry']]
    mix =gpd.GeoDataFrame(mix)
else:
    dataset=dataset[(dataset['REGION']!='Diamond Princess') & \
                (dataset['REGION']!='Virgin Islands') & \
                (dataset['REGION']!='American Samoa') & \
                (dataset['REGION']!='Northern Mariana Islands') & \
                (dataset['REGION']!='Grand Princess') & \
                (dataset['REGION']!='Guam') &
                (dataset['REGION']!='Puerto Rico') &
                (dataset['REGION']!='Hawaii') &
                (dataset['REGION']!='Alaska')]
    mix=pd.merge(dataset,world1,left_on='REGION',right_on='NAME',how='left')
    mix =mix[pd.notnull(mix['NAME'])]
    mix=mix[['NAME','MIC_WEIGHTED','CORR_WEIGHTED','geometry']]
    
last_date= dataset.loc[0]['LAST_DATE']
last_date=last_date.replace('/', '_')
mix =gpd.GeoDataFrame(mix)
colormap = "RdBu"   # https://matplotlib.org/tutorials/colors/colormaps.html 
if inp>1: 
    world1.plot(ax=ax,color='BLACK' )   
    mix.plot(ax=ax,column='CORR_WEIGHTED', cmap=colormap, \
                vmin=min(mix.CORR_WEIGHTED), vmax=max(mix.CORR_WEIGHTED), \
                legend=True   , legend_kwds={'label': \
                    "SIMILARITY BY COLOR left color less similarity. \
                        right color is more similar trends to others        THERE IS NOT PRECIOUS DATA FOR BLACK COLORED REGIONS", \
                'orientation': "horizontal"})    
else:      
    mix.plot(ax=ax, column='CORR_WEIGHTED', cmap=colormap, \
                vmin=min(mix.CORR_WEIGHTED), vmax=max(mix.CORR_WEIGHTED), \
                legend=True   , legend_kwds={'label': \
                    "SIMILARITY BY COLOR left color less similarity. \
                        right color is more similar trends to others        THERE IS NOT PRECIOUS DATA FOR BLACK COLORED REGIONS ", \
                'orientation': "horizontal"})    
    
# http://www.geraintianpalmer.org.uk/2017/09/22/plotting-geopandas/
#for annotating country or states names
mix['centroid'] = mix['geometry'].centroid
props = dict(boxstyle='round', facecolor='linen', alpha=.3)
if(inp>1):
    
    for point in mix.iterrows():
        ax.text(point[1]['centroid'].x,
                point[1]['centroid'].y,
                "%s\n%1.2f"%(point[1]['REGION'],point[1]['CORR_WEIGHTED']),
                horizontalalignment='center',
                fontsize=10,
                bbox=props)
else:
    for point in mix.iterrows():
        ax.text(point[1]['centroid'].x,
            point[1]['centroid'].y,
            "%s\n%1.2f"%(point[1]['NAME'],point[1]['CORR_WEIGHTED']),
            horizontalalignment='center',
            fontsize=9,
            bbox=props)


   
ax.set_title('SIMILARITY TO OTHERS IN TREND OF %s CASES on %s'%(Title[inp],last_date))
ax.axis('off')
fig1.set_size_inches((24, 15), forward=False)
fig2.set_size_inches((24, 15), forward=False)
fig3.set_size_inches((24, 15), forward=False)
fig.set_size_inches((24, 15), forward=False)

if not os.path.exists('fig'):
    os.makedirs('fig')
fig1.savefig("fig/%s_%s_Most_similar.png"%(datatype[inp],last_date),dpi=300)
fig2.savefig("fig/%s_%s_Least_similar.png"%(datatype[inp],last_date),dpi=300)
fig3.savefig("fig/%s_%s_similarity_trend.png"%(datatype[inp],last_date),dpi=300)
fig.savefig("fig/%s_%s_Heat_Map.png"%(datatype[inp],last_date),dpi=300)
plt.show()



    