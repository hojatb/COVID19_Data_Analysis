# ANALYZING **COVID19** DATASETS BY USING THE CORRELATION COEFFICIENT FACTOR

**Introduction**

This project tries to analyze **COVID19** virus spread data around the globe and the US separately. It seeks to find some meaningful relationship between its timely trends in different regions. By this measure, compare them to each other and find out how their behavior is different or similar to the other areas. Also, by finding similar behavioral trends predict the future behavior of the regions that the virus spread starts later. Feel free to use the application for the research purpose, and if it is useful, please reference this repository as the source.

**The development environment**

This application is developed based on [**Microsoft window 10**](https://www.microsoft.com/en-us/windows/get-windows-10) x64 base processor and [**python 3.7.5**](https://www.python.org/downloads/release/python-375/). It uses libraries such as [os, DateTime, sys, numpy, matplotlib.pyplot, datetime, matplotlib.dates, minepy, and geopandas.](https://www.python.org/downloads/release/python-375/) The _ **matplotlib** _ is the main library for viewing the figures and plots, as well as the geographical mapping. The geographic map implementation has been done through a powerful geo data frame tools well implemented in [**geopandas**](https://geopandas.org/gallery/cartopy_convert.html#sphx-glr-gallery-cartopy-convert-py). The library has several functions and methods for building, importing, and working with different geospatial data and especially standard shapefiles. It easily coordinated with the **matplot** library for viewing purposes. It is as easy as calling the **matplotlib** plot method with the **geodata** frame. It displays the maps with different properties and a wide variety of the coloring well enough for the project needs. Geopandas ver. 0.6.1 use for this project. Thanks for a good example and implementation of [Geraint Ian Palmer](http://www.geraintianpalmer.org.uk/2017/09/22/plotting-geopandas/). It is convenient.

**External data sources**

For the geographical data implementation, two sources of data used:

- The world country border as shapefile imported from [&quot;Made with Natural Earth. Free vector and raster map data @ naturalearthdata.com&quot;](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/). It must be copied in a directory named _gis_ for furthermore references.

shapefile=r&#39;gis\ne\_110m\_admin\_0\_countries.shp&#39;

- The states and the other territories border shapefile of the US have imported from the [US Census Bureau, Department of Commerce](https://catalog.data.gov/dataset/tiger-line-shapefile-2017-nation-u-s-current-state-and-equivalent-national). It must copied in a directory _gis_

shapefile\_states =r&#39;gis\tl\_2017\_us\_state.shp&#39;

- the COVID-19 time series data directly load from the source file in _CSV_ format form a [Github repository maintained and updated daily by the Johns Hopkins University Center for Systems Science and Engineering (JHU CSSE). Also, Supported by ESRI Living Atlas Team and the Johns Hopkins University Applied Physics Lab (JHU APL).](https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data/csse_covid_19_time_series)

Four different files daily update which is used by this application:

- Daily global confirmed cases:

url\_confirmed\_global = &#39;https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse\_covid\_19\_data/csse\_covid\_19\_time\_series/time\_series\_covid19\_confirmed\_global.csv&#39;

- Daily global death confirmed cases:

url\_death\_global = &#39;https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse\_covid\_19\_data/csse\_covid\_19\_time\_series/time\_series\_covid19\_deaths\_global.csv&#39;

- Daily US death cases:

url\_death\_US =&#39;https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse\_covid\_19\_data/csse\_covid\_19\_time\_series/time\_series\_covid19\_deaths\_US.csv&#39;

- Daily the US territory confirmed cases:

url\_confirmed\_US =&#39;https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse\_covid\_19\_data/csse\_covid\_19\_time\_series/time\_series\_covid19\_confirmed\_US.csv&#39;

All the data are in a comma-separated _ **CSV** _ format and updated daily around midnight (UTC). Every row in those tables represents a territory, and each day a new data column is added. The data series are in cumulative format. It means for every new day data is the summation of all the previous days&#39; data.

There is also another reference table in this repository. It represents the coding of geographical notions. The coding for the names of countries and states are listed up there. However, matching the right geographical position with names is no easy task. Some names are not matched with any popular databases at all. The [ISO3](https://unstats.un.org/unsd/tradekb/knowledgebase/country-code) coding is the best one, but for subcategories, there is the lake of the matching. Also, for some reason, there is an unmatched **ISO3** code for small countries and territories. It needs to improve this part of the data. This application mainly uses **ISO3** for matching purposes. Some small unmatched ones and errors are updated in the application internally.

- Lookup table :

Look = &#39;https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse\_covid\_19\_data/UID\_ISO\_FIPS\_LookUp\_Table.csv&#39;

**Application modules**

This application consist of three python files that must be copied into root directory:

- _ **Trend\_Similarity.py** _
- _ **HeatMap.py** _
- _ **input1.py** _

 
