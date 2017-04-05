
# EDA on Video Games Sales Dataset from Kaggle

jie.hu.ds@gmail.com

--------

* <a href='#Package'>1. Package</a>
* <a href='#Dataset'>2. Dataset</a>
* <a href='#Statistical Summary'>3. Statistical Summary</a>
* <a href='#Viz - Bivariate'>4. Viz - Bivariate</a>
* <a href='#Viz - Multivariate'>5. Viz - Multivariate</a>
* <a href='#Conclusion'>6. Conclusion</a>

------

<a id='Package'>Package</a>


```python
# Packages
import pandas as pd
import numpy as np
import scipy as sp
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
```


```python
# Plotly token
plotly.tools.set_credentials_file(username='your account', api_key='your token')
```

<a id='Dataset'>Dataset</a>


```python
df = pd.read_csv("vgsales.csv")
df[:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Name</th>
      <th>Platform</th>
      <th>Year</th>
      <th>Genre</th>
      <th>Publisher</th>
      <th>NA_Sales</th>
      <th>EU_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>Global_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Wii Sports</td>
      <td>Wii</td>
      <td>2006.0</td>
      <td>Sports</td>
      <td>Nintendo</td>
      <td>41.49</td>
      <td>29.02</td>
      <td>3.77</td>
      <td>8.46</td>
      <td>82.74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Super Mario Bros.</td>
      <td>NES</td>
      <td>1985.0</td>
      <td>Platform</td>
      <td>Nintendo</td>
      <td>29.08</td>
      <td>3.58</td>
      <td>6.81</td>
      <td>0.77</td>
      <td>40.24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Mario Kart Wii</td>
      <td>Wii</td>
      <td>2008.0</td>
      <td>Racing</td>
      <td>Nintendo</td>
      <td>15.85</td>
      <td>12.88</td>
      <td>3.79</td>
      <td>3.31</td>
      <td>35.82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Wii Sports Resort</td>
      <td>Wii</td>
      <td>2009.0</td>
      <td>Sports</td>
      <td>Nintendo</td>
      <td>15.75</td>
      <td>11.01</td>
      <td>3.28</td>
      <td>2.96</td>
      <td>33.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Pokemon Red/Pokemon Blue</td>
      <td>GB</td>
      <td>1996.0</td>
      <td>Role-Playing</td>
      <td>Nintendo</td>
      <td>11.27</td>
      <td>8.89</td>
      <td>10.22</td>
      <td>1.00</td>
      <td>31.37</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Tetris</td>
      <td>GB</td>
      <td>1989.0</td>
      <td>Puzzle</td>
      <td>Nintendo</td>
      <td>23.20</td>
      <td>2.26</td>
      <td>4.22</td>
      <td>0.58</td>
      <td>30.26</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>New Super Mario Bros.</td>
      <td>DS</td>
      <td>2006.0</td>
      <td>Platform</td>
      <td>Nintendo</td>
      <td>11.38</td>
      <td>9.23</td>
      <td>6.50</td>
      <td>2.90</td>
      <td>30.01</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Wii Play</td>
      <td>Wii</td>
      <td>2006.0</td>
      <td>Misc</td>
      <td>Nintendo</td>
      <td>14.03</td>
      <td>9.20</td>
      <td>2.93</td>
      <td>2.85</td>
      <td>29.02</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>New Super Mario Bros. Wii</td>
      <td>Wii</td>
      <td>2009.0</td>
      <td>Platform</td>
      <td>Nintendo</td>
      <td>14.59</td>
      <td>7.06</td>
      <td>4.70</td>
      <td>2.26</td>
      <td>28.62</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Duck Hunt</td>
      <td>NES</td>
      <td>1984.0</td>
      <td>Shooter</td>
      <td>Nintendo</td>
      <td>26.93</td>
      <td>0.63</td>
      <td>0.28</td>
      <td>0.47</td>
      <td>28.31</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 16598 entries, 0 to 16597
    Data columns (total 11 columns):
    Rank            16598 non-null int64
    Name            16598 non-null object
    Platform        16598 non-null object
    Year            16327 non-null float64
    Genre           16598 non-null object
    Publisher       16540 non-null object
    NA_Sales        16598 non-null float64
    EU_Sales        16598 non-null float64
    JP_Sales        16598 non-null float64
    Other_Sales     16598 non-null float64
    Global_Sales    16598 non-null float64
    dtypes: float64(6), int64(1), object(4)
    memory usage: 1.4+ MB



```python
df.shape
```




    (16598, 11)



<a id='Statistical Summary'>Statistical Summary</a>


```python
df.describe()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Rank</th>
      <th>Year</th>
      <th>NA_Sales</th>
      <th>EU_Sales</th>
      <th>JP_Sales</th>
      <th>Other_Sales</th>
      <th>Global_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16598.000000</td>
      <td>16327.000000</td>
      <td>16598.000000</td>
      <td>16598.000000</td>
      <td>16598.000000</td>
      <td>16598.000000</td>
      <td>16598.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8300.605254</td>
      <td>2006.406443</td>
      <td>0.264667</td>
      <td>0.146652</td>
      <td>0.077782</td>
      <td>0.048063</td>
      <td>0.537441</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4791.853933</td>
      <td>5.828981</td>
      <td>0.816683</td>
      <td>0.505351</td>
      <td>0.309291</td>
      <td>0.188588</td>
      <td>1.555028</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1980.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4151.250000</td>
      <td>2003.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.060000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8300.500000</td>
      <td>2007.000000</td>
      <td>0.080000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.170000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>12449.750000</td>
      <td>2010.000000</td>
      <td>0.240000</td>
      <td>0.110000</td>
      <td>0.040000</td>
      <td>0.040000</td>
      <td>0.470000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>16600.000000</td>
      <td>2020.000000</td>
      <td>41.490000</td>
      <td>29.020000</td>
      <td>10.220000</td>
      <td>10.570000</td>
      <td>82.740000</td>
    </tr>
  </tbody>
</table>
</div>



<a id='Viz - Bivariate'>4. Viz - Bivariate</a>

**Release vs. Platform**


```python
# Platform
df.Platform = df.Platform.astype('category')
df.Platform.describe()
```




    count     16598
    unique       31
    top          DS
    freq       2163
    Name: Platform, dtype: object




```python
platform_count = df.groupby('Platform', axis=0).count().reset_index()[['Platform','Name']].sort_values(by = "Name", ascending=True)
```


```python
# Game counts by platform

import plotly.graph_objs as go

layout = go.Layout(
    title='Total Release by Platforms',
    yaxis=dict(
        title='Platform'
    ),
    xaxis=dict(
        title='Count'
    ),
    height=900, width=900
)

trace = go.Bar(
            x=platform_count.Name,
            y=platform_count.Platform,
            orientation = 'h'
        )


fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/388.embed" height="900px" width="900px"></iframe>



**Release by Year**


```python
year_count = df.groupby('Year', axis=0).count().reset_index()[['Year','Name']]
year_count.Year = year_count.Year.astype('int')

# remove data after 2016
year_count = year_count[year_count.Year <= 2016]
```


```python
trace = go.Scatter(
    x = year_count.Year,
    y = year_count.Name,
    mode = 'lines',
    name = 'lines'
    
)


layout = go.Layout(
    title='Release by Year',
    yaxis=dict(
        title='Count'
    ),
    xaxis=dict(
        title='Year'
    ),
    height=900, width=900
)

fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/390.embed" height="900px" width="900px"></iframe>



Because the sales of new released games are still booming, the decreasing curve doesn't mean the market is decreasing

**Release by Genre**


```python
genre_count = df.groupby('Genre', axis=0).count().reset_index()[['Genre','Name']].sort_values(by = "Name", ascending=True)
layout = go.Layout(
    title='Releases by Genre',
    yaxis=dict(
        title='Genre'
    ),
    xaxis=dict(
        title='Releases'
    ),
    height=400, width=900
)

trace = go.Bar(
            x=genre_count.Name,
            y=genre_count.Genre,
            orientation = 'h'
        )


fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/392.embed" height="400px" width="900px"></iframe>



Action, sports and music games took top 3 in game releases.

**Release by Publisher**


```python
publisher_count = df.groupby('Publisher', axis=0).count().reset_index()[['Publisher','Name']].sort_values(by = "Name", ascending=True)
publisher_count = publisher_count.tail(n=30)
layout = go.Layout(
    title='Release by Publisher (Top 30)',

    xaxis=dict(
        title='Releases'
    ),
    height=700, width=800,
    margin=go.Margin(
        l=300,
        r=50,
        b=100,
        t=100,
        pad=4
    )
)

trace = go.Bar(
            x=publisher_count.Name,
            y=publisher_count.Publisher,
            orientation = 'h'
        )


fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/394.embed" height="700px" width="800px"></iframe>



**Sales by Publisher**


```python
publisher_sales = df.groupby('Publisher', axis=0).sum().reset_index()[['Publisher','Global_Sales']].sort_values(by = "Global_Sales", ascending=True)
publisher_sales = publisher_sales.tail(n=30)

layout = go.Layout(
    title='Sales by Publisher (Top 30)',

    xaxis=dict(
        title='Sales (in Millions)'
    ),
    height=700, width=800,
    margin=go.Margin(
        l=300,
        r=50,
        b=100,
        t=100,
        pad=4
    )
)

trace = go.Bar(
            x=publisher_sales.Global_Sales,
            y=publisher_sales.Publisher,
            orientation = 'h'
        )


fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/396.embed" height="700px" width="800px"></iframe>



**Revenue per game by Publisher**


```python
new_df = df
new_df['Game_Count'] = 1
new_df = new_df.groupby(['Publisher']).sum().reset_index()[['Publisher', 'Global_Sales','Game_Count']]
new_df['Revenue_per_game'] = new_df.Global_Sales/new_df.Game_Count

new_df = new_df.sort_values(by = "Revenue_per_game", ascending=True).\
                            tail(n=30)
layout = go.Layout(
    title='Revenue_per_game by Publisher (Top 30)',

    xaxis=dict(
        title='Revenue_per_game (in Millions)'
    ),
    height=700, width=800,
    margin=go.Margin(
        l=300,
        r=50,
        b=100,
        t=100,
        pad=4
    )
)

trace = go.Bar(
            x=new_df.Revenue_per_game,
            y=new_df.Publisher,
            orientation = 'h'
        )


fig = go.Figure(data=[trace], layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/398.embed" height="700px" width="800px"></iframe>



Average revenue per game shows the cashability of games published by the publishers.

**Sales by Genre**


```python
sales_by_genre = df.groupby(['Genre','Name'], axis = 0).sum().reset_index()[['Genre','Name','Global_Sales']]
```


```python
import random
from numpy import * 
genres = sales_by_genre.Genre.unique()
traces = []
c = ['hsl('+str(h)+',50%'+',50%)' for h in linspace(0, 360, len(genres))]

for i in range(len(genres)):
    genre = genres[i]
    df_genre = sales_by_genre[sales_by_genre.Genre == genre]
    trace = go.Box(
        y=np.array(df_genre.Global_Sales),
        name=genre,
        boxmean=True,
        marker={'color': c[i]}
    )
    
    traces.append(trace)

layout = go.Layout(
    title='Sales by Genre (A lot of outliers)',
    showlegend=False,
    yaxis=dict(
        title='Sales (in Millions)'
    )
)
    

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)

```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/400.embed" height="525px" width="100%"></iframe>




```python
# The outliers are like:
df.groupby(['Genre','Name'], axis = 0).\
         sum()[['Global_Sales']].\
         sort_values(by="Global_Sales", ascending = False).\
         reset_index()[:10]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Genre</th>
      <th>Name</th>
      <th>Global_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sports</td>
      <td>Wii Sports</td>
      <td>82.74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Action</td>
      <td>Grand Theft Auto V</td>
      <td>55.92</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Platform</td>
      <td>Super Mario Bros.</td>
      <td>45.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Puzzle</td>
      <td>Tetris</td>
      <td>35.84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Racing</td>
      <td>Mario Kart Wii</td>
      <td>35.82</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Sports</td>
      <td>Wii Sports Resort</td>
      <td>33.00</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Role-Playing</td>
      <td>Pokemon Red/Pokemon Blue</td>
      <td>31.37</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Shooter</td>
      <td>Call of Duty: Black Ops</td>
      <td>31.03</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Shooter</td>
      <td>Call of Duty: Modern Warfare 3</td>
      <td>30.83</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Platform</td>
      <td>New Super Mario Bros.</td>
      <td>30.01</td>
    </tr>
  </tbody>
</table>
</div>




```python
# After delete outlier

PERCENTAGE = 0.95
traces = []

for i in range(len(genres)):
    genre = genres[i]
    df_genre = sales_by_genre[sales_by_genre.Genre == genre]
    df_genre = df_genre[df_genre.Global_Sales < df_genre.Global_Sales.quantile(PERCENTAGE)]
    
    trace = go.Box(
        y=np.array(df_genre.Global_Sales),
        name=genre,
        boxmean=True,
        marker={'color': c[i]}
    )
    
    traces.append(trace)

layout = go.Layout(
    title='Sales by Genre (Less outliers)',
    showlegend=False,
    yaxis=dict(
        title='Sales (in Millions)'
    )
)
    

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)

```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/402.embed" height="525px" width="100%"></iframe>



Then let's check revenue distribution of TOP 1% sales of each genre


```python
# After delete outlier

PERCENTAGE = 0.99
traces = []

for i in range(len(genres)):
    genre = genres[i]
    df_genre = sales_by_genre[sales_by_genre.Genre == genre]
    df_genre = df_genre[df_genre.Global_Sales > df_genre.Global_Sales.quantile(PERCENTAGE)]
    
    trace = go.Box(
        y=np.array(df_genre.Global_Sales),
        name=genre,
        boxmean=True,
        marker={'color': c[i]}
    )
    
    traces.append(trace)

layout = go.Layout(
    title='Sales by Genre (TOP 1% games)',
    showlegend=False,
    yaxis=dict(
        title='Sales (in Millions)'
    )
)
    

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)

```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/404.embed" height="525px" width="100%"></iframe>



** Sales of games by Publisher **


```python
top10_publishers = np.array(df.groupby('Publisher', axis=0).sum().\
                           reset_index()[['Publisher','Global_Sales']].\
                           sort_values(by = "Global_Sales", ascending=True).\
                           tail(n=10)['Publisher'])

top10_df = df[[pub in top10_publishers for pub in df.Publisher]]
sales_by_publisher = top10_df.groupby(['Publisher','Name']).sum().reset_index()[['Publisher','Name','Global_Sales']]
```


```python
PERCENTAGE = 0.9
traces = []

for i in range(len(top10_publishers)):
    publisher = top10_publishers[i]
    df_pub = sales_by_publisher[sales_by_publisher.Publisher == publisher]
    df_pub = df_pub[df_pub.Global_Sales < df_pub.Global_Sales.quantile(PERCENTAGE)]
    
    trace = go.Box(
        y=np.array(df_pub.Global_Sales),
        name=publisher,
        boxmean=True,
        marker={'color': c[i]}
    )
    
    traces.append(trace)

layout = go.Layout(
    title='Sales by Publisher (Majority Games)',
    showlegend=False,
    yaxis=dict(
        title='Sales (in Millions)'
    )
)
    

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/406.embed" height="525px" width="100%"></iframe>



However, in game industry, only top games are extremely profitable, so let's see top games of these top publishers


```python
PERCENTAGE = 0.95
traces = []

for i in range(len(top10_publishers)):
    publisher = top10_publishers[i]
    df_pub = sales_by_publisher[sales_by_publisher.Publisher == publisher]
    df_pub = df_pub[df_pub.Global_Sales > df_pub.Global_Sales.quantile(PERCENTAGE)]
    
    trace = go.Box(
        y=np.array(df_pub.Global_Sales),
        name=publisher,
        boxmean=True,
        marker={'color': c[i]},
        boxpoints = 'all'
    )
    
    traces.append(trace)

layout = go.Layout(
    title='Sales by Publisher (TOP 5% Games)',
    showlegend=False,
    yaxis=dict(
        title='Sales (in Millions)'
    )
)
    

fig = go.Figure(data=traces, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/408.embed" height="525px" width="100%"></iframe>



The masterpieces of Nintendo, Activision and Take-Two Interactive are more powerful in cashability.


```python
sales_by_year = df.groupby('Year', axis=0).sum().reset_index()[['Year','NA_Sales','EU_Sales','JP_Sales','Other_Sales','Global_Sales']]
sales_by_year.Year = sales_by_year.Year.astype('int')
```


```python
sales_by_year = sales_by_year[sales_by_year.Year <= 2016]
```


```python
trace_Global = go.Scatter(
    x = sales_by_year.Year,
    y = sales_by_year.Global_Sales,
    mode = 'none',
    name = 'Global_Sales',
    fill='tonexty',
)

trace_NA = go.Scatter(
    x = sales_by_year.Year,
    y = sales_by_year.NA_Sales,
    mode = 'none',
    fill='tonexty',
    name = 'NA_Sales'
)

trace_EU = go.Scatter(
    x = sales_by_year.Year,
    y = sales_by_year.EU_Sales,
    mode = 'none',
    fill='tonexty',
    name = 'EU_Sales'
)

trace_JP = go.Scatter(
    x = sales_by_year.Year,
    y = sales_by_year.JP_Sales,
    mode = 'none',
    fill='tonexty',
    name = 'JP_Sales'
)

trace_Other = go.Scatter(
    x = sales_by_year.Year,
    y = sales_by_year.Other_Sales,
    mode = 'none',
    fill='tozeroy',
    name = 'Other_Sales'
)



layout = go.Layout(
    title='Sales by Region',

    xaxis=dict(
        title='Year'
    ),
    yaxis=dict(
        title='Sales (in Millions)'
    ),
    
    height=700, width=900,
    margin=go.Margin(
        l=100,
        r=50,
        b=100,
        t=100,
        pad=4
    )
)


fig = go.Figure(data=[trace_Other, trace_JP, trace_EU, trace_NA, trace_Global], layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/410.embed" height="700px" width="900px"></iframe>



- North America is always the biggest market for video games.
- Sales in other regions are booming

<a id='Viz - Multivariate'>5. Viz - Multivariate</a>

** Regional Sales by Genre across year (How genre in each region changes) **
    
I will use below function to get traces for plotly


```python
# Get list of unique genres
genres = np.sort(df.Genre.unique())[::-1]

def get_traces(df, region):
    regional_df = df.groupby(['Genre','Year'], axis=0).sum().reset_index()[['Genre','Year', region]]
    years = range(1980,2018)
    
    temp_dict = {}
    for genre in genres:
        temp_dict[genre] = {}
        for year in years:
            try:
                temp_value = round(np.array(regional_df[(regional_df.Genre == genre) & 
                                   (regional_df.Year == year)][region])[0],2)
            except:
                temp_value = 0
            temp_dict[genre][year] = temp_value
    
    traces = []
    for genre in genres:
        trace = go.Bar(
            x = years,
            y = temp_dict[genre].values(),
            name=genre
        )
        traces.append(trace)
    
    return traces
```

*Global*


```python
data = get_traces(df, 'Global_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales change in Global',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        )
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/412.embed" height="525px" width="100%"></iframe>



In Global market:
- Sale of Action and Shooter games are increasing
- Sale of Music, Sports, Fighting, Racing and Puzzle games are decreasing
- Much fewer revenue were generated by Strategy, Puzzle and Racing games

In *North America*


```python
data = get_traces(df, 'NA_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales change in North America',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        )
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/414.embed" height="525px" width="100%"></iframe>



North America has distribution pretty similar to Global market, because it takes up most of global sales. NA market tends to prefer Action and Shooter games to other games.

In *Japan*


```python
data = get_traces(df, 'JP_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales change in Japan',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        )
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/416.embed" height="525px" width="100%"></iframe>



In Japan, besides Action, Role-Playing games attracts most revenue, which is quite different from NA market. 

In *Europe*


```python
data = get_traces(df, 'EU_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales change in Europe',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        )
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/418.embed" height="525px" width="100%"></iframe>



European people tends to have similar taste with North American players.

In *Other Regions*


```python
data = get_traces(df, 'Other_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales change in Other (not JP, NA, EU)',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        )
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/420.embed" height="525px" width="100%"></iframe>



Sports and shooter games are booming in these regions.

- Sales **Percentage** of genres over time (How each market grows)

I change a little bit of the function to get traces


```python
def get_percent_traces(df, region):
    temp_df = df.groupby(['Year','Genre'], axis=0).sum()[[region]]
    df_pcts = temp_df.groupby(level=0).apply(lambda x: 100 * x / float(x.sum()))
    df_pcts = df_pcts.reset_index()
    regional_df = df_pcts[df_pcts.Year < 2017] 
    
    years = range(1980,2018)
    
    temp_dict = {}
    for genre in genres:
        temp_dict[genre] = {}
        for year in years:
            try:
                temp_value = round(np.array(regional_df[(regional_df.Genre == genre) & 
                                   (regional_df.Year == year)][region])[0],2)
            except:
                temp_value = 0
            temp_dict[genre][year] = temp_value
    
    
    traces = []
    for genre in genres:
        trace = go.Bar(
            x = years,
            y = temp_dict[genre].values(),
            name=genre
        )
        traces.append(trace)
    
    return traces
```

*Global*


```python
data = get_percent_traces(df, 'Global_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales Percentage of Genres over Years in Global',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        )
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/422.embed" height="525px" width="100%"></iframe>



By percentage of genres:
- Action and Shooter are both increasing rapidly
- Racing, puzzle, music, and strategy games are disapearing

*North America*


```python
data = get_percent_traces(df, 'NA_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales Percentage of Genres over Years in North America',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        )
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/424.embed" height="525px" width="100%"></iframe>



*Japan*


```python
data = get_percent_traces(df, 'JP_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales Percentage of Genres over Years in Japan',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        )
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/426.embed" height="525px" width="100%"></iframe>



In Japan, RPG is always most welcome genre. And Action games are booming.

In *Europe*


```python
data = get_percent_traces(df, 'EU_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales Percentage of Genres over Years in Europe',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        )
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/428.embed" height="525px" width="100%"></iframe>



Europe has quite similar style with NA market

In *Other regions*


```python
data = get_percent_traces(df, 'Other_Sales')
layout = go.Layout(
        barmode='stack',
        title = 'Sales Percentage of Genres over Years in Other regions',
        xaxis=dict(
            title='Year'
        ),
        yaxis=dict(
            title='Sales (in Millions)'
        )
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/430.embed" height="525px" width="100%"></iframe>




```python
len(df.Publisher.unique())
```




    579



**Sales by genre, publisher**


```python
# Prefered genres of top-5-sale publishers
genres = genres[::-1]

def get_traces_genre_publisher(region):
    top5_publishers = np.array(df.groupby('Publisher', axis=0).sum().\
                               reset_index()[['Publisher', 'Global_Sales']].\
                               sort_values(by = 'Global_Sales', ascending=True).\
                               tail(n=5)['Publisher'])

    top5_df = df[[pub in top5_publishers for pub in df.Publisher]]
    top5_genre_df = top5_df.groupby(['Publisher','Genre']).sum().reset_index()[['Publisher','Genre',region]]

    traces = []
    for i in range(len(top5_publishers)):
        publisher = top5_publishers[i]
        temp_df = top5_genre_df[top5_genre_df.Publisher == publisher]
        
       

        trace = go.Bar(
            x = genres,
            y = np.array(temp_df[region]),
            name=publisher
        )
        traces.append(trace)

    return traces


```


```python
data = get_traces_genre_publisher('Global_Sales')
layout = go.Layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(title='Sales (in Millions)'),
        barmode='group',
        title = 'Global Sales by Genre and Publisher'
    )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/432.embed" height="525px" width="100%"></iframe>



In global market:
- Nintendo focus more on Platform, RPG and Sports games
- EA focus more on Sports, shooter and racing games
- Activision earn money more on shooter games

Take a look at the top games of these publishers:


```python
# Top 5 games of Nintendo
df[df.Publisher == 'Nintendo'].sort_values(by = 'Global_Sales', ascending=False)[['Publisher','Name','Global_Sales']][:5]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Publisher</th>
      <th>Name</th>
      <th>Global_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Nintendo</td>
      <td>Wii Sports</td>
      <td>82.74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nintendo</td>
      <td>Super Mario Bros.</td>
      <td>40.24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Nintendo</td>
      <td>Mario Kart Wii</td>
      <td>35.82</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Nintendo</td>
      <td>Wii Sports Resort</td>
      <td>33.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Nintendo</td>
      <td>Pokemon Red/Pokemon Blue</td>
      <td>31.37</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top 5 games of EA
df[df.Publisher == 'Electronic Arts'].sort_values(by = 'Global_Sales', ascending=False)[['Publisher','Name','Global_Sales']][:5]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Publisher</th>
      <th>Name</th>
      <th>Global_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>77</th>
      <td>Electronic Arts</td>
      <td>FIFA 16</td>
      <td>8.49</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Electronic Arts</td>
      <td>FIFA Soccer 13</td>
      <td>8.24</td>
    </tr>
    <tr>
      <th>83</th>
      <td>Electronic Arts</td>
      <td>The Sims 3</td>
      <td>8.11</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Electronic Arts</td>
      <td>Star Wars Battlefront (2015)</td>
      <td>7.67</td>
    </tr>
    <tr>
      <th>99</th>
      <td>Electronic Arts</td>
      <td>Battlefield 3</td>
      <td>7.34</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Top 5 games of Activision
df[df.Publisher == 'Activision'].sort_values(by = 'Global_Sales', ascending=False)[['Publisher','Name','Global_Sales']][:5]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Publisher</th>
      <th>Name</th>
      <th>Global_Sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29</th>
      <td>Activision</td>
      <td>Call of Duty: Modern Warfare 3</td>
      <td>14.76</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Activision</td>
      <td>Call of Duty: Black Ops</td>
      <td>14.64</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Activision</td>
      <td>Call of Duty: Black Ops 3</td>
      <td>14.24</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Activision</td>
      <td>Call of Duty: Black Ops II</td>
      <td>14.03</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Activision</td>
      <td>Call of Duty: Black Ops II</td>
      <td>13.73</td>
    </tr>
  </tbody>
</table>
</div>




```python
data = get_traces_genre_publisher('NA_Sales')
layout = go.Layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(title='Sales (in Millions)'),
        barmode='group',
        title = 'North America - Sales by Genre and Publisher'
    )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/434.embed" height="525px" width="100%"></iframe>




```python
data = get_traces_genre_publisher('JP_Sales')
layout = go.Layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(title='Sales (in Millions)'),
        barmode='group',
        title = 'Japan - Sales by Genre and Publisher'
    )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/436.embed" height="525px" width="100%"></iframe>



Japan is almost taken up by its local publishers, Nintendo and SONY in all genres


```python
data = get_traces_genre_publisher('EU_Sales')
layout = go.Layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(title='Sales (in Millions)'),
        barmode='group',
        title = 'Europe - Sales by Genre and Publisher'
    )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/438.embed" height="525px" width="100%"></iframe>



In Europe, EA's adventure games 


```python
data = get_traces_genre_publisher('Other_Sales')
layout = go.Layout(
        xaxis=dict(tickangle=-45),
        yaxis=dict(title='Sales (in Millions)'),
        barmode='group',
        title = 'Other Regions - Sales by Genre and Publisher'
    )

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~jie.hu000/440.embed" height="525px" width="100%"></iframe>



<a id = "Conclusion"> Conclusion </a>

1. Global game market is increasing
2. North America and Europe have similar taste of games while Japan is different, with RPG taken up more marketshare
3. TOP 5 publishers are fighting at all genre, however, they have their advantageous genres


```python

```
