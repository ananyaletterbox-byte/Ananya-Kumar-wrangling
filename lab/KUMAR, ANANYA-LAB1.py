#Professor Peng's example code + CHatGPT was used to aid this lab
#ALL GRAPHS AND CHARTS PRODUCED ARE ATTACHED TO THE REPO IN A SEPARATE DOCUMENT

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re

import requests # Page requests

header =  {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
url = 'https://charlottesville.craigslist.org/search/charlottesville-va/boo?lat=37.9923&lon=-78.5028&search_distance=156#search=2~grid~1' #the page we want to scrape
raw = requests.get(url, headers=header) #get page

from bs4 import BeautifulSoup as soup # HTML parser
bsObj = soup(raw.content,'html.parser') # Parse the html
listings = bsObj.find_all(class_="cl-static-search-result") # Find all listings of the kind we want


import re # Regular expressions

brands = [
    'bayliner', 'sea ray', 'wellcraft', 'chaparral', 'cobalt', 'four winns',
    'maxum', 'trophy', 'tracker', 'nitro', 'smokercraft', 'lowe', 'grady', 'grady white',
    'catalina', 'hunter', 'beneteau', 'macgregor', 'tartan', 'hatteras',
    'carver', 'endeavor', 'alberg', 'pearson', 'columbia', 'newport',
    'sea hunt', 'hydrasport', 'hydrasports', 'edgewater', 'parker', 'pioneer',
    'tidewater', 'carolina skiff', 'nauticstar', 'nautic star', 'boston whaler', 'whaler',
    'sylvan', 'godfrey', 'aqua patio', 'sun tracker', 'bass buggy', 'bennington',
    'suncatcher', 'harris', 'sunchaser', 'gibson', 'harbor master',
    'baja', 'donzi', 'checkmate', 'fountain', 'ranger', 'correct craft', 'nautique',
    'yamaha', 'sea doo', 'seadoo', 'kawasaki', 'hobie', 'pelican', 'old town',
    'jackson', 'nucanoe', 'oru'
]

brands = sorted(brands, key=len, reverse=True)
brands += [
    'perception',
    'viscaya',
    'mercury',   # engines show up a lot
    'merc',
    'inflatable'
]

def detect_type(title):
    if 'kayak' in title:
        return 'kayak'
    if 'canoe' in title:
        return 'canoe'
    if 'inflatable' in title:
        return 'inflatable'
    return 'boat'

boat_type = detect_type('title')

data = [] # We'll save our listings in this object
brands = sorted(brands, key=len, reverse=True)

def extract_year(title):
    # full 4-digit years
    m = re.search(r'\b(19[7-9][0-9]|20[0-2][0-9])\b', title)
    if m:
        return int(m.group(1))
    # short years like '03 or 99
    m = re.search(r"\b'?(0[0-9]|[1-9][0-9])\b", title)
    if m:
        y = int(m.group(1))
        return 2000 + y if y < 30 else 1900 + y
    return np.nan

def detect_type(title):
    if 'kayak' in title:
        return 'kayak'
    if 'canoe' in title:
        return 'canoe'
    if 'inflatable' in title:
        return 'inflatable'
    return 'boat'

data = []

for k in range(len(listings)):
    title = listings[k].find('div', class_='title').get_text().lower()
    price = listings[k].find('div', class_='price').get_text()
    link = listings[k].find(href=True)['href']
    # brand detection
    hits = [b for b in brands if b in title]
    brand = hits[0] if hits else 'missing'
    # year extraction
    year = extract_year(title)
    # type detection
    boat_type = detect_type(title)
    # kill fake years for kayaks etc
    if boat_type != 'boat':
        year = np.nan
    data.append({
        'title': title,
        'price': price,
        'year': year,
        'link': link,
        'brand': brand,
        'type': boat_type
    })


## Wrangle the data
df = pd.DataFrame.from_dict(data)
df['price'] = df['price'].str.replace('$','')
df['price'] = df['price'].str.replace(',','')
df['price'] = pd.to_numeric(df['price'],errors='coerce')
df['year'] = pd.to_numeric(df['year'],errors='coerce')
df['age'] = 2025-df['year']
print(df.shape)
df.to_csv('craigslist_cville_boats.csv') # Save data in case of a disaster
df.head()

#avoiding truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head())


#price histogram + filtering
prices = df['price'].dropna()
prices = prices[(prices > 100) & (prices < 300000)]

print(prices.describe())

plt.figure(figsize=(8,5))
plt.hist(prices, bins=50)
plt.xlabel('Price ($)')
plt.ylabel('Count')
plt.title('Craigslist Charlottesville Boat Prices (Filtered)')
plt.show()

#ages histogram
print(df['age'].describe())
df['age'].hist(grid=False)
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Craigslist Charlottesville Boat Ages')
plt.show()

#type histogram
print(df['type'].describe())
df['type'].hist(grid=False)
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('Craigslist Charlottesville Boat Types')
plt.show()

#price by brand
df.loc[:,['price','brand']].groupby('brand').describe()
#age by brand
df.loc[:,['age','brand']].groupby('brand').describe()
df.plot.scatter('age','price')

ax = sns.scatterplot(data=df, x='age', y='price',hue='brand')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.show()

df['log_price'] = np.log(df['price'])
df['log_age'] = np.log(df['age'])

ax = sns.scatterplot(data=df, x='log_age', y='log_price',hue='brand')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.show()

print(df.loc[:,['log_price','log_age']].cov())
print(df.loc[:,['log_price','log_age']].corr())