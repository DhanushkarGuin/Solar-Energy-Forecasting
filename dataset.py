## Importing Dataset
import pandas as pd

generation_ds = pd.read_csv('original_Dataset/Plant_2_Generation_Data.csv')
weather_ds = pd.read_csv('original_Dataset/Plant_2_Weather_Sensor_Data.csv')

## Merging Datasets
solar_ds = pd.merge(generation_ds.drop(columns = ['PLANT_ID']), weather_ds.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')

solar_ds = solar_ds.drop(columns=['SOURCE_KEY','AC_POWER'])

## Converting Dates to Numerics
solar_ds['DATE_TIME'] = pd.to_datetime(solar_ds['DATE_TIME'],format = '%Y-%m-%d %H:%M:%S')

## Separating Dates and Times
solar_ds['DATE'] = pd.to_datetime(solar_ds['DATE_TIME']).dt.date
solar_ds['TIME'] = pd.to_datetime(solar_ds['DATE_TIME']).dt.time
solar_ds['YEAR'] = pd.to_datetime(solar_ds['DATE_TIME']).dt.year
solar_ds['MONTH'] = pd.to_datetime(solar_ds['DATE_TIME']).dt.month
solar_ds['DAY'] = pd.to_datetime(solar_ds['DATE_TIME']).dt.day

solar_ds = solar_ds.drop(columns=['DATE_TIME'])

solar_ds['HOURS'] = pd.to_datetime(solar_ds['TIME'], format='%H:%M:%S').dt.hour
solar_ds['MINUTES'] = pd.to_datetime(solar_ds['TIME'], format='%H:%M:%S').dt.minute
solar_ds['TOTAL MINUTES PASS'] = solar_ds['MINUTES'] + solar_ds['HOURS']*60

# print(solar_ds.isnull().sum())
# No missing values

solar_ds.to_csv('dataset.csv', index=False)