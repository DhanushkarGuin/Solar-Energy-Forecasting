## Importing Dataset
import pandas as pd

generation_ds = pd.read_csv('dataset/Plant_2_Generation_Data.csv')
weather_ds = pd.read_csv('dataset/Plant_2_Weather_Sensor_Data.csv')

## Merging Datasets
solar_ds = pd.merge(generation_ds.drop(columns = ['PLANT_ID']), weather_ds.drop(columns = ['PLANT_ID', 'SOURCE_KEY']), on='DATE_TIME')
solar_ds.to_csv('dataset.csv', index=False)