import pandas as pd
import seaborn as sns
import numpy as np

DS_PATH = 'datasets/'

#load dataset
def getDSFuelConsumptionCo2():
    return pd.read_csv(DS_PATH + 'FuelConsumptionCo2.csv')

def getDSPriceHousing():
    return pd.read_csv(DS_PATH + 'USA-priceHousing.csv')

#classification problem
def getDSPriceHousing_ClassProb():
    ds_house_classprob = getDSPriceHousing()
    price_75 = ds_house_classprob.Price.describe()['75%']
    ds_house_classprob['high_price'] = ds_house_classprob['Price']>price_75
    ds_house_classprob.drop('Price', axis=1, inplace=True)
    return ds_house_classprob

def getDSWine_RED():
    return pd.read_csv(DS_PATH + 'winequality-red.csv', sep=';')

def getDSWine_RED_ClassProb():
    df = getDSWine_RED()
    threashold = df.quality.describe()['75%']
    df['high_quality'] = 0
    for i, row in df.iterrows():
        if row.quality > threashold:
            df.loc[i, 'high_quality'] = 1
    
    return df.drop('quality', axis=1)

def getCorrHeatMap(ds, annot=True):
    return sns.heatmap(ds.corr(), cmap='coolwarm', fmt='.2f', linewidths=0.1,vmax=1.0, square=True
                       , linecolor='white', robust=True, annot=annot);
    
def getDSIris():
    return pd.read_csv(DS_PATH + 'iris.csv', header=None
                       , names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])

