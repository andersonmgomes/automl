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

def getCorrHeatMap(ds, annot=True):
    return sns.heatmap(ds.corr(), cmap='coolwarm', fmt='.2f', linewidths=0.1,vmax=1.0, square=True
                       , linecolor='white', robust=True, annot=annot);
    


