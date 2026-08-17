import pandas as pd
import numpy as np

print("-"*50)
print("\n 1.Data Cleaning")

df=pd.read_csv(r'C:\Users\ERAI ALLEN\Downloads\retail_sales.csv')
print(df.head())

print("\n[1.1] MISSING VALUES")
print("-" * 50)
missing_rows = df[df.isna().any(axis=1)]
print(missing_rows)

print("\n[1.2] CATEGORY CLEANING")
print("-" * 50)

print("Unique values before cleaning:")
print(df['Category'].unique())

df['Category']=df['Category'].replace(['NaN?','Null','Nan'],np.nan)
df['Category']=df['Category'].fillna('Unknown')

print("\nUnique values after cleaning:")
print(df['Category'].unique())

print("\n[1.3] SALES DATA CLEANING")
print("-" * 50)

print("Missing Sales values removed:")
print(df.dropna(subset=['Sales'],inplace=True))
print("Zero Sales records removed:")
print(df.drop(df[df['Sales']==0].index,inplace=True))
print("Remaining zero Sales records:")
print((df['Sales']==0).sum())

print("\n[1.4] REGION CLEANING")
print("-" * 50)

print("Unique values before cleaning:")
print(df['Region'].unique())

df['Region']=df['Region'].replace(['Nan'],np.nan)
df['Region']=df['Region'].fillna('Unknown')

print("\nUnique values after cleaning:")
print(df['Region'].unique())

print("\n[1.5] QUANTITY CLEANING")
print("-" * 50)

df['Quantity'] = df['Quantity'].fillna(df['Quantity'].mean())

print("Unique Quantity values:")
print(df['Quantity'].unique())

print("\n[1.6] DATA TYPE VALIDATION")
print("-" * 50)

print(df['Sales'].dtype)


print("-"*50)
print("\n 2.Exploratory Data Analysis")
print("-"*50)

print("\nDataset Information:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe().round(2))

print("\nUnique Values:")
print(df.nunique())

print("\nDate Range:")
print(df['Date'].max())
print(df['Date'].min())

print("-"*50)
print("\n 3.Sales Analysis")
print("-"*50)

print("\nTotal Sales:")
print(df['Sales'].sum())

print("\nAverage Sales:")
print(df['Sales'].mean())

print("\nMaximum Sales:")
print(df['Sales'].max())

print("\nMinimum Sales:")
print(df['Sales'].min())

print("-"*50)
print("\n 4.Category Analysis")
print("-"*50)

print("\nTotal Sales by Category:")
print(df.groupby('Category')['Sales'].sum().sort_values(ascending=False))

print("\nMaximum Sales by Category:")
print(df.groupby('Category')['Sales'].max().sort_values(ascending=False))

print("\nMinimum Sales by Category:")
print(df.groupby('Category')['Sales'].min().sort_values(ascending=False))

print("-"*50)
print("\n 5.Profit Analysis")
print("-"*50)

print("\nTotal Profit by Category:")
print(df.groupby('Category')['Profit'].sum().sort_values(ascending=False))

print("\nMaximum Profit by Category:")
print(df.groupby('Category')['Profit'].max().sort_values(ascending=False))

print("\nMinimum Profit by Category:")
print(df.groupby('Category')['Profit'].min().sort_values(ascending=False))

print("-"*50)
print("\n 6.Region Analysis")
print("-"*50)

print("\nTotal Sales by Region:")
print(df.groupby('Region')['Sales'].sum())

print("\nMaximum Sales by Region:")
print(df.groupby('Region')['Sales'].max())

print("\nMinimum Sales by Region:")
print(df.groupby('Region')['Sales'].min())

print("-"*50)
print("\n 7.Quantity Analysis")
print("-"*50)

print("\nTotal Quantity by Category:")
print(df.groupby('Category')['Quantity'].sum().sort_values(ascending=False))

print("\nMaximum Quantity by Category:")
print(df.groupby('Category')['Quantity'].max().sort_values(ascending=False))

print("\nMinimum Quantity by Category:")
print(df.groupby('Category')['Quantity'].min().sort_values(ascending=False))

print("-"*50)
print("\n 8.Time Series Analysis & Monthly Trends")
print("-"*50)

print("\n 8.1 Monthly Sales Trends")
print("-"*50)

df['Date']=pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
print(df.groupby('Month')['Sales'].sum())

print("\n 8.2 Daily Sales Trends")
daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
print(daily_sales)

print("-"*50)
print("\n 9.Sales Forecasting - Data Preparation")

daily_sales.rename(columns={
    'Date':'ds',
    'Sales':'y'
}, inplace=True)

train_size = int(len(daily_sales) * 0.8)

train = daily_sales[:train_size]
test = daily_sales[train_size:]

print("-"*50)
print("10. SALES FORECASTING")
print("-"*50)

print("\n 10.1 Train Prophet Model")
print("-"*50)

from prophet import Prophet

model = Prophet()
model.fit(train)

print("\n 10.2 Generate Future Dates")
print("-"*50)

future = model.make_future_dataframe(periods=len(test))
print(future.tail())

print("\n 10.3 Generate Forecast")
print("-"*50)

forecast = model.predict(future)

predictions = forecast[['ds', 'yhat']].tail(len(test))

print("\nForecasted Sales:")
print(predictions.head())

print("-"*50)
print("11. FORECAST MODEL EVALUATION")
print("-"*50)

print("\n 11.1 Actual Vs Predicted Sales")
print("-"*50)

evaluation = test.merge(predictions, on='ds')

print(evaluation.head())

print("\n 11.2 Model Performance Metrics")
print("-"*50)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

mae = mean_absolute_error(evaluation['y'], evaluation['yhat'])

print(mae)

rmse = np.sqrt(mean_squared_error(
    evaluation['y'],
    evaluation['yhat']
))

print(rmse)

mape = np.mean(
    np.abs(
        (evaluation['y'] - evaluation['yhat'])
        / evaluation['y']
    )
) * 100

print(mape)

print("-"*50)
print("12. Future Sales Forecast")
print("-"*50)

print("\n 12.1 Train Model Using Complete Historical Data")
print("-"*50)

model = Prophet()
model.fit(daily_sales)

print("\n 12.2 Forecast Next 30 Days")
print("-"*50)

future = model.make_future_dataframe(periods=30)

forecast = model.predict(future)
print(future.tail())

print("\n 12.3 Display Next 30 Days Forecast")
print("-"*50)

model.plot(forecast)

model.plot_components(forecast)

print(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(30))

print("\n 12.4 Forecast Visualization")
print("-"*50)
from matplotlib import pyplot as plt

model.plot(forecast)
plt.title("30-Day Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

print("\n 12.5 Forecast Components")
print("-"*50)

model.plot_components(forecast)
plt.show()

df.to_csv("forecasting sales.csv", index=False)

import os

print(os.getcwd())












