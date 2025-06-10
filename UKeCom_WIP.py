# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 01:01:07 2025

@author: tapad
"""

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'D:/DS profiling/Projects/DataExplorationsVisualisation/UKeCommerceKaggle/data1.csv'
df = pd.read_csv(file_path, encoding='latin1')

df.head()
df.info()
df.describe() 

#DATA CLEANING

#Identify Columns with Missing Values

df.isnull().sum()
'''
Output:
    df.isnull().sum()
    Out[7]: 
    InvoiceNo           0
    StockCode           0
    Description      1454
    Quantity            0
    InvoiceDate         0
    UnitPrice           0
    CustomerID     135080
    Country             0
    dtype: int64
'''
'''Description has 1,454 missing values & CustomerID has 135,080 missing values'''

#Check the Percentage of Missing Values: to understand the scale of missingness:
#This helps assess whether to drop rows/columns or whether you need to impute values.

missing_percent = df.isnull().mean() * 100
print(missing_percent.map("{:.2f}%".format))
#Alternative way:
#formatted_missing = missing_percent.apply(lambda x: f"{x:.2f}%")
#print(formatted_missing)
'''
Description     0.27%
CustomerID     24.93%
'''

#Check Number of Rows with Any Missing Value
#This tells how many rows have at least one missing value.
df.isnull().any(axis=1).sum()
'''135080'''

# May flag the missing CustomerID as a separate customer segment (CustomerID = 'Guest').

#Types of Missing Data:
#MCAR (Missing Completely At Random): No pattern; truly random. Ex: A survey respondent skips a random question.
#MAR (Missing At Random)	Related to observed data	Males are more likely to skip income info
#MNAR (Missing Not At Random)	Related to unobserved/missing data itself	High-income people avoid answering income questions

#Statistical models assume MCAR or MAR for unbiased estimation.

#Visualize Missing Data (Optional but Very Helpful)
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

df.isnull().sum().plot(kind='bar')
plt.title("Missing Values per Column")
plt.ylabel("Count")
plt.show()

# Fill or drop missing values
df.fillna(method='ffill', inplace=True)
# or
df.dropna()
# Handle data types
df['date_column'] = pd.to_datetime(df['date_column'])
# Remove duplicates
df.drop_duplicates(inplace=True)

missing_rows = df[df.isnull().any(axis=1)]

# Print first 10 such rows
print(missing_rows.head(10))








