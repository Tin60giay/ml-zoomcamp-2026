import pandas as pd
import numpy as np
from pathlib import Path


#Q1
print(pd.__version__)


df = pd.read_csv(Path(__file__).parent/"data"/"car_fuel_efficiency_2026.csv")
print(df.head())

#Q2
print(len(df))


#Q3
print(df["fuel_type"].nunique())

#Q4
print(df.isnull().sum())

#Q5
print(df.loc[df["origin"]=="Asia", "fuel_efficiency_mpg"].max())

#Q6
horsepower_before = df["horsepower"].median()
common = df["horsepower"].mode().iloc[0]
horsepower_after = df["horsepower"].fillna(common).median()
print(horsepower_before)
print(horsepower_after)


#Q7
X = df.loc[df["origin"]=="Asia", ["vehicle_weight", "model_year"]].head(7).to_numpy()
XTX = X.T @ X
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = np.linalg.inv(XTX) @ X.T @ y
print(w.sum())