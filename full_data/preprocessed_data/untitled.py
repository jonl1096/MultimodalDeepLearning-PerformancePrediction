import pandas as pd
import numpy as np
from datetime import date as Date, timedelta, datetime

df = pd.read_csv("../final_data/statistics_data.csv")
print(df.shape)
for index, row in df.iterrows():
	if(row['previous_date'] == row['original_date']):
		df = df.drop(index)

df.to_csv("statistics_data_nodup.csv", sep=",")
print(df.shape)