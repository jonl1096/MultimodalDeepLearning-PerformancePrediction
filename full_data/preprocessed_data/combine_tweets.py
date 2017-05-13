import pandas as pd
import numpy as np
from datetime import date as Date, timedelta, datetime

df = pd.read_csv("orioles_tweets_LDA.csv")
dates = pd.read_csv("../final_data/dates.csv")

previous_date_list = dates['previous_date']
original_date_list = dates['original_date']

# m/d/year -> year-month-date
def convert_date_format(old_date_list):
	new_date_list = []
	for date in old_date_list:
		month, date, year = date.split('/')

		new_month = month.zfill(2)
		new_date = date.zfill(2)
		new_year = "20" + year
		new_date_list.append(new_year + "-" + new_month + "-" + new_date)
	return new_date_list

previous_date_list = convert_date_format(previous_date_list)
original_date_list = convert_date_format(original_date_list)

print(original_date_list[:10])
prev_date = previous_date_list.pop(0)
curr_date = original_date_list.pop(0)

count = 0
lda_vector = 0
avg_vectors = []

write_df = pd.DataFrame(columns=['date', 'tweet_lda'])

for index, row in df.iterrows():
	print("DF:" + row['date'])
	lda_vector += row[1:]
	count+=1

	if row['date'][3] != curr_date[3] or row['date'] == curr_date:
		avg_vectors.append((curr_date, [i for i in lda_vector / count]))
		count = 0
		lda_vector = 0
		curr_date = original_date_list.pop(0)

#avg_vectors.append((curr_date, lda_vector / count))

df = pd.DataFrame(avg_vectors)
df.to_csv("lda_output.csv")
print(df.shape)


