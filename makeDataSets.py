import csv
import sys
csv.field_size_limit(sys.maxsize)
import re
from datetime import date as Date, timedelta, datetime
import enchant
d = enchant.Dict("en_US")

yearrange = (2014,2016)

def getDomain(url):
	domain_pattern = re.compile("/[^/]+?\.[^/]+?/")
	return re.findall(domain_pattern, url)[0][1:-1]

dumbwords = []
with open('stopwords.txt', 'r') as stopwords:
	for line in stopwords:
		if line[0] != "#":
			dumbwords.append(line.strip())

domain_map = {}
'''
for year in range(*yearrange):
	with open('Data/season'+str(year)+'articles.csv', 'r') as file, open('Data/season'+str(year)+'formclda.txt', 'w') as training:
		reader = csv.reader(file)
		for day,row in enumerate(reader):
			articles = eval(row[1])
			with open('Data/season'+str(year)+'formclda/day'+str(day)+'.txt', 'w') as training_byday:
				print(len(articles))
				for article in articles:
					if article[2] not in domain_map.keys():
						domain_map[article[2]] = len(domain_map)

					#select proper words
					text = re.findall("[\w]*", str(article[1]))
					text = [word for i,word in enumerate(text) if len(word) > 1 and (word[0].isupper() or d.check(word))]
					text = [word.strip().lower() for word in text]
					text = [word for word in text if word not in dumbwords]


					row = str(domain_map[article[2]])+" "+" ".join(text)
					training.write(row+"\n")
					training_byday.write(row+"\n")
'''

def parse_article_words(article):
	text = re.findall("[\w]*", str(article))
	text = [word for i,word in enumerate(text) if len(word) > 2 and (word[0].isupper() or d.check(word))]
	text = [word.strip().lower() for word in text]
	text = [word for word in text if word not in dumbwords]
	text = " ".join(text)
	#print(text)
	return text

num_days_to_concat = 3
statsbydate = {}
with open('Data/scrapers/Orioles_train.csv', 'r') as stats, open('Data/scrapers/Orioles_dates.csv', 'r') as dates:
	statsreader = csv.reader(stats)
	datesreader = csv.reader(dates)
	next(statsreader)
	next(datesreader)
	statslist = []
	dateslist = []
	for row in statsreader:
		statslist.append(row)
	for row in datesreader:
		dateslist.append(str(row[0]))
	statsbydate = dict(zip(dateslist,statslist))
#print(statsbydate)

datadict = {}
for year in range(*yearrange):
	with open('Data/news_articles/season'+str(year)+'articles.csv', 'r') as articles:
		articlereader = csv.reader(articles)
		for i,row in enumerate(articlereader):
			#if i == 30: break
			if len(row) > 1:
				articles = eval(str(row[1]))
				#print(articles)
				print(str(row[0]))
				tempdate = datetime.strptime(str(row[0]), '%m/%d/%Y').date()
				end_date = tempdate+timedelta(num_days_to_concat)
				while tempdate < end_date:
					#print(tempdate, end_date-timedelta(1))
					datestr = tempdate.strftime('%Y/%m/%d')
					if datestr in statsbydate.keys():
						articlestr = ""
						for article in articles:
							articlestr += parse_article_words(str(article[1])) + " "
						if datestr not in datadict.keys():
							stats = statsbydate[datestr]
							datadict[datestr] = stats+[articlestr]
						datadict[datestr][len(datadict[datestr])-1] += articlestr
					tempdate += timedelta(1)


with open('Data/data'+str(yearrange[0])[-2:]+'_'+str(yearrange[1])[-2:]+'.csv', 'w') as data:
	datawriter = csv.writer(data)
	for key, value in sorted(list(datadict.items()), key=lambda x: x[0]):
		datawriter.writerow([key]+value)
