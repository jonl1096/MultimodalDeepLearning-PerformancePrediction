import csv
import sys
csv.field_size_limit(sys.maxsize)
import re
from datetime import date as Date, timedelta, datetime
import enchant
d = enchant.Dict("en_US")

yearrange = (2014,2017)
numword_per_article = 10
'''
def getDomain(url):
	domain_pattern = re.compile("/[^/]+?\.[^/]+?/")
	return re.findall(domain_pattern, url)[0][1:-1]
'''
dumbwords = []
with open('stopwords.txt', 'r') as stopwords:
	for line in stopwords:
		if line[0] != "#":
			dumbwords.append(line.strip())

#domain_map = {}
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
	text = " ".join(text[:numword_per_article])
	#text = " ".join(text)
	#print(text)
	return text

#num_days_to_concat = 3
statsbydate = {}
with open('Data/scrapers/Orioles_train.csv', 'r') as stats, open('Data/scrapers/dates.csv', 'r') as dates:
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

with open('Data/scrapers/Orioles_test.csv', 'r') as stats:
	statsreader = csv.reader(stats)
	next(statsreader)
	for row in statsreader:
		statslist.append(row)
#statsbydate = dict(zip(dateslist,statslist))
print(len(dateslist), len(statslist))
statslist = list(zip(dateslist,statslist))
statslist = sorted(list(statslist), key=lambda x: x[0])
prevdate = 0
for i,(date, stats) in enumerate(statslist):
	print(date)
	if prevdate == date:
		print(date)
		del statslist[i]
	prevdate = date
print(len(statslist))
'''
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
							datadict[datestr] = [stats,articlestr]
						datadict[datestr][1] += articlestr
					tempdate += timedelta(1)
'''

articlesbydate = {}
for year in range(*yearrange):
	with open('Data/news_articles/season'+str(year)+'articles.csv', 'r') as articles:
		articlereader = csv.reader(articles)
		for i,row in enumerate(articlereader):
			date = datetime.strptime(str(row[0]), '%m/%d/%Y').date()
			datestr = str(date.strftime('%Y/%m/%d'))
			#print(str(datestr))
			articlesbydate[datestr] = eval(row[1])

#print(articlesbydate)
#tempdate.strftime('%Y/%m/%d')
#compile articles alongside stats
with open('Data/data_compiled'+str(yearrange[0])[-2:]+'_to_'+str(yearrange[1]-1)[-2:]+'_'+str(numword_per_article)+'w.csv', 'w') as outputfile:
	filewriter = csv.writer(outputfile)
	for i,(date, stats) in enumerate(statslist):
		#print(date)
		if i == len(statslist)-1: break
		row = []
		tempdate = datetime.strptime(date, '%Y/%m/%d').date()
		enddate = datetime.strptime(statslist[i+1][0], '%Y/%m/%d').date()
		if tempdate.strftime('%Y/%m/%d')[:4] != enddate.strftime('%Y/%m/%d')[:4]: continue
		row.append(tempdate.strftime('%Y/%m/%d'))
		articlewords = ""
		while tempdate < enddate:
			#print(tempdate)
			tempdatestr = tempdate.strftime('%Y/%m/%d')
			if tempdatestr in articlesbydate.keys():
				for article in articlesbydate[tempdatestr]:
					articlewords += parse_article_words(str(article[1])) + " "
			tempdate = tempdate+timedelta(1)
		row.append(tempdate.strftime('%Y/%m/%d'))
		row.append(stats[1])
		row.append(articlewords)
		filewriter.writerow(row)
		print(row)
		#print(row)




'''
with open('Data/data'+str(yearrange[0])[-2:]+'_'+str(yearrange[1]-1)[-2:]+'_'+str(numword_per_article)+'.csv', 'w') as data:
#with open('Data/data'+str(yearrange[0])[-2:]+'_'+str(yearrange[1]-1)[-2:]+'.csv', 'w') as data:
	datawriter = csv.writer(data)
	for key, value in sorted(list(datadict.items()), key=lambda x: x[0]):
		datawriter.writerow([key]+value)
'''