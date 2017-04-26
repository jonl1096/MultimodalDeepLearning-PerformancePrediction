import csv

articles = []
with open('season2016articles.csv', 'r') as csvfile:
	urlreader = csv.reader(csvfile)
	for row in urlreader:
		#print(row)
		articles.append((row[0],eval(row[1])))

print(articles)
