import csv

articles = []
with open('season2016.csv', 'r') as csvfile:
	urlreader = csv.reader(csvfile)
	urlreader.next()
	for row in urlreader:
		#print(row)
		articles.append((row[0],eval(row[1])))

print(articles)