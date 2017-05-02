import csv
import re

yearrange = (2011,2012)

def getDomain(url):
	domain_pattern = re.compile("/[^/]+?\.[^/]+?/")
	return re.findall(domain_pattern, url)[0][1:-1]

domain_map = {}
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
					row = str(domain_map[article[2]])+" "+str(article[1])
					training.write(row+"\n")
					training_byday.write(row+"\n")