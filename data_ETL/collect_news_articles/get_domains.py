import csv
import re
import os.path

numdomains = 10
yearrange = (2011,2017)
domainfilename = 'domains.txt'

def get_domains(filename):
	articles = []
	with open(filename, 'r') as csvfile:
		urlreader = csv.reader(csvfile)
		for row in urlreader:
			#print(row)
			articles.append((row[0],eval(row[1])))

	domain_pattern = re.compile("/[^/]+?\.[^/]+?/")
	domain_names = {}
	for articlesfordate in articles:
		for url in articlesfordate[1]:
			domainname = re.findall(domain_pattern,url[1])[0][1:-1]
			if domainname not in domain_names.keys():
				domain_names[domainname] = 0
			domain_names[domainname] += 1

	domain_names_tuples = domain_names.items()
	domain_names_tuples = sorted(domain_names_tuples, key=lambda x: -x[1])
	return domain_names_tuples

finaldomains = {}
for year in range(*yearrange):
	domains = get_domains('Data/season'+str(year)+'websites.csv')
	for domain in domains:
		if domain[0] not in finaldomains.keys():
			finaldomains[domain[0]] = domain[1]
		else:
			finaldomains[domain[0]] += domain[1]
	print(float(sum(d[1] for d in domains[:numdomains]))/sum(d[1] for d in domains))
finaldomains = sorted(list(finaldomains.items()), key=lambda x: -x[1])
print(finaldomains[:numdomains])

print(float(sum(d[1] for d in finaldomains[:numdomains]))/sum(d[1] for d in finaldomains))

#manually edit
tempfinaldomains = dict(finaldomains)
del tempfinaldomains['www.youtube.com']
del tempfinaldomains['www.milb.com']
del tempfinaldomains['www.baseball-reference.com']
del tempfinaldomains['www.baseballamerica.com']
del tempfinaldomains['www.cbssports.com']
tempfinaldomains = sorted(list(tempfinaldomains.items()), key=lambda x: -x[1])
print(tempfinaldomains[:numdomains])
print(float(sum(d[1] for d in tempfinaldomains[:numdomains]))/sum(d[1] for d in tempfinaldomains))

finaldomains_selected = tempfinaldomains[:numdomains]


with open(domainfilename, 'w') as file:
	for domain in finaldomains_selected:
		file.write(domain[0]+'\n')

