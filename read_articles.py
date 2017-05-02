import csv
import mechanize
from bs4 import BeautifulSoup
import mechanize
import requests
import re

def getDomain(url):
	domain_pattern = re.compile("/[^/]+?\.[^/]+?/")
	return re.findall(domain_pattern, url)[0][1:-1]

def read_domains(domainfile):
	domains = set()
	with open(domainfile,'r') as file:
		for line in file:
			domains.add(line.strip())
	return domains

def parse_page(url, articletag):
	page = requests.get(url)
	soup = BeautifulSoup(page.content, 'html.parser')
	if str(soup) == "Forbidden":
		#print(soup, url)
		br = mechanize.Browser()
		br.set_handle_robots(False)
		br.addheaders = [('User-agent','chrome')]
		try:
			webpage = br.open(url).read()
		except Exception:
			return None
		soup = BeautifulSoup(webpage, 'html.parser')
	soup = soup.find_all(articletag[0], attrs=articletag[1])
	#print(soup)
	if len(soup) == 0:
		return None
	tempsoup = soup[0].find_all('p')
	if len(tempsoup) == 0:
		tempsoup = soup[0].find_all('br')
	text = ""
	for par in soup:
		text += par.get_text().encode('utf-8') + " "
	text = text.replace('\n', ' ')
	#print(text)
	return text

def get_article_text(article):
	description = article[0]
	url = article[1]

	domain = getDomain(url)
	print(url)
	if domain == 'www.baltimoresun.com':
		text = parse_page(url, ('div',{'itemprop':'articleBody'}))
	if domain == 'articles.baltimoresun.com':
		text = parse_page(url, ('div',{'id':'area-center-w-left'}))
		#NOT COMPLETELY RIGHT
	elif domain == 'www.camdenchat.com':
		text = parse_page(url, ('div',{'class':'c-entry-content'}))
	elif domain == 'www.yahoo.com':
		text = parse_page(url, ('article',{'data-type':'story'}))
	elif domain == 'baltimore.cbslocal.com':
		text = parse_page(url, ('div',{'class':'story'}))
	elif domain == 'www.nytimes.com':
		text = parse_page(url, ('article',{'id':'story'}))
	elif domain == 'www.usatoday.com':
		pattern = re.compile("/20[0-9][0-9]/[0-9][0-9]/")
		if len(re.findall(pattern,url)) != 0:
			text = parse_page(url, ('div',{'itemprop':'articleBody'}))
		else:
			text = parse_page(url, ('div',{'class':'story-asset'}))
	elif domain == 'www.foxsports.com':
		text = parse_page(url, ('div',{'itemprop':'articleBody'}))
	# elif domain == 'm.mlb.com':
	# 	pattern = re.compile("/final")
	# 	if len(re.findall(pattern,url)) != 0:
	# 		text = parse_page(url, ('p',{'class':'blurb-text'}))
	# 		#always returns None
	# 		#NEED TO FIX THIS, NOT SURE HOW
	# 	else:
	# 		text = parse_page(url, ('div',{'class':'body'}))
	# elif domain == 'www.espn.com':
	# 	text = parse_page(url, ('div',{'class':'article-body'}))
	elif domain == 'thebaltimorewire.com':
		text = parse_page(url, ('section',{'class':'article-content'}))
	elif domain == 'birdswatcher.com':
		text = parse_page(url, ('section',{'class':'article-content'}))
	else:
		text = None

	if text == "":
		print("NOT YET PARSED")
		return description, None, domain

	return description, text, domain

yearrange = (2011,2017)
i = 0
for year in range(*yearrange):
	dates = []
	articles = [] #per season per date
	with open('Data/season'+str(year)+'websites.csv','r') as file:
		reader = csv.reader(file)
		for row in reader:
			dates.append(row[0])
			articles.append([])
			for article in eval(row[1]):
				description, text, domain = get_article_text(article)
				if text != None:
					dateidx = len(articles) - 1
					articles[dateidx].append((description, text, domain))
	
	with open('Data/season'+str(year)+'articles.csv','w') as file:
		writer = csv.writer(file)
		for i,date in enumerate(dates):
			writer.writerow([date,articles[i]])
	

#print(articles)
print("done")