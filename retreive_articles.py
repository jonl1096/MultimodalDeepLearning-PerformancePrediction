from bs4 import BeautifulSoup
import re
from datetime import date as Date, timedelta
try:
    # Python 2.6-2.7 
	from HTMLParser import HTMLParser
except ImportError:
	# Python 3
	from html.parser import HTMLParser
import csv
import requests

#NOTE: google results doesn't actually do this by the right date.
#  It seems that the browser opening a url from python gives 
#  a different result html with an unspecified date.
#  Tricky, might come back later
def get_google_results(search_term, date, num_results):
	term = search_term
	mindate = date
	maxdate = date

	urlterm = term.replace(" ","+")
	google_search_news = "https://www.google.com/search?num="+str(num_results)+"&q="+urlterm+"&tbs=cdr:1,cd_min:"+mindate+",cd_max:"+maxdate+",sbd:1&tbm=nws"
	#google_search_news = "https://www.google.com/search?q=news&hl=en&biw=1491&bih=752&pws=1&source=lnt&tbs=cdr%3A1%2Ccd_min%3A4%2F19%2F2017%2Ccd_max%3A4%2F19%2F2017&tbm=nws"
	#google_search_news = "https://www.bing.com/search?q=red+sox&filters=ex1%3a%22ez5_17260_17260%22&qpvt=red+sox"
	#google_search_news = "https://www.bing.com/news/search?q=red+sox&FORM=HDRSC6"
	#google_search_news = "https://www.bing.com/search?q=kjl&filters=ex1%3a%22ez5_17259_17260%22&qs=n&sp=-1&pq=kjl&sc=9-3&cvid=34653109728C4B5A9CAD764726F8DAD9&qpvt=kjl"
	print(google_search_news)
	print("")

	page = requests.get(google_search_news)
	results = BeautifulSoup(page.content, 'html.parser')
	#print(results_html)
	#print("")

	results = results.findAll('div', attrs={'id':'search'})[0]

	#links
	links = results.findAll('h3', attrs={'class':'r'})

	#dates
	dates = results.findAll('span')
	print(dates)
	print("")

	link_pattern = re.compile("\?q=.+?\"")
	description_pattern = re.compile("\"\>.+?\</a\>")
	urls = []
	for linktext in links:
		linktext = BeautifulSoup(str(linktext), "lxml").findAll('a')[0]

		#find link
		link = re.findall(link_pattern,str(linktext))[0][3:-1]
		link = HTMLParser().unescape(link)

		#find description
		description = re.findall(description_pattern,str(linktext))[0][1:-1]
		description = description[1:-3].replace("<b>","").replace("</b>","")

		urls.append((description,link))

	return urls

def get_bing_results(search_term, date, num_results):
	#get search term
	term = search_term
	urlterm = term.replace(" ","+")


	#calculate date

	#17257 = 4/1/2017
	date_base = Date(2017, 4, 1)
	date_base_num = 17257

	date_split = date.split("/")
	date_diff = date_base - Date(int(date_split[2]),int(date_split[0]),int(date_split[1]))

	mindate_num = date_base_num - date_diff.days
	maxdate_num = mindate_num


	bing_search_news = "https://www.bing.com/search?q="+urlterm+"&filters=ex1%3a%22ez5_"+str(mindate_num)+"_"+str(maxdate_num)+"%22&qs=n&sp=-1&count="+str(num_results)
	
	print(bing_search_news)
	print("")

	page = requests.get(bing_search_news)
	results = BeautifulSoup(page.content, 'html.parser')

	results = results.findAll('li', attrs={'class':'b_algo'})
	results = BeautifulSoup(str(results), "lxml").findAll('h2')

	url_pattern = re.compile("href=\".+?\"")
	des_pattern = re.compile("\"\>.+?\</a\>")
	webpages = []
	for i,result in enumerate(results):
		result = str(result)

		#get url
		url = re.findall(url_pattern,result)
		url = HTMLParser().unescape(url[0][6:-1])

		#get description
		description = re.findall(des_pattern,result)
		description = description[0][2:-4].replace("<strong>","").replace("</strong>","")

		webpages.append((description,url,i))
	return webpages



#urls = get_google_results("red sox", "06/06/2012", 2)
#urls = get_bing_results("red sox", "06/06/2012", 2)
#print(urls)

#data is a list of lists
#each list is [date, list0] where
#each list0 is a list of urls of articles on the day
#in the form of (description, url)

def get_season_articles(search_term, num_per_day, start_date, end_date):
	data = []
	currdate = start_date
	while currdate <= end_date:
		datestr = currdate.strftime('%m/%d/%Y')
		urls = get_bing_results(search_term, datestr, num_per_day)
		data.append([datestr,urls])
		currdate = currdate + timedelta(1)
	return data

def write_season_csv(year, data):
	with open('Data/season'+year+'extended.csv', 'w') as csvfile:
		urlwriter = csv.writer(csvfile)
		#urlwriter.writerow(["date", "articles"])
		for item in data:
			urlwriter.writerow(item)

write_season_csv("2016websites", get_season_articles("baltimore orioles baseball", 10, Date(2016, 3, 3), Date(2016, 10, 2)))
write_season_csv("2015websites", get_season_articles("baltimore orioles baseball", 10, Date(2015, 3, 3), Date(2015, 10, 4)))
write_season_csv("2014websites", get_season_articles("baltimore orioles baseball", 10, Date(2014, 3, 1), Date(2014, 10, 15)))
#write_season_csv("2013websites", get_season_articles("baltimore orioles baseball", 10, Date(2013, 3, 31), Date(2013, 9, 30)))
#write_season_csv("2012websites", get_season_articles("baltimore orioles baseball", 10, Date(2012, 3, 28), Date(2012, 10, 3)))
#write_season_csv("2011websites", get_season_articles("baltimore orioles baseball", 10, Date(2011, 3, 31), Date(2011, 9, 28)))

