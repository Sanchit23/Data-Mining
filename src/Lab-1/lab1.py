from bs4 import BeautifulSoup
from os import listdir

dir_path = "/home/0/srini/WWW/674/public/reuters/"

class preProcessor:
	documents = []

	def getRawDocuments(self):
		for file in listdir(dir_path):
			xmlFiles = BeautifulSoup(open(dir_path+file),"html.parser")
			print len(xmlFiles.find_all('topics'))
		return

	def process(self):
		return

preProcessor().getRawDocuments()