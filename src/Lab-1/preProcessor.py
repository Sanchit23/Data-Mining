import time
from os import listdir
from bs4 import BeautifulSoup
from string import punctuation
from nltk import PorterStemmer
from nltk.corpus import stopwords

# path of the reuters data set
dir_path = "/home/0/srini/WWW/674/public/reuters/"

# variable to hold all the processed documents
documents = []

# class for fetching the files and procesing them.
class preProcessor:
	begin_time = time.time()

	# method to get documents from reuters files and process them
	def getDocuments(self):
		
		# variable for keeping track of the number of documents
		counter = -1

		# iterate over all files in the reuters dataset directory
		for file in listdir(dir_path):
			print "\n----- processing file: "+ file +" -----\n"

			# open file with Beautiful Soup parser
			xmlFiles = BeautifulSoup(open(dir_path+file),"html.parser")
			
			# iterate over each document in file
			for document in xmlFiles.find_all('reuters'):
				counter = counter + 1;

				# call methods to extract information from <topics>, <places>, <title> and <body> tags.
				self.extract(document,counter)

		print str(len(documents)) + " \n documents found and processed"
		# print "time taken: " + str((time.time() - (self.begin_time)))

		return

	# method to extract information from <topics> tag 
	def extract(self,document,counter):
		topics = document.find('topics')
		places = document.find('places')
		title = document.find('title')
		body = document.find('body')

		if not topics:
			documents.append({"topics":""})
		else:
			documents.append({"topics":(','.join([word for word in document.find('topics').stripped_strings]))})

		if not title:
			documents[counter]["title"] = ""
		else:
			documents[counter]["title"] = ','.join([word for word in document.find('title').stripped_strings])

		if not places:
			documents[counter]["places"] = ""
		else:
			documents[counter]["places"] = ','.join([word for word in document.find('places').stripped_strings])

		if not body:
			documents[counter]["body"] = ""
		else:
			documents[counter]["body"] = ','.join([self.process(word) for word in document.find('body').stripped_strings])

		return

	# method to process the raw data. Removes punctuation and stop words. Also stems words and capitalizes all letters.
	def process(self,s):
		return self.stemWords(self.strip_punctuation(self.strip_stopwords(s)))

	# method to remove punctuation
	def strip_punctuation(self,s):
		return ''.join(c for c in s if c not in punctuation)

	# method to remove stop words
	def strip_stopwords(self,s):
		stop_words = stopwords.words("english")
		additional_stop_words = ["reuter"]
		stop_words = stop_words + additional_stop_words
		return ' '.join([word for word in s.split() if word not in stop_words])

	# method for stemming the words and converting to lower case
	def stemWords(self,s):
		stemmer = PorterStemmer()		
		return ' '.join([stemmer.stem(word.lower()) for word in s.split()])

preProcessor().getDocuments()