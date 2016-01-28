from os import listdir
from bs4 import BeautifulSoup
from string import punctuation
from nltk.stem import *
from nltk.corpus import stopwords

# path of the reuters data set
dir_path = "/home/0/srini/WWW/674/public/reuters/"

# variable to hold all the processed documents
documents = []

# class for fetching the files and procesing them.
class preProcessor:

	# method to get documents with raw data from reuters files
	def getDocuments(self):
		
		# class variable for keeping track of the number of documents
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
				self.extractTopics(document,counter);
				self.extractPlaces(document,counter);
				self.extractTitle(document,counter);
				self.extractBody(document,counter);

		print str(len(documents)) + " documents found and processed"
		
		return

	# method to process raw data. Removes punctuation and stop words. Also stems words and capitalizes all letters.
	def process(self,s):
		return ' '.join([each.upper() for each in self.stemWords(self.strip_stopwords(self.strip_punctuation(s))).split()])

	# method to extract information from <topics> tag 
	def extractTopics(self,document,counter):
		
		# check if <topic> tag exists
		if not document.find_all('topics'):
			documents[counter]["topics"] = ""

		# else, get text from tag(s) 
		else:
			
			for topics in document.find_all('topics'):
				topic = "";
				for text in topics.stripped_strings:
					if topic is "":
						topic = text
					else:
						topic = topic + "," + text;

				#append to the global documents list
				documents.append({"topics":topic})
		return

	# method to extract information from <places> tag 
	def extractPlaces(self,document,counter):

		# check if <topic> tag exists	
		if not document.find_all('places'):
			documents[counter]["places"] = ""

		# else, get text from tag(s)
		else:
			for places in document.find_all('places'):
				place = ""
				if not places.get_text():
					place = ""
				else:
					for text in places.stripped_strings:
						if place is "":
							place = text
						else:
							place = place + "," + text;

				#append to the global documents list
				documents[counter]["places"] = place
		return

	# method to extract information from <title> tag 
	def extractTitle(self,document,counter):

		# check if <topic> tag exists
		if not document.find_all('title'):
			documents[counter]["title"] = ""

		# else, get text from tag(s)
		else:
			for titles in document.find_all('title'):
				title = ""
				if not titles.get_text():
					title = ""
				else:
					for text in titles.stripped_strings:
						if title is "":
							title = text
						else:
							title = title + "," + text;

				#append to the global documents list
				documents[counter]["title"] = self.process(title)
		return

	# method to extract information from <body> tag 
	def extractBody(self,document,counter):

		# check if <body> tag exists	
		if not document.find_all('body'):
			documents[counter]["body"] = ""
		
		# else, get text from tag(s) 
		else:
			for bodies in document.find_all('body'):
				body = "";
				if not bodies.get_text():
					body = ""
				else:
					for text in bodies.stripped_strings:
						if body is "":
							body = text
						else:
							body = body + "," + text;

				#append to the global documents list
				documents[counter]["body"] = self.process(body)
		return

	# method to remove punctuation
	def strip_punctuation(self,s):
		return ''.join(c for c in s if c not in punctuation)

	# method to remove stop words
	def strip_stopwords(self,s):
		return ' '.join([word for word in s.split() if word not in stopwords.words('english')])

	# method for stemming the words
	def stemWords(self,s):
		stemmer = PorterStemmer()		
		return ' '.join([stemmer.stem(word) for word in s.split()])

preProcessor().getDocuments()