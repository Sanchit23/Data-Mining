import time
import copy
import operator
import math
import csv
from os import listdir
from bs4 import BeautifulSoup
from string import punctuation
from nltk import PorterStemmer
from nltk.corpus import stopwords

# path of the reuters data set
#dir_path = "/home/0/srini/WWW/674/public/reuters/"
dir_path = "http://web.cse.ohio-state.edu/~srini/674/public/reuters/"

documents = [] # variable to hold all the processed documents

TF_Dict = []   # holds the frequency of a word in a document-body
TF_Count = [] # holds the TF for each word document-body-wise
term_count = []  # holds the number of words per body of the document 
word_info = {}  # holds the information that in how many different documents(body) a perticular word appera (not document-body-wise)
idf_info = {}  # holds the IDF for each distict word (not document-body-wise)
TF_IDF=[]     #  holds the TF_IDF for each word document-body-wise
MF_TFIDF=[]

# class for fetching the files and procesing them.
class preProcessor:

    # constructor
    def __init__(self): pass

    # method to get documents from reuters files and process them
    def getDocuments(self):

        begin_time = time.time()
        
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

        print str(len(documents)) + " documents found and processed"
        # print "time taken: " + str((time.time() - (begin_time)))

        return

    # method to extract information from tags
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
        additional_stop_words = ["reuter","the","said"]
        stop_words = stop_words + additional_stop_words
        return ' '.join([str(word.encode('utf-8')) for word in s.split() if word not in stop_words])

    # method for stemming the words and converting to lower case
    def stemWords(self,s):
        stemmer = PorterStemmer()       
        return ' '.join([stemmer.stem(word.lower()) for word in s.split()])
                
    def initialize_dict(self, documents):
        for Doc in documents:
            for word in Doc['body'].split(): 
                 word_info[word] = {'doc_count': 0}
                 idf_info[word] = {'idf': 0}
                 
        self.word_freq_per_doc(documents) 
        self.total_word_count()
        return    
        
    def word_freq_per_doc(self, documents):
        for DOC in documents:
            word_freq = {}
            word_list = DOC['body'].split()
            term_count.append(len(word_list))
            for word in set(word_list):
                word_freq[word] = word_list.count(word)
            TF_Dict.append(word_freq) 
        return
        
    def total_word_count(self):          
          for doc in TF_Dict:
               for word in doc:
                   word_info[word]['doc_count'] = word_info[word]['doc_count'] + 1
          return
    
    def tf(self,documents):
        cnt = 0
        for item in TF_Count:
            for k in item.keys():
                item[k] = round((item[k] /float(term_count[cnt])),3)
            cnt = cnt +1 
        self.idf(documents)             
        return
        
    def idf(self,documents):
        for WORD in word_info:
            idf_info[WORD]['idf'] =  round((math.log(len(documents) / float(word_info[WORD]['doc_count']))),3)
        return
    
    def tf_idf(self):
        for ITEM in TF_IDF:
            for K in ITEM.keys():
                ITEM[K] = round((ITEM[K] * idf_info[K]['idf']),3)
        return 
        
#    def tfidf_process(self):   
#        sorted_idf = sorted(idf_info.items(), key=operator.itemgetter(1))
#        sorted_idf.reverse()
#        tmp = sorted_idf[0:100]
#        return tmp
#        
#    def write_to_file_dtm(document_term_matrix, term_lexicon, file_name):
#        with open(file_name, 'wb') as csvfile:
#            writer = csv.writer(csvfile, delimiter=',',
#                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
#    
#            headers = []
#            headers.append('TOPICS')
#            headers.append('PLACES')
#            for term in term_lexicon:
#                headers.append(term)
#    
#            writer.writerow([header for header in headers])
#            for document_feature_vector in document_term_matrix:
#                writer.writerow([weight for weight in document_feature_vector])
        
preProcessor().getDocuments()
preProcessor().initialize_dict(documents)
TF_Count = copy.deepcopy(TF_Dict)
preProcessor().tf(documents)   
TF_IDF = copy.deepcopy(TF_Count)
preProcessor().tf_idf() 
MF_TFIDF=preProcessor().tfidf_process()
