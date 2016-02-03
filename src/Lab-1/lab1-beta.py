import time
import copy
import operator
import math
import csv
import numpy as np
from os import listdir
from bs4 import BeautifulSoup
from string import punctuation
from nltk import PorterStemmer
from nltk.corpus import stopwords

# path of the reuters data set
dir_path = "/home/0/srini/WWW/674/public/reuters/"
#dir_path = "H:\\Data Mining\\Dataset\\"

documents = [] # variable to hold all the processed documents
begin_time = None

TF_Dict = []   # holds the frequency of a word in a document-body
TF_Count = [] # holds the TF for each word document-body-wise
term_count = []  # holds the number of words per body of the document 
word_info = {}  # holds the information that in how many different documents(body) a perticular word appera (not document-body-wise)
idf_info = {}  # holds the IDF for each distict word (not document-body-wise)
TF_IDF=[]     #  holds the TF_IDF for each word document-body-wise
header_words=[]
document_matrix_tfidf = []
document_matrix_frequency = []
        
# class for fetching the files and procesing them.
class preProcessor:

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

        print str(len(documents)) + " documents found and processed"
        # print "time taken: " + str((time.time() - (self.begin_time)))

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
            documents.append({"topics":(':'.join([word for word in topics.stripped_strings]))})

        if not title:
            documents[counter]["title"] = ""
        else:
            documents[counter]["title"] = ':'.join([word for word in title.stripped_strings])

        if not places:
            documents[counter]["places"] = ""
        else:
            documents[counter]["places"] = ':'.join([word for word in places.stripped_strings])

        if not body:
            documents[counter]["body"] = ""
        else:
            documents[counter]["body"] = self.process(':'.join([word.lower() for word in body.stripped_strings]))

        return

    # method to process the raw data. Removes punctuation and stop words. Also stems words and capitalizes all letters.
    def process(self,s):
        return str((self.stemWords(self.strip_punctuation(self.strip_stopwords(s)))).encode('utf-8'))

    # method to remove punctuation
    def strip_punctuation(self,s):
        return ''.join(c for c in s if c not in punctuation)

    # method to remove stop words
    def strip_stopwords(self,s):
        stop_words = stopwords.words("english")
        additional_stop_words = ["reuter","said","&#3","and"]
        stop_words = stop_words + additional_stop_words
        return ' '.join([word for word in s.split() if word not in stop_words])

    # method for stemming the words and converting to lower case
    def stemWords(self,s):
        stemmer = PorterStemmer()       
        return ' '.join([stemmer.stem(word) for word in s.split()])
                
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
        
    def idf_process(self):  
        count = 0        
        temp=[]
#        pro_data=[]
        for val in range(0,len(idf_info.values())):
            temp.append(idf_info.values()[val]['idf'])
        MEAN=np.mean(temp)
        SD = np.std(temp)
        Range = MEAN - 2*SD
        sorted_idf = sorted(idf_info.items(), key=operator.itemgetter(1))
        
        while(sorted_idf[count][1]['idf'] <= Range) :
               header_words.append(sorted_idf[count][0])
               count = count + 1
               
#        for item in range(0,len(pro_data)):
#            if not pro_data[item][0].isdigit():
#               header_words.append(pro_data[item][0])
        return
    
    def write_to_file(self, file_name, matrix):
        with open(file_name, 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
    
            headers = []
            headers.append('TOPICS')
            headers.append('PLACES')
            for term in header_words:
                headers.append(term)
    
            writer.writerow([header for header in headers])
            for document_feature_vector in matrix:
                writer.writerow([weight for weight in document_feature_vector])

                   
    def create_doc_matrix_tfidf(self):
        for idx in range(0,len(documents)): 
            feature_vector = [0]*(len(header_words) + 2)
            feature_vector[0] = documents[idx]['topics']
            feature_vector[1] = documents[idx]['places']
            for word in TF_IDF[idx].keys():
                if word in header_words:
                    feature_vector[header_words.index(word)+2] = TF_IDF[idx][word]       
            document_matrix_tfidf.append(feature_vector) 
        self.write_to_file('doc_tfidf.csv',document_matrix_tfidf)
        return 
        
    def create_doc_matrix_frequency(self):
        for idx in range(0,len(documents)): 
            feature_vector = [0]*(len(header_words) + 2)
            feature_vector[0] = documents[idx]['topics']
            feature_vector[1] = documents[idx]['places']
            for word in TF_Dict[idx].keys():
                if word in header_words:
                    feature_vector[header_words.index(word)+2] = TF_Dict[idx][word]       
            document_matrix_frequency.append(feature_vector) 
        self.write_to_file('doc_frequency.csv',document_matrix_frequency)
        return 
        
begin_time = time.time()        
preProcessor().getDocuments()
preProcessor().initialize_dict(documents)
TF_Count = copy.deepcopy(TF_Dict)
preProcessor().tf(documents)   
TF_IDF = copy.deepcopy(TF_Count)
preProcessor().tf_idf() 
preProcessor().idf_process()
print "time taken: " + str((time.time() - (begin_time))) 
preProcessor().create_doc_matrix_tfidf()
preProcessor().create_doc_matrix_frequency()
print "total time taken: " + str((time.time() - (begin_time)))
print TF_Dict[0]
