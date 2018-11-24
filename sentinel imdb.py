from os import listdir
import string 
from random import shuffle
from sklearn.model_selection import train_test_split
import codecs
def load_doc(filename):
    file = codecs.open(filename, 'r', 'utf-8')
    text = file.read()
    file.close()
    return text

from nltk.corpus import stopwords
stop = stopwords.words('english')
docs_review =[]

def parse_doc(directory, category):    
    i=1
    for filename in listdir(directory):
        path = directory + '\\' + filename    
        doc = load_doc(path)
        print(path)
        tokens = doc.split(' ')
        temp = [w for w in tokens if w not in stop and w.isalpha() and w not in string.punctuation]
        docs_review.append((temp, category)) 
        if i == 2400:
                return
        i = i+1

parse_doc('C:\\Users\\Kasettakorn\\Desktop\\IMDB_review\\aclImdb\\train\\neg', 'Negative')
parse_doc('C:\\Users\\Kasettakorn\\Desktop\\IMDB_review\\aclImdb\\train\\pos', 'Positive') 

shuffle(docs_review)
all_words = []

for (doc, cat) in docs_review:               
    for w in doc:        
        if(len(w) > 1): 
            all_words.append(w)
        
from nltk import FreqDist 
all_words_frequency = FreqDist(all_words)

vocab = [w for (w, f) in all_words_frequency.most_common(4000)] 
def find_features(document):                
    features = {}
    for w in vocab:
        features[w] = (w in document)
    return features

featuresets = [(find_features(doc), cat) for (doc, cat) in docs_review]

training_set = featuresets[:4000]
testing_set = featuresets[4000:]

from nltk import classify
from nltk.classify import NaiveBayesClassifier 
from nltk.stem import PorterStemmer
classifier = NaiveBayesClassifier.train(training_set)
print("Classifier accuracy percent:",(classify.accuracy(classifier, testing_set))*100, "%")

while (1):
        movie = input('Enter movie or Series: ')
        X = input('Enter review: ')
        if X == '0':
                break
        tokens = X.split(' ')
        J = [PorterStemmer().stem(w) for w in tokens 
        if w not in stop and w not in string.punctuation] 
        feature_set = find_features(J) 
        print ("===========\n",classifier.classify(feature_set), "\n===========\n") 

