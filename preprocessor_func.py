#Bibliotecas
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from string import punctuation

def token_TextList(text_list):
  stoplist = set(nltk.corpus.stopwords.words('english') + list(punctuation))
  r = []
  for i in range(0,len(text_list)):
    tokens = [token.lower() for token in nltk.word_tokenize(text_list[i]) if (token.lower() not in stoplist) and token.isalpha()]
    r = r+tokens
  return r

def lemma_tokens(token_list):
  lemmatizer = nltk.stem.WordNetLemmatizer()
  l = [lemmatizer.lemmatize(token, pos="v") for token in token_list]
  return l

def stem_tokens(token_list):
  stemmer = nltk.stem.PorterStemmer()
  s = [stemmer.stem(token) for token in token_list]
  return s

def my_preprocessorToken(text):
  text=text.lower() 
  
  # stem words
  stemmer = nltk.stem.PorterStemmer()

  stoplist = set(nltk.corpus.stopwords.words('english') + list(punctuation))
  
  words =[token.lower() for token in nltk.word_tokenize(text) if (token.lower() not in stoplist) and token.isalpha()]

  return ' '.join(words)

def my_preprocessorStem(text):

  text=text.lower() 
  
  # stem words
  stemmer = nltk.stem.PorterStemmer()

  stoplist = set(nltk.corpus.stopwords.words('english') + list(punctuation))
  
  words =[token.lower() for token in nltk.word_tokenize(text) if (token.lower() not in stoplist) and token.isalpha()]
  stemmed_words=[stemmer.stem(word=word) for word in words]

  return ' '.join(stemmed_words)

def my_preprocessorLemma(text):

  text=text.lower() 
  
  # stem words
  lemmatizer = nltk.stem.WordNetLemmatizer()

  stoplist = set(nltk.corpus.stopwords.words('english') + list(punctuation))
  
  words =[token.lower() for token in nltk.word_tokenize(text) if (token.lower() not in stoplist) and token.isalpha()]
  lemma_words=[lemmatizer.lemmatize(word=word) for word in words]

  return ' '.join(lemma_words)

# function to convert nltk tag to wordnet tag
def nltk_tag_to_wordnet_tag(nltk_tag):
  if nltk_tag.startswith('J'):
      return nltk.corpus.wordnet.ADJ
  elif nltk_tag.startswith('V'):
      return nltk.corpus.wordnet.VERB
  elif nltk_tag.startswith('N'):
      return nltk.corpus.wordnet.NOUN
  elif nltk_tag.startswith('R'):
      return nltk.corpus.wordnet.ADV
  else:          
      return None

def my_preprocessorLemmaPOS(sentence):
  #tokenize the sentence and find the POS tag for each token
  lemmatizer = nltk.stem.WordNetLemmatizer()
  stemmer = nltk.stem.PorterStemmer()
  stoplist = set(nltk.corpus.stopwords.words('english') + list(punctuation))
  words =[token.lower() for token in nltk.word_tokenize(sentence) if (token.lower() not in stoplist) and token.isalpha()]
  nltk_tagged = nltk.pos_tag(words)  
  #tuple of (token, wordnet_tag)
  wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
  lemmatized_sentence = []
  for word, tag in wordnet_tagged:
      if tag is None:
          #if there is no available tag, append the token as is
          lemmatized_sentence.append(word)
      else:        
          #else use the tag to lemmatize the token
          lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
  return " ".join(lemmatized_sentence)


def my_preprocessor2(sentence):
  #tokenize the sentence and find the POS tag for each token
  lemmatizer = nltk.stem.WordNetLemmatizer()
  stemmer = nltk.stem.PorterStemmer()

  stoplist = set(nltk.corpus.stopwords.words('english') + list(punctuation))
  words =[token.lower() for token in nltk.word_tokenize(sentence) if (token.lower() not in stoplist) and token.isalpha()]
  nltk_tagged = nltk.pos_tag(words)  
  #tuple of (token, wordnet_tag)
  wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
  stem_sentence = []
  for word, tag in wordnet_tagged:
      if tag is None:
          #if there is no available tag, append the token as is
          stem_sentence.append(stemmer.stem(word))
      else:        
          #else use the tag to lemmatize the token
          stem_sentence.append(stemmer.stem(lemmatizer.lemmatize(word, tag)))
  return " ".join(stem_sentence)

def hist_docfreq(train_list, vectorizer):
  freq = vectorizer.transform(train_list)
  doc_freq = np.sum(1*(freq.todense() != 0),0) #cuenta por col
  arr = doc_freq.getA1().tolist()
  h = plt.hist(arr,bins=np.arange(0,50))
  plt.title("Histograma: cantidad de palabras sobre doc_freq")
  plt.xlabel("Cantidad de Textos")
  plt.ylabel("Número de palabras")

