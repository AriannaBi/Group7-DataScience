import json
from collections import defaultdict
import gzip
import pandas as pd
from lxml import html,etree
import numpy as np
import ipywidgets as widgets
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import os

# set stopwords vocabulary
nltk.download('stopwords')

# set tokenizer
nltk.download('punkt')

# preprocessing for user description and sentiment analysis 
def user_description_sentiment_analysis(s):
    stop_words = set(stopwords.words('english'))
    stemmer= PorterStemmer()
    if not s or s.isspace(): 
        return ''
    try:
        # remove html tags 
        strr = str(html.fromstring(s).text_content())
        # remove URLs
        strr = re.sub(r"(https|http|href)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", ' ', strr)
        # remove html hidden carachters 
        strr = strr.replace('\n', ' ').replace('\t', ' ').replace("&nbsp", ' ').replace('\r', ' ')
        # remove punctuation
        strr = re.sub(r'[^\w\s]|[_+]', ' ', strr)
        # lowercase
        strr = strr.lower()
        # remove numbers
        strr = re.sub(r'\d+', '', strr)
        # remove stop words
        tokens = nltk.word_tokenize(strr)
        strr = [i for i in tokens if not i in stop_words]
        # print(len(strr))
        # if (len(strr) == 0):
        #     print("-------------------------")
        strr = ' '.join(strr)
        return strr
        # return str(html.fromstring(s).text_content(s))
    except etree.ParserError: # I am not able to find out why the error occur so i continued by catching the exception. Seem to happen on some empty description strings 
        return ''


# preprocessing for similar items
# A lot of the descriptions (and other features) contain HTML.
# The function parses and "translates" into plain text descriptions more suitable for analysis.
def process_and_stemming(s):
    stop_words = set(stopwords.words('english'))
    stemmer= PorterStemmer()
    if not s or s.isspace(): 
        # print("Empty description", s, "empty")
        return ''
    try:
        # remove html tags 
        strr = str(html.fromstring(s).text_content())
        # remove URLs
        strr = re.sub(r"(https|http|href)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", ' ', strr)
        # remove html hidden carachters 
        strr = strr.replace('\n', ' ').replace('\t', ' ').replace("&nbsp", ' ').replace('\r', ' ')
        # remove punctuation
        strr = re.sub(r'[^\w\s]|[_+]', ' ', strr)
        # remove numbers
        strr = re.sub(r'\d+', '', strr)
        # lowercase
        strr = strr.lower()
        # remove stop words
        tokens = nltk.word_tokenize(strr)
        strr = [i for i in tokens if not i in stop_words]
        # stemming
        strr = [stemmer.stem(word) for word in strr]
        strr = ' '.join(strr)
        return strr 
    except etree.ParserError: 
        return ''
    
# # preprocessing for similar items
# # A lot of the descriptions (and other features) contain HTML.
# # The function parses and "translates" into plain text descriptions more suitable for analysis.
# def stemming_data(s):
#     # stop_words = set(stopwords.words('english'))
#     stemmer= PorterStemmer()
#     if not s or s.isspace(): 
#         # print("Empty description", s, "empty")
#         return ''
#     try:
#         # stemming
#         strr = s
#         strr = [stemmer.stem(word) for word in strr]
#         strr = ' '.join(strr)
#         return strr 
#     except etree.ParserError: 
#         return ''
    
    

# preprocessing only for removing html, urls and hidden characters
def html_url_hidden_chars(s):
    stop_words = set(stopwords.words('english'))
    stemmer= PorterStemmer()
    if not s or s.isspace(): 
        return ''
    try:
        # remove html tags 
        strr = str(html.fromstring(s).text_content())
        # remove URLs
        strr = re.sub(r"(https|http|href)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b", ' ', strr)
        # remove html hidden carachters 
        strr = strr.replace('\n', ' ').replace('\t', ' ').replace("&nbsp", ' ').replace('\r', ' ')
        return strr
        # return str(html.fromstring(s).text_content(s))
    except etree.ParserError: # I am not able to find out why the error occur so i continued by catching the exception. Seem to happen on some empty description strings 
        return ''