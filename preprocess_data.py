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



# A lot of the descriptions (and other features) contain HTML.
# The function parses and "translates" into plain text descriptions more suitable for analysis.
def preprocess_data(s):
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