import nltk
import string
import re

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import string
import nltk
import random
import random
#import google_trans_new
#from google_trans_new import google_translator

# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
#from nltk.corpus import stopwords
lem = WordNetLemmatizer()
#translator = google_translator()

def clean_text(text):
    ## lower case
    if not isinstance(text, str):
      return str(text) 
    cleaned = text.lower()

    urls_pattern = re.compile(r'https?://\S+|www.\S+')
    cleaned = urls_pattern.sub(r'',cleaned)
    
    ## remove punctuations
    punctuations = string.punctuation
    cleaned_temp = "".join(character for character in cleaned if character not in punctuations)
    
    ## remove stopwords 
    words = cleaned_temp.split()
    #stopword_lists = stopwords.words("english")
    #cleaned = [word for word in words if word not in stopword_lists]
    cleaned = words
    
    ## normalization - lemmatization
    #cleaned = [lem.lemmatize(word, "v") for word in cleaned]
    #cleaned = [lem.lemmatize(word, "n") for word in cleaned]
    
    ## join 
    cleaned = " ".join(cleaned)
    return cleaned