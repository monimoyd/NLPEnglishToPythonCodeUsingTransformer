import nltk
import string
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import pandas as pd
import string
import random
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
import string
import google_trans_new
from google_trans_new import google_translator

def back_translate(sentence):
  
    available_langs = list(google_trans_new.LANGUAGES.keys()) 
    trans_lang = random.choice(available_langs) 
    #print(f"Translating to {google_trans_new.LANGUAGES[trans_lang]}")
    translator = google_translator()
    translations = translator.translate(text=sentence, lang_src='en', lang_tgt=trans_lang) 
    #print(translations)

    translations_en_random = translator.translate(text=translations, lang_src=trans_lang, lang_tgt='en') 
    # print(translations_en_random)
    return translations_en_random

def random_deletion(sentence, p=0.3): 
    words = sentence.split()
    ret_val = ""
    if len(words) == 1: # return if single word
        ret_val = words
        return ret_val
    remaining = list(filter(lambda x: random.uniform(0,1) > p,words)) 
    if len(remaining) == 0: # if not left, sample a random word
        ret_val = [random.choice(words)] 
    else:
        ret_val = remaining
    return " ".join(ret_val)

def random_swap(sentence, n=5):
    words =  sentence.split()
    length_words = len(words)
    if length_words < 2:
        return  sentence
    length = range(length_words)

    for _ in range(n):
        idx1, idx2 = random.sample(length, 2)
        words[idx1], words[idx2] = words[idx2], words[idx1] 
    return " ".join(words)
	
def synonym_word(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lm in syn.lemmas():
            synonyms.append(lm.name())
    if len(synonyms) <=0:
        return word
    return random.sample(synonyms, 1)[0]

def synonym_sentence(sentence, prob=0.5):
    words =  sentence.split()
    synonym_words = []
    for word in words:
        if random.random() < prob:
            synonym_words.append(synonym_word(word))
        else:
            synonym_words.append(word)
    return " ".join(synonym_words)
	
