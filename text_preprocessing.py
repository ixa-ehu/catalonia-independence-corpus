# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:42:55 2019

@author: EZOTOVA
"""

import pandas as pd
import re 

filename = 'stopwords-es.txt'
with open(filename, encoding='utf-8') as f:
    stop = f.readlines()
    
stopwords = []
for line in stop:
    l = line.strip()
    stopwords.append(l)

file_lemma = 'lemmatization-lists-master/lemmatization-es.txt'
lemma_table = pd.read_csv(file_lemma, encoding='utf-8', sep='\t')

cols = list(lemma_table.columns)
        
def lemmatization(tweets, lemma_table):
	
	"We create lemma dict"
	
	flex_all = list(lemma_table[cols[1]])
	lemma_all = list(lemma_table[cols[0]])
	dic_lemmas = {}

	for flex,lemma in zip(flex_all, lemma_all):
		dic_lemmas[flex] = lemma
		
	"Create a copy of the table and include a new column"
	
	lemmas = tweets.copy()
	lemmas = lemmas.fillna("")
	lemmas['LEMMA'] = ''
	
	for i, l in enumerate(lemmas['CLEAN_FULL']):
		tweet_lemmas = []
		
		"Get lowercase tweet and split tokens"
		tokens = l.lower().split()
		
		"If the word is in the dictionary include the lemma in the text_lemma"
		for t in tokens:
			if t in dic_lemmas:
				tweet_lemmas.append(dic_lemmas[t])
			else:
				tweet_lemmas.append(t)
		lemmas['LEMMA'][i] = ' '.join(tweet_lemmas)
#		print(i, tweet_lemmas)
#		print(lemmas['text_lemma'][i])
	"Return the new dataframe with new lemma column"
	return lemmas

j = re.compile(r"j{2,}") #detects the character the occurs two and more times
jaja = re.compile(r'(ja){2,}')
jeje = re.compile(r'(je){2,}')
haha = re.compile(r'(ha){2,}')
jiji = re.compile(r'(ji){2,}')
a = re.compile(r'a{2,}')
e = re.compile(r'e{3,}')
i = re.compile(r'i{2,}')
o = re.compile(r'o{2,}')
u = re.compile(r'u{2,}')
f = re.compile(r'f{2,}')
h = re.compile(r'h{2,}')
m = re.compile(r'm{2,}')
rt = re.compile(r'rt')
link = re.compile(r'(https?|http)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]')

punctuation = ['!', '"', '$', '%', '&', "'", '€'
               '(', ')', '*', '+', ',', '-', '.', 
               '/', ':', ';', '<', '=', '>', '?', 
               '[', '\\', ']', '^', '_', '`', 
               '{', '|', '}', '~', '–', '—', '"', 
               "¿", "¡", "''", "...", '_', 
               '“', '”', '…', '‘',  '’', "'", "``", 
               '°', '«', '»', '×', '》》', 'ʖ', '(']

## '‘'  '’' "'" "``"

diacritica = {
    "á": "a",
    "ó": "o",
    "í": "i",
    "é": "e",
    "ú": "u",
    "ü": "u",
    "ù": "u",
    "à": "a",
    "è": "e",
    "ï": "i",
    "ò": "o"
}

def removePunctuation(line): 
    for i in punctuation: 
        line = line.replace(i, '')
    return line   

def getTokens(list_of_strings):
    list_of_strings_tokenized = []
    for line in list_of_strings:
        line = re.sub(link, '', line)
        line = removePunctuation(line)
        line_tokens = line.lower().split()
        list_of_strings_tokenized.append(line_tokens)
    return list_of_strings_tokenized

def removeUsername(tokens):
    for t in tokens:
        if t.startswith('@'):
            tokens.remove(t)
    return tokens

def isNumber(s):
    number_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
    number = True
    for c in s:
        if c not in number_list:
            number = False
    return number
             
def normalizeLine(tokens_list):
    tokens_line_preproc = [] #lines with tokens normalized
    for l in tokens_list: 
        token_line = []
        for t in l: 
            t = re.sub(j, 'j', t)
            t = re.sub(jaja, 'jaja', t)
            t = re.sub(jeje, 'jaja', t)
            t = re.sub(haha, 'jaja', t)
            t = re.sub(jiji, 'jaja', t)
            t = re.sub(a, 'a', t)
            t = re.sub(e, 'e', t)
            t = re.sub(i, 'i', t)
            t = re.sub(o, 'o', t)
            t = re.sub(u, 'u', t)
            t = re.sub(f, 'f', t)
            t = re.sub(h, 'h', t)
            t = re.sub(m, 'm', t)
            t = re.sub(rt, '', t)
                      
            token_line.append(t)
        tokens_line_preproc.append(token_line)
    return tokens_line_preproc

import emoji

def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)


"Load CSV"
filename = 'castellano_dataset.csv'
df = pd.read_csv(filename, dtype={'id_str': 'str', 'user_id_str': 'str'}, encoding='utf-8', sep='\t')
df = df.fillna('')

list_of_tweets = list(df.TWEET.values)

list_of_tokens = getTokens(list_of_tweets)

list_of_tokens_norm = normalizeLine(list_of_tokens)

list_of_tokens_clean = []
for l in list_of_tokens_norm:
    t_line = []
    for t in l:
        if t not in stopwords and not t.isdigit() and not t.startswith(('@', 'http', '#')) and len(t) > 3: #filter stopwords and user names
            t_line.append(t)
    list_of_tokens_clean.append(t_line)

list_of_tokens_clean_full = []
for l in list_of_tokens_norm:
    t_line = []
    for t in l:
        if t not in stopwords and not t.isdigit() and not t.startswith(('@', 'http', '#')): 
            t_line.append(t)
    list_of_tokens_clean_full.append(t_line)
    
list_of_tweets_clean_full = []
for line in list_of_tokens_clean_full:
    l = ' '.join(line)
    l_e = remove_emoji(l)
    l_e = re.sub(r'[0-9]', '', l_e) 
    l_s = l_e.strip()
    list_of_tweets_clean_full.append(l_s)
    
    
list_of_tweets_clean =  []
for line  in list_of_tokens_clean:
    l = ' '.join(line)
    l_e = remove_emoji(l)
    l_e = re.sub(r'[0-9]', '', l_e) 
    l_s = l_e.strip()
    list_of_tweets_clean.append(l_s)

df['CLEAN'] = list_of_tweets_clean
df['CLEAN_FULL'] = list_of_tweets_clean_full 

#lemmatization over cleaned tweets
df_preprocessed = lemmatization(df, lemma_table)

list_of_tweets_lemmatized = list(df_preprocessed.LEMMA.values)

list_of_tweets_lema_clean = []

for line in list_of_tweets_lemmatized:
    line_clean = []
    tokens = line.split()
    for t in tokens:
        t = t.translate({ord(k): v for k, v in diacritica.items()})
        if len(t) > 3:
            line_clean.append(t)
    
    list_of_tweets_lema_clean.append(' '.join(line_clean))
    
df_preprocessed['LEMMA_CLEAN'] = list_of_tweets_lema_clean

import numpy as np 

train, val, test = np.split(df_preprocessed.sample(frac=1), [int(.6*len(df_preprocessed)), int(.8*len(df_preprocessed))])

df_preprocessed.to_csv('preprocessed_for_SVM_'+filename, sep='\t', encoding='utf-8', index=False)


train_sh = train.drop(['CLEAN', 'CLEAN_FULL', 'LEMMA', 'LEMMA_CLEAN'], axis=1)
test_sh = test.drop(['CLEAN', 'CLEAN_FULL', 'LEMMA', 'LEMMA_CLEAN'], axis=1)
val_sh = val.drop(['CLEAN', 'CLEAN_FULL', 'LEMMA', 'LEMMA_CLEAN'], axis=1)

train_sh.to_csv('DATASET/spanish_train.csv', sep='\t', encoding='utf-8', index=False)
test_sh.to_csv('DATASET/spanish_test.csv', sep='\t', encoding='utf-8', index=False)
val_sh.to_csv('DATASET/spanish_val.csv', sep='\t', encoding='utf-8', index=False)
