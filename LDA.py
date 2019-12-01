# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 21:57:27 2019

@author: EZOTOVA
"""
import pandas as pd
from gensim.corpora import Dictionary
from gensim import corpora
import gensim
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
col_names = list(lemma_table.columns.values)
   
#flex_all = list(lemma_table[col_names[1]])
#     
def lemmatization(tweets, lemma_table):
	
	"We create lemma dict"
	
	flex_all = list(lemma_table[col_names[1]])
	lemma_all = list(lemma_table[col_names[0]])
	dic_lemmas = {}

	for flex,lemma in zip(flex_all, lemma_all):
		dic_lemmas[flex] = lemma
		
	"Create a copy of the table and include a new column"
	
	lemmas = tweets.copy()
	lemmas = lemmas.fillna("")
	lemmas['text_lemma'] = ''
	
	for i, l in enumerate(lemmas['CLEAN']):
		tweet_lemmas = []
		
		"Get lowercase tweet and split tokens"
		tokens = l.lower().split()
		
		"If the word is in the dictionary include the lemma in the text_lemma"
		for t in tokens:
			if t in dic_lemmas:
				tweet_lemmas.append(dic_lemmas[t])
			else:
				tweet_lemmas.append(t)
		lemmas['text_lemma'][i] = ' '.join(tweet_lemmas)
#		print(i, tweet_lemmas)
#		print(lemmas['text_lemma'][i])
	"Return the new dataframe with new lemma column"
	return lemmas

j = re.compile(r"j{2,}") #detects the character the occurs two and more times
jaja = re.compile(r'(ja){2,}')
jeje = re.compile(r'(je){2,}')
haha = re.compile(r'(ha){2,}')
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

punctuation = ['!', '"', '$', '%', '&', "'", 
               '(', ')', '*', '+', ',', '-', '.', 
               '/', ':', ';', '<', '=', '>', '?', 
               '[', '\\', ']', '^', '_', '`', 
               '{', '|', '}', '~', '–', '—', '"', 
               "¿", "¡", "``", "''", "...", '_', 
               '“', '”', '…', '‘',  '’']

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
             
def normalizeLine(tokens_list):
    tokens_line_preproc = [] #lines with tokens normalized
    for l in tokens_list: 
        token_line = []
        for t in l: 
            t = re.sub(j, 'j', t)
            t = re.sub(jaja, 'jaja', t)
            t = re.sub(jeje, 'jaja', t)
            t = re.sub(haha, 'jaja', t)
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


def compute_coherence_values(dictionary, corpus, texts, limit, start=10, step=2):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    count = 0
    for num_topics in range(start, limit, step):
        
        model=gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='u_mass')
        print(coherencemodel.get_coherence())
        coherence_values.append(coherencemodel.get_coherence())
        count += 1
        print(count)

    return model_list, coherence_values

def prepare_text_for_lda(text):
    tokens = text.split()
    tokens = [token for token in tokens if len(token) > 3] #we use only "long" words, they are usually more significant
    tokens = [token for token in tokens if token not in stopwords] #filter stopwords and punctuation
    return tokens

"Load CSV"
table = pd.read_csv('castellano_neutro.csv', dtype={'id_str': 'str', 'user_id_str': 'str'}, encoding='utf-8', sep='\t')
table = table.fillna('')
print('Tweets Neutro ', len(table))
list_of_tweets = list(table.TWEET.values)

list_of_tokens = getTokens(list_of_tweets)

list_of_tokens_norm = normalizeLine(list_of_tokens)

list_of_tokens_clean = []
for l in list_of_tokens_norm:
    t_line = []
    for t in l:
        if t not in stopwords and not t.startswith(('@', 'http')): #filter stopwords and user names
            t_line.append(t)
    list_of_tokens_clean.append(t_line)

list_of_tweets_clean =  []
for line  in list_of_tokens_clean:
    l = ' '.join(line)
    l_s = l.strip()
    list_of_tweets_clean.append(l_s)

table['CLEAN'] = list_of_tweets_clean

lemmas = lemmatization(table, lemma_table)

text = list(lemmas.text_lemma.values)

text_data = []
for line in text: 
    tokens = prepare_text_for_lda(line)
    text_data.append(tokens)

dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]

NUM_TOPICS = 100

from gensim.models import CoherenceModel

import os

os.environ.update({'MALLET_HOME':r'C:\\Users\\ezotova\\Desktop\\Python\\Dataset_Español\\new_mallet\\mallet-2.0.8\\'})

mallet_path = 'C:\\Users\\ezotova\\Desktop\\Python\\Dataset_Español\\new_mallet\\mallet-2.0.8\\bin\\mallet' # update this path
ldamallet_model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print(NUM_TOPICS)
print('\nCoherence Score Mallet: ', coherence_ldamallet)

coherencemodel_c_v = CoherenceModel(model=ldamallet_model, texts=text_data, dictionary=dictionary, coherence='c_v')
print(coherencemodel_c_v)
#print('\nPerplexity LDA: ', ldamallet_model.log_perplexity(corpus)) 

model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=text_data, start=10, limit=NUM_TOPICS, step=2)
# Show graph
import matplotlib.pyplot as plt
limit=NUM_TOPICS; start=10; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


#apply LDA model with the best number of topics
ldamallet_corpus = ldamallet_model[corpus]

BEST_NUM_TOPICS = 64
ldamallet_topics = []
for top in ldamallet_model.print_topics(num_topics=BEST_NUM_TOPICS, num_words=30):
    ldamallet_topics.append(top)
#    print(top)

ldamallet_corpus = [max(prob,key=lambda y:y[1]) for prob in ldamallet_model[corpus] ]
playlists_ldamallet = [[] for i in range(BEST_NUM_TOPICS)]
for i, x in enumerate(ldamallet_corpus):
    playlists_ldamallet[x[0]].append(text_data[i])



lemmas['LDA'] = ldamallet_corpus
new_col_list = ['TOPIC','CONF']
for n,col in enumerate(new_col_list):
    lemmas[col] = lemmas['LDA'].apply(lambda location: location[n])

lemmas = lemmas.drop('LDA',axis=1)

topic_counts = lemmas['TOPIC'].value_counts()
print(topic_counts)

df_hashtags_count = lemmas['HASHTAG'].value_counts()
#print(df_hashtags_count)

df_true = lemmas.loc[lemmas['HASHTAG'] == True] #tweets that contain the hashtags are always used for the dataset
df_false = lemmas.loc[lemmas['HASHTAG'] == False] #tweets that do not contain the hashtags


for i in ldamallet_topics: #print topic words to revise them manually
    print(i)


indepe_topics = [0, 8, 44] #selected topics
df_indepe_topic = df_false.loc[df_false['TOPIC'].isin(indepe_topics)] #select tweets from not-hashtag

clean = list(df_indepe_topic.CLEAN.values)       
clean_tokenized = []
for line in clean:
    l = line.split()
    clean_tokenized.append(l)
                       
df_indepe_topic['CLEAN_T'] = clean_tokenized

#filter tweets shorter than 3 tokens after applying stopwords and removing usernames
df_indepe_topic_filtrado = df_indepe_topic[df_indepe_topic['CLEAN_T'].apply(len)>2]
df_indepe_topic_filtrado = df_indepe_topic_filtrado.drop('CLEAN_T',axis=1)

frames = [df_true, df_indepe_topic_filtrado]
df_cat_prepared = pd.concat(frames)
print(df_cat_prepared.dtypes)
print(len(df_cat_prepared))

df_cat_prepared.to_csv('castellano_neutro_LDA.csv', sep='\t', encoding='utf-8', index=False)
