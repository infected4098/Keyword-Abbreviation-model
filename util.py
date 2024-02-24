#!/usr/bin/env python
# coding: utf-8

# In[ ]:



get_ipython().system('set -x && pip install konlpy && curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash -x')
    
from nltk.lm import NgramCounter
from google.colab import drive
drive.mount('/content/gdrive')
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
import json
import konlpy
from nltk.tokenize import word_tokenize
import re
from konlpy.tag import Okt
from konlpy.tag import Mecab, Komoran
import time
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from itertools import combinations
from itertools import product
with open("/content/gdrive/MyDrive/origin/BWSC217000049025.json", "r") as f:

    json_data=json.load(f)
#print(json.dumps(json_data))
kor_text=json_data["SJML"]["text"][1]["content"]


# In[ ]:


pattern_1=r"
"
pattern_2=r"\.{2}"
def preprocess_sent(lst_text):
    wholelist=[]
    buf=str()
    subkor_1=re.sub(pattern=pattern_1, repl="", string=lst_text)
    subkor_2=re.sub(pattern=pattern_2, repl="", string=subkor_1)
    clear_list=re.compile("[가-힣]+|\.{1}").findall(subkor_2)
    for word in clear_list:
        if word==".":
            buf+=(word+" ")
            wholelist.append(buf)
            buf=str()
        
        else:
            buf+=(word+" ")
    return wholelist

komoran=Komoran()
mecab=Mecab()
lst_sent=preprocess_sent(kor_text)


# In[1]:


def sent_to_morph(lst_sent):
    double_list=[]
    for sentence in lst_sent:
        #listed=komoran.morphs(sentence)
        listed=mecab.morphs(sentence)
        double_list.append(listed)
    return double_list

def bigram_probs(model, pre_word, post_word):
    word_dict=dict(model.counts[[pre_word]])
    wholecount=model.counts[pre_word]
    if wholecount!=0:

      try:
          post_count=word_dict[post_word]
      except:
          post_count=0
     

      emer_prob=post_count/wholecount

    else:
      emer_prob=0
    return emer_prob

def trigram_probs(model, pre_word, post_word):
    tulist=[]
    word_dict=dict(model.counts[[pre_word]])
    
    for key in list(word_dict.keys()):
        if key !=pre_word:
            first_prob=bigram_probs(model, pre_word, key)
            second_prob=bigram_probs(model, key, post_word)
            whole_prob=first_prob*second_prob
            if whole_prob!=0:
                #normalization
                whole_prob=whole_prob*2
                tulist.append([pre_word, key, post_word, whole_prob])
        else:
            continue
    return tulist
     
    
    
    
#bigram, trigram이 모두 되는지 확인해보기. 
def canconnect(model, pre_word, post_word):
    possiblelist=[]
    biprob=bigram_probs(model, pre_word, post_word)
    if biprob!=0:
        possiblelist.append([pre_word, post_word, biprob])
    
    tulist=trigram_probs(model, pre_word, post_word)
    for li in tulist:
        possiblelist.append(li)
    return possiblelist
  
#  가능한 케이스들과 이에 대한 확률 제시. 인덱스도 만들어야 해(가능한 경우의 수 개수)
def bigconnect(model, list_of_words):
    dummylist=[]
    index_list=[]
    for i in range(len(list_of_words)-1):
        connection=canconnect(model, list_of_words[i], list_of_words[i+1])
        if len(connection)!=0:
            n=len(connection)
            index_list.append(n)
            for possible in connection:
                dummylist.append(possible)
    
        else:
            print("connection failed")
            break
    return dummylist, index_list

def process(model, list_of_words):
    doublelists, index=bigconnect(model, list_of_words)
    for indexes in range(len(doublelists)):
        doublelists[indexes]=tuple(doublelists[indexes])
    return doublelists, index

def separate_connection(model,list_of_words):
    newconlist=[]
    pro_doublelists, pro_index=process(model, list_of_words)
    end, start=0, 0
    for i in pro_index:
        end+=i
        newconlist.append(pro_doublelists[start:end])

        start+=i
    return newconlist
  
def make_it_one(model, list_of_words):
    wholeconcat=[]
    wholeprob=1.0
    newlist=separate_connection(model, list_of_words)
    maybe_one=list(product(*newlist))
    for i in maybe_one:
        for index in range(len(i)):
            wholeconcat.append(i[index][:-2])
            wholeprob*=i[index][-1]
        wholeconcat.append(list_of_words[-1])
        
        wholeconcat.append(wholeprob)
        wholeprob=1.0
    return wholeconcat



def ngrams(double_list, model):
    
        
    train, vocab=padded_everygram_pipeline(2, double_list)
    vocab=list(vocab)
    
    model.fit(train, vocab)
    
def pre_pipeline(stringed_text):
    listed=preprocess_sent(stringed_text)
    double=sent_to_morph(listed)
    
    return double



#파일 받아서 전체과정
def file_to_string(filenick):
    dummylist=[]
    
    with open(filenick) as f:
        json_data=json.load(f)
    for i in range(0, 1000):
        try:
        
            kor_text=json_data["SJML"]["text"][i]["content"]
            dummylist.append(kor_text)
        except:
            break
    return dummylist

def whole_pipeline(filenick, given_model):
    dummies=file_to_string(filenick)
    
    for dummy in dummies:
        
        double_list=pre_pipeline(dummy)
        ngrams(double_list, given_model)

