#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from konlpy.tag import Mecab
from pykospacing import Spacing
from itertools import permutations, combinations, product
import pickle
import numpy as np

mecab = Mecab()
spacing = Spacing()

def characters_extract(words):

   splited_words = []
   for word in words:
     splited_words.append(word.split())

   characters = []
   for word in splited_words:
     if len(word) == 1:
       characters.append(word[0][0])
     else:
       characters.append([w[0] for w in word])

   characters_permutation = list(product(*characters))

   return characters_permutation

  first = []
  for word in words:
    first.append(word[0])
  
  return [first]


def words_extract(characters): # input: 글자 리스트

  windows = [[], characters]  # windows 에는 모든 경우 포함
  for i in range(2, 5):
    permutation = list(permutations(characters, i))
    for j, p in enumerate(permutation):
      concat = ''
      for character in p:
        concat += character
        permutation[j] = concat
    windows.append(permutation)

  is_noun = [[] for _ in range(5)]  # is_noun 에는 단어만 포함
  for i in range(1, 5):
    for w in windows[i]:
      if len(mecab.pos(w)) == 1:
        # is_noun[i].append(w)
        if mecab.pos(w)[0][1][0] in 'NVM':  # 명사, 동사, 형용사, 관형사, 부사
          is_noun[i].append(w)  # index 가 window_size 를 의미함

  return is_noun  # output: window_size 별 조합 단어 리스트


def words_combination(is_noun, characters): # input: 조합 단어 리스트, 글자 리스트

  is_noun_concat = []
  for _ in is_noun:
    is_noun_concat.extend(_)

  combination = []
  for i in range(1, len(characters)):  # 모든 단어의 길이가 1인 경우는 제외
    combination.extend(list(combinations(is_noun_concat, i)))

  final_combination = []
  for c in combination:
    concat = ''
    one_cnt = 0
    for w in c:
      if len(w) == 1:
        one_cnt += 1
      concat += w
    if len(concat) == len(characters) and one_cnt < 3:
      flag = 1
      for f in characters:
        if concat.count(f) != characters.count(f):
          flag = 0
          break
      if flag:
        final_combination.append(c)
  
  final_combination = list(set(final_combination))

  return final_combination  # output: 최종 조합 리스트
     
    
    
def words2abb(words):

  output = []

  characters_permutation = characters_extract(input_words)

  while not output:
    if not characters_permutation:
      break
    
    characters = characters_permutation.pop(0)
    words_by_windows = words_extract(characters)
    output = words_combination(words_by_windows, characters)
    
  return output



def abb2sent(combs, model):  # input: window_size 별 조합 단어 리스트
  
  made = ['']
  prob = [0]
  idx = [[]]

  for t in combs:
    per = list(permutations(t))
    for k in range(len(per)):
      input = per[k]
      n = len(input)
      sentence = ""
      pp = ['은', '는', '이', '가', '을', '를', '와', '과', '다', '고', '의', '에', '로', '며', '만', '도']  # 조사/어미 리스트
      pp_cnt = 0
      flag = 1
      total_freq = model.unigrams.freq(input[0])
      position = []
      for i in range(n-1):

        if model[[input[i]]]:  # corpus 에 단어가 존재하는 경우

          if model[[input[i]]][input[i+1]]:  # 두 단어 bigram 으로 바로 연결되는 경우
            sentence += input[i] + " "
            total_freq *= model[[input[i]]].freq(input[i+1])

          else:  # 두 단어가 bigram 으로 연결되지 않고 조사/어미를 활용해 trigram 시도
            pre_freq = []
            post_freq = []
            for p in pp:
              pre_freq.append(model[[input[i]]].freq(p))
              post_freq.append(model[[p]].freq(input[i+1]))
            con_freq = np.array(pre_freq) * np.array(post_freq)
            if max(con_freq):  # trigram 으로 가능한 경우
              sentence += input[i] + pp[con_freq.argmax()] + " "
              total_freq *= max(con_freq)
              pp_cnt += 1
              position.append(len(sentence) - 2)
            else:  # trigram 으로도 불가능한 경우
              flag = 0
              break
                
                else:  # corpus 에 단어가 존재하지 않는 경우
          flag = 0
          break
          
      if flag:  # 해당 조합으로는 문장을 만들 수 없는 경우
        sentence += input[-1]
        made.append(sentence)
        prob.append((total_freq)**(1/(n + pp_cnt)))
        idx.append(position)

      else:  # 문장이 만들어진 경우
        made.append("")
        prob.append(0)
        idx.append([])
  
  return made, prob, idx  # output: 만들어진 문장과 각각 문장의 확률



def words2sent(words, model):  # input: 줄임말을 생성하고 싶은 단어 리스트

  abb = words2abb(words)
  sents, probs, idxs = abb2sent(abb, model)

  best10 = []
  for _ in range(10):
    if max(probs):
      index = np.array(probs).argmax()
      best10.append((sents.pop(index), round(np.sqrt(1/probs.pop(index)), 3), idxs.pop(index)))
    else:
      break

  return best10  # output: 최대 10개의 가장 자연스러운 문장
     
    
    
output = words2sent(input_words, trained_model)

for rank in range(len(output)):
  color = '\033[34m'
  cend = '\033[0m'
  sent, pp, index = output[rank]
  idx = index[:]
  idx.insert(0,-1)
  print(f'{rank+1}.', end=' ')
  for i in range(len(idx) - 1):
    print(f'{color}{sent[idx[i]+1:idx[i+1]]}{cend}\033[30m{sent[idx[i+1]]}{cend}', end='')
  print(f'{color}{sent[idx[-1]+1:]}{cend}, {pp}')

