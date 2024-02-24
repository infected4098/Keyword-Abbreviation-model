#!/usr/bin/env python
# coding: utf-8

# In[ ]:



from konlpy.tag import Mecab
from itertools import permutations, combinations, product
import numpy as np


mecab = Mecab()


def words_extract(characters): # input: 글자 리스트

  windows = [[], characters]  # windows 에는 모든 경우 포함
  for i in range(2, 5):
    permutation = list(set(permutations(characters, i)))
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


def words_combination(is_noun, characters): # input: 조합 단어 리스트, 글자 리스트

  is_noun_concat = []
  for _ in is_noun:
    is_noun_concat.extend(_)

  combination = []
  for i in range(1, len(characters) + 1):
    combination.extend(list(combinations(is_noun_concat, i)))

  final_combination = []
  for c in combination:
    concat = ''
    for w in c:
      concat += w
    if len(concat) == len(characters):
      flag = 1
      for f in characters:
        if concat.count(f) != characters.count(f):
          flag = 0
          break
      if flag:
        final_combination.append(c)
  
  final_combination = list(set(final_combination))

  return final_combination  # output: 최종 조합 리스트



input_words = ['키', '킥 의', '킼 의', '킫 의']

characters_permutation = characters_extract(input_words)

output = []

while not output:

  if not characters_permutation:
    print('All impossible.')
    break
  
  characters = characters_permutation.pop(0)

  words_by_windows = words_extract(characters)

  output = words_combination(words_by_windows, characters)

  if not output:
    print(f'{characters} is impossible.')
  

