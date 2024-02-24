# Keyword-Abbreviation-model
2022년 7월 수행한 키워드 줄임말 만들기 프로젝트 정리본입니다. 
# 지루한 암기는 이제 그만! 키워드 줄임말 만들기 모델

# 0. 팀 & 프로젝트 소개

<aside>
☁️ 안녕하세요, NLP을 공부하는 위키팀의 구름빵입니다. 구름빵은 @용용 ‍ @Junpyter @허인 으로 구성되어 있습니다. 저희는 제1회 deep daiv. On-board 컨퍼런스 주제로 ‘키워드 줄임말 생성 모델’을 선택하였는데요! 본격적으로, 프로젝트에 대한 설명을 시작합니다 😎


</aside>





# 1. 데이터 전처리

쉬운 설명을 위해, 위에서 등장한 단어를 사용하겠습니다. 사용자가 입력하는 키워드가 words = [김천, 대구, 부산, 서귀포, 원주, 진천]이라고 가정합시다. 이때 사용자는 앞서 프로젝트 소개 부분에서 설명한 학생에 해당하고, 입력하는 키워드는 암기하고자 하는 개념들을 말합니다. 사용자가 직접 원하는 개념을 프롬프트에 입력하면 결과를 출력하도록 프로그램을 설계하였기에 ‘입력’이라고 이야기하겠습니다.

## 1-1 첫 음절 추출

여기서 저희가 설정한 규칙, 그리고 이에 따른 방법이 등장합니다. 아래 함수는 규칙을 직접 구현한 코드입니다.

```python
def characters_extract(words):

  first = []
  for word in words:
    first.append(word[0])
  
  return [first]
```

바로 키워드의 첫 음절을 추출하는 것인데요. 규칙에 따르면, chars = [김, 대, 부, 서, 원, 진] 이 추출되는 것입니다. 

## 1-2 Window size에 따른 단어 추출

이후, window size에 따라 chars에서 단어를 추출하게 됩니다. 아래 함수는 해당 과정을 구현한 코드인데, 여기서 window size는 음절의 크기를 의미합니다. 즉 window size 1은 김, 대, 부, 서, 원, 진과 같은 1음절 단어, window size2는 서원, 부서,…, 진부 등의 2음절 단어, window size 3는 부대원,…, 부서진 등의 3음절 단어를 생성하게 되는 것입니다.

```python
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
```

위의 함수를 적용하면, is_noun → [부대원, 김, 서, 진], [부서진, 김, 대, 원], …, [원서, 김진, 부대] 등 여러 조합의 단어 리스트가 생성됩니다. 이때 🤓 is_noun에 포함된 각 리스트의 음절의 합이 5인 것을 확인할 수 있는데요. 음절의 합 = 입력한 키워드의 개수가 되어야 하기 때문입니다.

# 2. 모델 관련 개념

전처리한 데이터를 바탕으로 모델 구축을 시작하게 됩니다. 프로젝트 모델은 n-gram에서 아아디어를 얻었습니다. 입력값이 영어일 경우, 줄임말을 이용하여 자연스러운 문장을 생성하는, 고성능의 좋은 모델(KeyToText 등)이 존재합니다. 그러나 한국어의 경우, 직관적으로 사용 가능한 모델이 부재합니다. 물론 BERT 등을 이용한 모델이 존재하나 직관적인 이해가 어렵다는 등의 여러 문제가 있습니다. 그리하여, 고성능의 모델은 아니지만, 직관적으로 이해가 가능하면서도 나름의 성능을 보일 수 있는 n-gram의 개념을 모델 구축에 사용하였습니다.

이제부터 n-gram이 무엇인지, 차근차근 살펴보시죠 🙌🏻

## 2-1 n-gram

n-gram에서 ‘n’은 문장에서 몇 개의 단어를 볼지를 의미하는 것입니다. n-gram은 n개의 연속적인 단어 나열이라고 볼 수 있고요. 가령 아래와 같은 문장이 있다고 가정합시다.

*로절린드의 지도 체제에 불만인 블룸과 친구들은 실바를 풀어줄 위험한 계획을 세운다.*

위의 문장에 대한 여러 n-gram의 경우의 수는 다음과 같습니다.

Unigrams: 로절린드의, 지도, 체제에, 불만인, 블룸과, 친구들은, 실바를, 풀어줄, 위험한, 계획을, 세운다
Bigrams: 로절린드의 지도, 지도 체제에, 체제에 불만인, 불만인 블룸과, …, 계획을 세운다
Trigrams: 로절린드의 지도 체제에, 지도 체제에 불만인, 체제에 불만인 블룸과,…, 위험한 계획을 세운다
4-grams: 로절린드의 지도 체제에 불만인, 지도 체제에 불만인 블룸과,…, 풀어줄 위험한 계획을 세운다

여러 n-grams 중, 프로젝트를 위한 모델은 Bigrams과 Trigrams을 이용하였습니다. 이러한 N-gram에도 한계점이 존재합니다.

N-gram의 아이디어를 활용한 모델링은 문장이 제시된 순서를 바탕으로 몇몇 인접한 단어들의 Sequence를 연산하고, 이들의 동시 발생 확률을 세는 식으로 이루어집니다. 이러한 방법론은 문장의 정보가 단순히 순방향으로만 전달될 수 있다는 심각한 문제를 안고 있는데요. **한국말는 문장 내 단어들의 나열 순서가 그다지 중요하지 않기 때문입니다.** 아래 문장들을 살펴봅시다. 

*나는 너가 좋다* 

*너가 나는 좋다*

*나는 좋다 너가*

*너가 좋다 나는*

*좋다 나는 너가*

*좋다 너가 나는*

이 문장들은 약간의 뉘앙스 차이가 있긴 하지만 거의 비슷한 의미를 전달하고 있음을 알 수 있습니다. 이에 최신의 딥러닝 프레임워크는 Bidirectionality나 Attention based mechanism을 모델에 반영하곤 합니다. 하지만 이러한 프레임워크를 바탕으로 corpus를 학습시키에는 저희가 가진 학습 자원이 부족했습니다. 이에 저희는 N-gram을 사용하되 단어 나열 순서 변수를 제거하기 위해 permutation을 활용했습니다. Permutation을 활용해 모든 케이스들을 확률 순으로 나열해 모두 고려하는 방식으로 문제를 해결할 수 있었습니다. 

## 2-2 Markov Property

모델 구축에 활용된, 또 다른 개념인 Markov Property를 살펴봅시다. Markov Property란, 다음에 등장할 단어가 오직 이전 단어에만 의존하는 것을 의미합니다. 위에서 등장한 문장을 다시 한 번 살펴봅시다.

*로절린드의 지도 체제에 불만인 블룸과 친구들은 실바를 풀어줄 위험한 _______.*

위에서 등장한 것과의 차이점은 바로 ‘위험한’ 뒤에 빈칸이 존재한다는 것인데요. 이때, n = 3 이라고 하는 Trigrams을 이용한 언어 모델을 활용한다고 합시다. 그렇다면 ‘위험한’ 다음에 올 단어를 예측할 때에는 n-1에 해당하는 앞의 2개의 단어만 고려하는 것입니다.

수식을 통해 구체적으로 살봅시다.

조건부 확률: P(위험한, 풀어줄) = P(위험한|풀어줄)P(풀어줄)

연쇄 확률: P(위험한, 풀어줄) = P(위험한|풀어줄)P(풀어줄)

→ P(X, 위험한, 풀어줄) = P(X│위험한)P(위험한│풀어줄)P(X)

# 3. 모델 구축

## 3-1 Trigram, Bigram을 이용한 모델링

위에서 살펴본 개념들을 토대로, 모델을 적용한 사례를 살펴보도록 하겠습니다. 앞서 [부대원, 김, 서, 진], [부서진, 김, 대, 원], …, [원서, 김진, 부대] 등의 단어 리스트를 도출했는데요. 

아래 Bigram과 Trigram을 이용하여 가능한 자연스러운 문장의 경우와 이에 대한 확률을 생성해줍니다.

```python
def abb2sent(combs, model):  # input: window_size 별 조합 단어 리스트
  
  made = ['']
  prob = [0]

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
            else:  # trigram 으로도 불가능한 경우
              flag = 0
              break

        else:  # corpus 에 단어가 존재하지 않는 경우
          flag = 0
          break
          
      if flag:  # 문장이 만들어진 경우
        sentence += input[-1]
        made.append(sentence)
        prob.append((total_freq)**(1/(n + pp_cnt)))

      else:  # 해당 조합으로는 문장을 만들 수 없는 경우
        made.append("")
        prob.append(0)
  
  return made, prob  # output: 만들어진 문장과 각각 문장의 확률
```

코드를 유심히 살펴보면, 완전한 trigram 이 아닌 조사나 어미를 활용한 trigram 만 사용한다는 것을 알 수 있습니다. 모든 경우의 수를 다 고려하면 좋겠지만, 중간에 다른 단어가 들어가는 경우 필요하지 않은 단어가 문장에 너무 많이 등장할 수 있어서 배제했습니다. 또한 학습한 corpus 의 크기가 너무 방대해서 모든 경우의 수를 고려할 경우 output 을 생성해내는데 너무 많은 시간이 걸려서 조사와 어미만 활용했습니다.

## 3-2 Perplexity

위 함수를 바탕으로 생성할 수 있는 굉장히 다양한 문장들이 생성되었습니다. 하지만 이 문장들 모두가 현실 유사도, 즉 얼마나 자연스럽고 ‘인간’이 사용할 만한 문장인가에 대해서는 면밀히 검증해야 합니다. 이에 Language Modeling tasks에서 주로 활용되는 accuracy metrics 의 하나로 Perplexity를 연산하게 되었습니다. Perplexity는 번역하면 ‘복잡함’,  ’헷갈림’ 등으로 해석됩니다. 이 수치는 아래와 같은 수식으로 정의됩니다.

 

$PPL(W)=P(w_{1}, w_{2}, w_{3}, ... , w_{N})^{-\frac{1}{N}}=\sqrt[N]{\frac{1}{P(w_{1}, w_{2}, w_{3}, ... , w_{N})}}$

수식은 문장의 길이로 정규화된 문장 확률의 역수를 나타낸 것이며, N은 문장 W의 길이를 의미합니다. Perplexity를 N-gram에서 Markov property와 접목해 활용하면 아래과 같은 식이 만들어집니다.

$PPL(W)=\sqrt[N]{\frac{1}{\prod_{i=1}^{N}P(w_{i}| w_{i-1})}}$

 

```python
def words2sent(words, model):  # input: 줄임말을 생성하고 싶은 단어 리스트

  abb = words2abb(words)
  sents, prob = abb2sent(abb, model)

  best10 = []
  for _ in range(10):
    if max(prob):
      index = np.array(prob).argmax()
      best10.append(((sents.pop(index)), round(np.sqrt(1/prob.pop(index)), 3)))
    else:
      break

  return best10  # output: 최대 10개의 가장 자연스러운 문장
```

Perplexity 지수가 낮을수록 주어진 문장의 현실 유사도가 높다, 즉 모델의 완성도가 높다고 해석할 수 있습니다. 자연스러운 문장의 경우의 수와 이에 각 대응하는 Perplexity를 계산하면, 다음과 같은 결과가 도출됩니다.

[('부서와 대의 진원은 김', 18.589),

('대는 부서의 진원은 김', 20.84),

('원 대의 서진은 김 부', 21.543),

('서도 진원은 김 부 대', 21.989),

('원의 김진은 부서와 대', 22.199),

('진의 부만 원대에 서 김', 22.643),

('원의 서진은 김 부 대', 22.747),

('부진에 대의 원서는 김', 23.293),

('서 부 대의 진원은 김', 23.318),

('원 대 부는 진에 서 김', 23.345)]

이 중에서, Perplexity가 가장 낮은 ‘부서와 대의 진원은 김’을 최종 문장으로 선택하게 됩니다.

# 4. 연구의 한계

본 프로젝트는 다음과 같은 한계가 존재합니다.

- N-gram을 바탕으로 제작한 모델이기에 Transformers나 Attention 등 최신 딥러닝 네트워크에 비해 문장의 자유도 및 정확도가 낮다.
    
    본 프로젝트의 근간이 되는 아이디어는 NLP 모델링의 가장 고전적이면서도 기초적인 접근 방식인 N-gram 입니다. 이 방식은 본질적으로 사전에 정해진 규칙을 바탕으로 학습하기 때문에 규칙에 구애받지 않는 딥러닝 프레임워크에 비해 상대적으로 문장 표현의 자유도와 정확도 차원에서 비교적 어색한 결과를 내게 됩니다. 하지만 이러한 문제는 단순 노트북, 데스크탑에 기초한 환경이 아니라 더욱 더 심도있고 복잡한 연산을 무리없이 수행할 수 있는 환경에서는 충분히  해결될 것으로 보입니다. 
    
- Markov Property에 의존했기에 문장 전체의 응집도가 낮다.
    
    모델이 Markov Property에 의존하기에 한 단어의 생성과 sequence modeling을 위해서 바로 앞 두 단어만을 참고하여 Language modeling을 진행합니다. 이러한 모델의 전제 조건 때문에 전체 응집도가 높은 문장 생성의 차원에서는 다소 한계점을 보였습니다. 
    

본 프로젝트가 NLP라는 분야에 대해 관심을 가지는 기회를 제공하였기를 바랍니다 🏃🏻‍♂️ 감사합니다 😀
