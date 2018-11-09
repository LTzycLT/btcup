
# coding: utf-8

# In[31]:


import json
data = []
with open("data/bytecup.corpus.train.0.txt") as f:
    for line in f:
        data.append(json.loads(line))


# In[32]:


import nltk


# In[33]:


s_tokenizer = nltk.RegexpTokenizer('[\w\' ]+')
w_tokenizer = nltk.RegexpTokenizer('[\w\']+')


# In[34]:


for cnt, d in enumerate(data):
    if cnt % 10000 == 0:
        print(cnt)
    d['sents'] = s_tokenizer.tokenize(d['content'])


# In[35]:


def LCS(x, y):
    dp = [[0 for _ in y] for _ in x]
    for i in range(len(x)):
        for j in range(len(y)):
            if i > 0:
                dp[i][j] =  max(dp[i][j], dp[i - 1][j])
            if j > 0:
                dp[i][j] = max(dp[i][j], dp[i][j - 1])
            if x[i] == y[j]:
                if i > 0 and j > 0: dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1)
                else: dp[i][j] = 1   
    maxx = 0
    for i in range(len(x)):
        for j in range(len(y)):
            maxx = max(maxx, dp[i][j])
    return maxx

def rouge(x, y):
    global lx, ly
    lx += len(x)
    ly += len(y)
    lcs = LCS(x, y)
    if lcs == 0: return 0
    recall = lcs * 1.0 / len(y)
    precision = lcs * 1.0 / len(x)
    beta = precision / (recall + 1e-12)
    return (1 + beta * beta) * recall * precision / (recall + beta * beta * precision)


# In[36]:


s, c = 0, 0
lx = 0
ly = 0
for d in data[3000:4000]:
    maxx = 0
    y = w_tokenizer.tokenize(d['title'])
    for sent in d['sents']:
        x = w_tokenizer.tokenize(sent)
        maxx = max(maxx, rouge(x, y))
    s += maxx
    c += 1


# In[37]:


d['content']


# In[38]:


d['title']


# In[71]:


from __future__ import print_function
with open('data/bytecup.corpus.validation_set.txt') as f:
    for line in f:
        obj = json.loads(line)
        content = obj['content']
        with open('data/result/%s.txt' % obj['id'], 'w') as fout:
            print(s_tokenizer.tokenize(content)[0].encode('utf8'), file=fout)

