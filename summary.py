from summa.summarizer import summarize
from summa import keywords
import json

with open("data/raw/bytecup.corpus.train.0.txt") as f:
    for cnt, line in enumerate(f):
        if cnt > 10: break
        obj = json.loads(line)
        s = summarize(obj['content'])
        print(s)
        #print(keywords.keywords(obj['content']))
        print(obj['content'].split('.')[0])
        print(obj['title'])
        print("==================")
