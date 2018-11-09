import json

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

def get_data(fnames):
    content = ""
    for fname in fnames:
        with open(fname) as f:
            for line in f:
                line = line.strip().lower();
                if line == "": continue # empty line
                if line.startswith("ychzhou"):
                    yield line.replace("ychzhou", "").strip(), content
                    content = ""
                else:
                    if line[-1] not in END_TOKENS:
                        line += ' .'
                    content += ' ' + line

def _summa(content):
    from summa.summarizer import summarize
    from summa import keywords
    ss = summarize(content, scores=True)
    sorted(ss, key=lambda o:o[1], reverse=True)
    print(ss[0])

def _gensim(content):
    from gensim.summarization.summarizer import summarize

#def _teaser(content):
#    from teaser import Sum

data = get_data(["data/tokenized/bytecup.corpus.train.0.txt"])
for cnt, (title, content) in enumerate(data):
    if cnt > 10: break
    _summa(content)
    print(content.split('.')[0])
    print(title)
    print("==================")
