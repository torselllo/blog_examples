#this is a simple and not-optimized sketch I created for the blogpost http://nlpx.net/archives/330
#you need to have the following external libraries installed: pattern, scikit-learn, gensim, stopwords
import gensim
from sklearn.datasets import fetch_20newsgroups
import logging
from collections import defaultdict
import stopwords
from pattern.en import lemma

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def lda_comparison(corpus_savepath):
    '''string -> lda topics
    corpus_savepath is the path to save the prepared corpus for lda'''

    #basic preprocessing and lemmatization almost like in lda2vec implementation
    texts = fetch_20newsgroups(subset='train').data
    texts = [unicode(d.lower()) for d in texts]
    texts = ["".join((char if char.isalpha() else " ") for char in text).split() for text in texts]
    texts = [stopwords.clean([lemma(i) for i in text[:1000]], "en") for text in texts]

    #creating frequency dictionary for tokens in text
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    #removing very infrequent and very frequent tokens in corpus
    texts = [[token for token in text if (frequency[token] > 10 and len(token) > 2 and frequency[token] < len(texts)*0.2)] for text in texts]

    #creating an LDA model
    dictionary = gensim.corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    gensim.corpora.MmCorpus.serialize(corpus_savepath, corpus)
    modelled_corpus = gensim.corpora.MmCorpus(corpus_savepath)
    lda = gensim.models.ldamodel.LdaModel(modelled_corpus, num_topics=20, update_every=100, passes=20, id2word=dictionary, alpha='auto', eval_every=5)

    #returning the resulting topics
    return lda.show_topics(num_topics=20, num_words=10, formatted=True)

if __name__ == '__main__':
    path = '/home/user/lda_test/gensimldatest.mm'
    print lda_comparison(path)
