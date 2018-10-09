from topics import learn_vocab_corpus,\
                   trigram_bow_generator,\
                   create_bow,\
                   create_topics,\
                   explore_topic

from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore

import os
import pyLDAvis
import pyLDAvis.gensim
import warnings
#import cPickle as pickle



if __name__ == '__main__':
    # trigram_reviews_filepath = os.path.join('results', 'trigram_transformed_reviews_all.txt')

    # trigram_dictionary_filepath = os.path.join('trigram_dict_all.dict')
    
    # learn_vocab_corpus(trigram_reviews_filepath, trigram_dictionary_filepath)
    
    # # load the finished dictionary from disk
    # trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)
    
    # trigram_bow_filepath = os.path.join('trigram_bow_corpus_all.mm')

    # create_bow(trigram_reviews_filepath, trigram_bow_filepath, trigram_dictionary)

    # # load the finished bag-of-words corpus from disk
    # trigram_bow_corpus = MmCorpus(trigram_bow_filepath)


    lda_model_filepath = os.path.join('lda_model_all')

    # create_topics(lda_model_filepath, trigram_bow_corpus, trigram_dictionary)

    # # load the finished LDA model from disk
    lda = LdaMulticore.load(lda_model_filepath)

    explore_topic(lda, topic_number=49)
