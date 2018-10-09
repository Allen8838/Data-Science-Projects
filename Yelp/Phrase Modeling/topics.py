"""
module to find topics in reviews
"""

import warnings

from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore

#import cPickle as pickle

def learn_vocab_corpus(trigram_reviews_filepath, trigram_dictionary_filepath):
    """
    create a dictionary based on the reviews
    """
    trigram_reviews = LineSentence(trigram_reviews_filepath)

    # learn the dictionary by iterating over all of the reviews
    trigram_dictionary = Dictionary(trigram_reviews)

    # filter tokens that are very rare or too common from
    # the dictionary (filter_extremes) and reassign integer ids (compactify)
    trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)
    trigram_dictionary.compactify()

    trigram_dictionary.save(trigram_dictionary_filepath)


def trigram_bow_generator(filepath, trigram_dictionary):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """

    for review in LineSentence(filepath):
        yield trigram_dictionary.doc2bow(review)

def create_bow(trigram_reviews_filepath, trigram_bow_filepath, trigram_dictionary):
    """
    generate bag-of-words representations for
    # all reviews and save them as a matrix
    """

    MmCorpus.serialize(trigram_bow_filepath,
                       trigram_bow_generator(trigram_reviews_filepath, trigram_dictionary))


def create_topics(lda_model_filepath, trigram_bow_corpus, trigram_dictionary):
    """
    creates and saves topic to file called lda
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # workers => sets the parallelism, and should be
        # set to number of physical cores minus one
        lda = LdaMulticore(trigram_bow_corpus,
                           num_topics=50,
                           id2word=trigram_dictionary,
                           workers=3)

    lda.save(lda_model_filepath)


def explore_topic(lda, topic_number, topn=25):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """

    print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')

    for term, frequency in lda.show_topic(topic_number, topn=25):
        print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))
