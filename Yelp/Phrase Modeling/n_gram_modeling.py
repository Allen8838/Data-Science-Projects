"""
use gensim to learn one word, two word and three words
phrases in reviews
"""

import codecs
import spacy

from sentence_cleanup import punct_space, line_review, lemmatized_sentence_corpus, extended_is_stop

nlp = spacy.load('en')

def create_unigram(unigram_sentences_filepath, review_txt_filepath):
    """
    learn one word phrase in reviews
    """

    with codecs.open(unigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for sentence in lemmatized_sentence_corpus(review_txt_filepath):
            f.write(sentence + '\n')

def create_bigram(bigram_sentences_filepath, unigram_sentences, bigram_model):
    """
    learn two words phrase in reviews
    """

    with codecs.open(bigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for unigram_sentence in unigram_sentences:
            bigram_sentence = u' '.join(bigram_model[unigram_sentence])
            f.write(bigram_sentence + '\n')


def create_trigram(trigram_sentences_filepath, bigram_sentences, trigram_model):
    """
    learn three words phrase in reviews
    """

    with codecs.open(trigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for bigram_sentence in bigram_sentences:
            trigram_sentence = u' '.join(trigram_model[bigram_sentence])
            f.write(trigram_sentence + '\n')


def wrt_trigram_rvs_to_txt(trigram_reviews_filepath,
                           review_txt_filepath,
                           trigram_model,
                           bigram_model):

    """
    save words containing three word phrases to file
    """

    with codecs.open(trigram_reviews_filepath, 'w', encoding='utf_8') as f:

        for parsed_review in nlp.pipe(line_review(review_txt_filepath),
                                      batch_size=10000, n_threads=4):

            # lemmatize the text, removing punctuation and whitespace
            unigram_review = [token.lemma_ for token in parsed_review
                              if not punct_space(token)]

            # apply the first-order and second-order phrase models
            bigram_review = bigram_model[unigram_review]
            trigram_review = trigram_model[bigram_review]

            # remove any remaining stopwords
            trigram_review = [term for term in trigram_review
                              if not extended_is_stop(term)]

            # write the transformed review as a line in the new file
            trigram_review = u' '.join(trigram_review)
            f.write(trigram_review + '\n')
