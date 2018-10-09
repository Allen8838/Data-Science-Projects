"""
main module to run submodules
"""
import os

from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import Phrases

from data_preprocessing import create_frozen_set_of_business_id, create_review_text_file
from n_gram_modeling import create_unigram, create_bigram, create_trigram, wrt_trigram_rvs_to_txt

from topics import learn_vocab_corpus,\
                   create_bow,\
                   create_topics,\
                   explore_topic


if __name__ == '__main__':
    """
    data preprocessing step
    """
    #set path for data
    data_directory = os.path.join('..', 'Yelp Dataset')

    businesses_filepath = os.path.join(data_directory, 'yelp_academic_dataset_business.json')
    review_json_filepath = os.path.join(data_directory, 'yelp_academic_dataset_review.json')

    restaurant_ids = create_frozen_set_of_business_id(businesses_filepath)
    # print the number of unique restaurant ids in the dataset
    print('{:,}'.format(len(restaurant_ids)), u'restaurants in the dataset.')
    #57,173 restaurants in the dataset.

    #create a new file that contains only the text from reviews about restaurants,
    #with one review per line in the file
    intermediate_directory = os.path.join('..', 'Reviews')

    review_txt_filepath = os.path.join(intermediate_directory, 'review_text_all.txt')

    review_count = create_review_text_file(review_txt_filepath, review_json_filepath, restaurant_ids)

    print(u'''Text from {:,} restaurant review written to the new txt file.'''.format(review_count))
    #Text from 3,654,797 restaurant review written to the new txt file

    """
    unigram modeling
    """

    #stream in unigram sentences to reduce load to RAM
    #step 2
    unigram_sentences_filepath = os.path.join('results', 'unigram_sentences_all.txt')
    create_unigram(unigram_sentences_filepath, review_txt_filepath)
    unigram_sentences = LineSentence(unigram_sentences_filepath)

    """
    bigram modeling
    """

    #bigram model
    bigram_model_filepath = os.path.join('results', 'bigram_model_all')

    bigram_model = Phrases(unigram_sentences)

    bigram_model.save(bigram_model_filepath)

    #load the finished model from disk
    bigram_model = Phrases.load(bigram_model_filepath)

    bigram_sentences_filepath = os.path.join('results', 'bigram_sentences_all.txt')

    create_bigram(bigram_sentences_filepath, unigram_sentences, bigram_model)

    """
    trigram modeling
    """
    bigram_sentences = LineSentence(bigram_sentences_filepath)

    trigram_model = Phrases(bigram_sentences)

    trigram_model_filepath = os.path.join('results', 'trigram_model_all')

    trigram_model.save(trigram_model_filepath)

    # load the finished model from disk
    trigram_model = Phrases.load(trigram_model_filepath)


    trigram_sentences_filepath = os.path.join('results', 'trigram_sentences_all.txt')

    create_trigram(trigram_sentences_filepath, bigram_sentences, trigram_model)

    trigram_sentences = LineSentence(trigram_sentences_filepath)

    trigram_reviews_filepath = os.path.join('results', 'trigram_transformed_reviews_all.txt')

    review_txt_filepath = os.path.join('../Reviews', 'review_text_all.txt')

    wrt_trigram_rvs_to_txt(trigram_reviews_filepath, review_txt_filepath,
                           trigram_model, bigram_model)

    """
    create bag of words
    """
    trigram_reviews_filepath = os.path.join('results', 'trigram_transformed_reviews_all.txt')

    trigram_dictionary_filepath = os.path.join('trigram_dict_all.dict')

    learn_vocab_corpus(trigram_reviews_filepath, trigram_dictionary_filepath)

    # load the finished dictionary from disk
    trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)

    trigram_bow_filepath = os.path.join('trigram_bow_corpus_all.mm')

    create_bow(trigram_reviews_filepath, trigram_bow_filepath, trigram_dictionary)

    # load the finished bag-of-words corpus from disk
    trigram_bow_corpus = MmCorpus(trigram_bow_filepath)

    """
    find topics
    """

    lda_model_filepath = os.path.join('lda_model_all')

    create_topics(lda_model_filepath, trigram_bow_corpus, trigram_dictionary)

    # load the finished LDA model from disk
    lda = LdaMulticore.load(lda_model_filepath)

    explore_topic(lda, topic_number=0)
