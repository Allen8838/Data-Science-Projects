from gensim.models.phrases import Phrases
from gensim.models.word2vec import LineSentence
from helper import punct_space, line_review, lemmatized_sentence_corpus, extended_is_stop
import time
import os
import codecs
import spacy
nlp = spacy.load('en')

def create_bigram_sentences_from_unigram_sent(bigram_sentences_filepath, unigram_sentences, bigram_model):
    with codecs.open(bigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for unigram_sentence in unigram_sentences:        
            bigram_sentence = u' '.join(bigram_model[unigram_sentence])    
            f.write(bigram_sentence + '\n')


def create_trigram_sentences_from_bigram_sent(trigram_sentences_filepath, bigram_sentences, trigram_model):
    with codecs.open(trigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for bigram_sentence in bigram_sentences:        
            trigram_sentence = u' '.join(trigram_model[bigram_sentence])    
            f.write(trigram_sentence + '\n')


def write_trigram_reviews_to_text_file(trigram_reviews_filepath, review_txt_filepath, trigram_model, bigram_model):
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
                              if term not in extended_is_stop(term)]
            
            # write the transformed review as a line in the new file
            trigram_review = u' '.join(trigram_review)
            f.write(trigram_review + '\n')










if __name__ == '__main__':
    #stream in unigram sentences to reduce load to RAM
    #step 2
    unigram_sentences_filepath = os.path.join('results', 'unigram_sentences_all.txt')
    unigram_sentences = LineSentence(unigram_sentences_filepath)


    #bigram model
    bigram_model_filepath = os.path.join('results', 'bigram_model_all')

    bigram_model = Phrases(unigram_sentences)

    bigram_model.save(bigram_model_filepath)

    #load the finished model from disk
    bigram_model = Phrases.load(bigram_model_filepath)

    bigram_sentences_filepath = os.path.join('results', 'bigram_sentences_all.txt')

    create_bigram_sentences_from_unigram_sent(bigram_sentences_filepath, unigram_sentences, bigram_model)



    bigram_sentences = LineSentence(bigram_sentences_filepath)

    trigram_model = Phrases(bigram_sentences)

    trigram_model_filepath = os.path.join('results', 'trigram_model_all')

    trigram_model.save(trigram_model_filepath)

    # load the finished model from disk
    trigram_model = Phrases.load(trigram_model_filepath)


    trigram_sentences_filepath = os.path.join('results', 'trigram_sentences_all.txt')

    create_trigram_sentences_from_bigram_sent(trigram_sentences_filepath, bigram_sentences, trigram_model)

    
    
    trigram_sentences = LineSentence(trigram_sentences_filepath)

    trigram_reviews_filepath = os.path.join('results', 'trigram_transformed_reviews_all.txt')

    review_txt_filepath = os.path.join('../Reviews','review_text_all.txt')
    
    write_trigram_reviews_to_text_file(trigram_reviews_filepath, review_txt_filepath, trigram_model, bigram_model)
