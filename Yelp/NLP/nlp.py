import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from memory_profiler import profile

@profile
def read_data_n_create_histo(filepath):
    df = pd.read_csv(filepath)
    figure = plt.figure(figsize=(800/96, 800/96), dpi=96)
    df.target = df.stars
    plt.hist(df.target)
    figure.tight_layout()
    figure.savefig('Distribution of ratings', dpi=figure.dpi)

    return df

@profile 
def create_dataset(df, column):
    documents = [t for i,t in enumerate(df[column])]
    return documents

@profile 
def create_training_test_set(dataset, target):
    documents_train, documents_test, target_train, target_test = train_test_split(dataset, target, test_size=0.33, random_state=42)

    return documents_train, documents_test, target_train, target_test

@profile 
def find_similar_reviews(dataset_train, dataset_test, vectorizer, vectors_train, n):
    # Draw an arbitrary review from test (unseen in training) documents# Draw a 
    #hardcoding this for now. using randint returned memoryerror
    doc_test = dataset_test[1]
    #doc_test = dataset_test[np.random.randint(len(dataset_test))]
    # Transform the drawn review(s) to vector(s)
    doc_test_vector = vectorizer.transform([doc_test]).toarray()
    # Calculate the similarity score(s) between vector(s) and training vectors
    similarity_scores = cosine_similarity(doc_test_vector, vectors_train.toarray())
    #search for n similar reviews
    searched_result = get_top_values(similarity_scores[0], n, dataset_train)
    return doc_test, searched_result


@profile 
def create_nlp_rep_of_train_test(dataset_train, dataset_test):
    # Create TfidfVectorizer, and name it vectorizer
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features=5000)

    # Train the model with your training data
    vectors_train = vectorizer.fit_transform(dataset_train)

    # Get the vocab of your tfidf
    words = vectorizer.get_feature_names()

    # Use the trained model to transform your test data
    vectors_test = vectorizer.transform(dataset_test).toarray() 

    return vectorizer, vectors_train, words, vectors_test

@profile 
def get_top_values(lst, n, labels):
    '''
    INPUT: LIST, INTEGER, LIST
    OUTPUT: LIST

    Given a list of values, find the indices with the highest n values.
    Return the labels for each of these indices.

    e.g.
    lst = [7, 3, 2, 4, 1]
    n = 2
    labels = ["cat", "dog", "mouse", "pig", "rabbit"]
    output: ["cat", "pig"]
    '''
    return [labels[i] for i in np.argsort(lst)[::-1][:n]] # np.argsort by default sorts values in ascending order

def get_bottom_values(lst, n, labels):
    '''
    INPUT: LIST, INTEGER, LIST
    OUTPUT: LIST

    Given a list of values, find the indices with the lowest n values.
    Return the labels for each of these indices.

    e.g.
    lst = [7, 3, 2, 4, 1]
    n = 2
    labels = ["cat", "dog", "mouse", "pig", "rabbit"]
    output: ["mouse", "rabbit"]
    '''
    return  [labels[i] for i in np.argsort(lst)[::1][:n]]




if __name__ == '__main__':
    df = read_data_n_create_histo('../Data_Preprocessing/restaurant_n_reviews.csv')
    documents = create_dataset(df, 'text')
    dataset_train, dataset_test, target_train, target_test = create_training_test_set(documents, df['stars'])
    vectorizer, vectors_train, words, vectors_test = create_nlp_rep_of_train_test(dataset_train, dataset_test)
    doc_test, searched_result = find_similar_reviews(dataset_train, dataset_test, vectorizer, vectors_train, 5)
    print(doc_test)
    print(searched_result)