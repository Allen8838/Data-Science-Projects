import pandas as pd 
import numpy as np 
import dask.dataframe as dd
import matplotlib.pyplot as plt 
from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from memory_profiler import profile
from multiprocessing import Pool
import gc
import pickle 
from dask_ml.model_selection import train_test_split

@profile
def read_data_n_create_histo(filepath):
    
    df = dd.read_csv(filepath)

    df = df[['business_id', 'city', 'state', 'name', 'attributes.Ambience', 'stars', 'text', 'funny']]
    figure = plt.figure(figsize=(800/96, 800/96), dpi=96)
    df['target'] = df['stars']
    plt.hist(df['target'])
    figure.tight_layout()
    figure.savefig('Distribution of ratings', dpi=figure.dpi)

    return df

@profile 
def create_dataset(df, column):
    #using iterrows for now. tried to use enumerate to get each text value but received a not implemented error
    # documents = [row.text for i, row in enumerate(df.itertuples(), 1)]
    # filename = 'review documents'
    # outfile = open(filename, 'wb')
    # pickle.dump(documents, outfile)
    # outfile.close()
    dataframe_review_unpacked = df[column].compute()
    documents = [review for review in dataframe_review_unpacked.values]
    print(documents[1])
    #documents = [t for i,t in enumerate(df[column])]
    
    
    # documents = df[column]
    # documents.compute()
    #print(documents[1])
    #print(type(documents.head(1)))

    return documents

@profile 
def create_training_test_set(df):
    # dataset = dataset.compute()
    # target = target.compute()

    #page 17 of dask documentation
    df_train, df_test = df.random_split([0.8, 0.2], random_state=2)

    documents_train = df_train['text']
    documents_test = df_test['stars']

    gc.collect()
    return documents_train, documents_test

@profile 
def find_similar_reviews(dataset_train, dataset_test, vectors_train, n):
    # Draw an arbitrary review from test (unseen in training) documents# Draw a 
    #hardcoding this for now. using randint returned memoryerror
    doc_test = dataset_test[1]
    #doc_test = dataset_test[np.random.randint(len(dataset_test))]
    # Transform the drawn review(s) to vector(s)
    doc_test_vector = TfidfVectorizer(stop_words = 'english', max_features=5000).transform([doc_test]).toarray()
    # Calculate the similarity score(s) between vector(s) and training vectors
    similarity_scores = cosine_similarity(doc_test_vector, vectors_train.toarray())
    #search for n similar reviews
    searched_result = get_top_values(similarity_scores[0], n, dataset_train)
    return doc_test, searched_result

def process_vector_transform_parallel(dataset, function_for_parallel, partitions, processes, axis, arguments):
    pool = Pool(processes)

    dataset_split = np.array_split(dataset, partitions, axis=axis)
    dataframe = pd.concat(pool.map(function_for_parallel, dataset_split))
    pool.close()
    pool.join()

    return dataframe


def transform_vectorizer_to_array(dataset_test):
    vectors_test = TfidfVectorizer(stop_words = 'english', max_features=5000).transform(dataset_test).to_array()
    return vectors_test


@profile 
def create_nlp_rep_of_train_test(dataset_train, dataset_test):
    #change the dask dataframes to pandas dataframes
    dataset_train = dataset_train.compute()
    dataset_test = dataset_test.compute()

    # Create TfidfVectorizer, and name it vectorizer
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features=5000)

    # Train the model with your training data
    vectors_train = vectorizer.fit_transform(dataset_train)

    # Get the vocab of your tfidf
    words = vectorizer.get_feature_names()

    # Use the trained model to transform your test data
    #vectors_test = process_vector_transform_parallel(dataset_test, transform_vectorizer_to_array, 10, 6, 0, vectorizer)
    vectors_test = vectorizer.transform(dataset_test).toarray() 

    return vectors_train, words, vectors_test

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
    # df.info(memory_usage='deep')
    # for dtype in ['float','int','object']:
    #     selected_dtype = df.select_dtypes(include=[dtype])
    #     mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    #     mean_usage_mb = mean_usage_b / 1024 ** 2
    #     print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))

    documents = create_dataset(df, 'text')
    #infile = open('review documents', 'rb')
    #documents = pickle.load(infile)
    

    # dataset_train, dataset_test = create_training_test_set(df)
    # df = None
    # gc.collect()
    # vectors_train, words, vectors_test = create_nlp_rep_of_train_test(dataset_train, dataset_test)
    # doc_test, searched_result = find_similar_reviews(dataset_train, dataset_test, vectors_train, 5)
    # print(doc_test)
    # print(searched_result)