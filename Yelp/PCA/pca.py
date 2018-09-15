import pandas as pd 
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np 

def create_target_variable_and_define_feature_variable():
    #read in dataset
    df = pd.read_csv(r'../Data_Preprocessing/restaurant_n_reviews.csv')

    #print(df.columns.values)
    #get reviews
    documents = df.text.values

    #create target variable. in this case the target variable is the star rating
    df['target'] = df.stars.apply(lambda x: 1 if x > 4 else 0)

    return df, documents
    
def create_training_set_and_testing_set(df, documents):
    
    #create training set and testing dataset
    documents_train, documents_test, target_train, target_test = train_test_split(
        documents, df['target'], test_size=0.9, random_state=3
    )

    return documents_train, documents_test, target_train, target_test

def cluster_by_kmeans(documents_train):
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features=1000)

    vectorized_train = vectorizer.fit_transform(documents_train).toarray()

    # Get the vocab of your tfidf
    #vectorizer.vocabulary_
    vocab = vectorizer.get_feature_names()

    # Use the trained model to transform all the reviews
    reviews = vectorizer.transform(documents).toarray()

    kmeans = KMeans()
    kmeans.fit(vectorized_train)

    cluster_pred = kmeans.predict(reviews)

    key10 = np.argsort(kmeans.cluster_centers_,1)[:,-1:-11:-1]

    #inspect the top 10 features for each cluster
    #type(key10)
    for i,c in enumerate(key10):
        print("%d: %s" % (i, ", ".join(vocab[i] for i in c)))

    return None


if __name__ == '__main__':
    df, documents = create_target_variable_and_define_feature_variable()
    documents_train, documents_test, target_train, target_test = create_training_set_and_testing_set(df, documents)
    cluster_by_kmeans(documents_train)    





