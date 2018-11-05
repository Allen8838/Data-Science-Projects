from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import csv

def lda_modelling(projects_df):
    need = projects_df['Project Need Statement'].dropna().values[:100000]
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')

    tf = tf_vectorizer.fit_transform(need)

    tf_feature_names = tf_vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_components=10, max_iter=5, learning_method='online', learning_offset=50, random_state=123).fit(tf)

    # write to csv file
    lda_csv_file = open('lda_csv_file.csv', 'w')
    writer = csv.writer(lda_csv_file)
    
    for topic in lda.components_:
        writer.writerows(" ".join([tf_feature_names[i] for i in topic.argsort()[:-10 -1: -1]]))

    return None




