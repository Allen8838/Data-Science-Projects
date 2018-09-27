import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import time

def create_initial_conditions(filepath):
    df = pd.read_csv(filepath)

    #select certain columns from dataframe
    df_utility = pd.pivot_table(data=df, values='stars', index='user_id', columns='business_id', fill_value=0)

    df.utility_info()

    df.set_index('user_id', inplace=True)
    utility_mat = df_utility.as_matrix()
    item_sim_mat = cosine_similarity(utility_mat.T)
    least_to_most_sim_indexes = np.argsort(item_sim_mat, axis=1)
    neighborhood_size = 75
    neighborhoods = least_to_most_sim_indexes[:, -neighborhood_size:]

    return df, df_utility, item_sim_mat, neighborhood_size, neighborhoods


def create_recommendations(df, df_utility, n, pred_ratings, utility_mat):
    businesses_to_recommend = []
    # Get item indexes sorted by predicted rating
    item_index_sorted_by_pred_rating = list(np.argsort(pred_ratings))

    # Find items that have been rated by user
    items_rated_by_this_user = utility_mat[user_id].nonzero()[0]

    # We want to exclude the items that have been rated by user
    unrated_items_by_pred_rating_it = [item for item in item_index_sorted_by_pred_rating if item not in items_rated_by_this_user]

    unrated_items_by_pred_rating_it[-n:]

    recommend_business_id = df_utility.columns[unrated_items_by_pred_rating_it[-n:]].values

    for business in recommend_business_id:
        businesses_to_recommend.append(df.loc[df.business_id == business].name.values[0])

    return businesses_to_recommend




def find_ratings_of_a_user(user_id, utility_mat, item_sim_mat):
    n_users = utility_mat.shape[0]
    n_items = utility_mat.shape[1]

    items_rated_by_this_user = utility_mat[user_id].nonzero()[0]

    #for rating predictions
    out = np.zeros(n_items)

    for item_to_rate in range(n_items):
        relevant_items = np.intersect1d(neighborhoods[item_to_rate], items_rated_by_this_user, assume_unique=True)
        if len(relevant_items) != 0:
            out[item_to_rate] = sum(utility_mat[user_id, relevant_items])*item_sim_mat[item_to_rate, relevant_items]/item_sim_mat[item_to_rate, relevant_items].sum()
        else:
            out[item_to_rate] = np.nan
    
    pred_ratings = np.nan_to_num(out)

    return out[~np.isnan(out)], pred_ratings

if __name__ == '__main__':
    df, df_utility, item_sim_mat, neighborhood_size, neighborhoods = create_initial_conditions('../Data_Preprocessing/restaurant_n_reviews.csv')
    output_not_nan, pred_ratings = find_ratings_of_a_user(user_id, utility_mat, item_sim_mat)
    businesses_to_recommend = create_recommendations(df, df_utility, n, pred_ratings, utility_mat)  




