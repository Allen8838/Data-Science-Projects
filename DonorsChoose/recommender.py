import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import scipy
import sklearn

projects = pd.read_csv('Projects.csv')

donations = pd.read_csv('Donations.csv')

donors = pd.read_csv('Donors.csv')

# creates a new column, project_id, that starts from 10 and ends last count of the project plus 10
f = len(projects)
projects['project_id'] = np.nan
g = list(range(10, f+10))
g = pd.Series(g)
projects['project_id'] = g.values

projects.head(5000).to_csv('projects_from_Projects_file.csv')

# merge datasets
donations = donations.merge(donors, on='Donor ID', how='left')
df = donations.merge(projects, on='Project ID', how='left')

donations_df = df

# Deal with missing values
donations['Donation Amount'] = donations['Donation Amount'].fillna(0)

# Define event strength as donated amount to a certain project
donations_df['eventStrength'] = donations_df['Donation Amount']


def smooth_donor_preference(x):
    return math.log(1+x, 2)

donations_full_df = donations_df\
                    .groupby(['Donor ID', 'Project ID'])['eventStrength'].sum()\
                    .apply(smooth_donor_preference).reset_index()

# donations_full_df will look in the form of 

# Donor ID,                           Project ID,                          eventStrength
# 00000ce845c00cbf0686c992fc369df4, 5bab6101eed588c396a59f6bd64274b6,    5.67242534197149

# update project dataset
# Not sure what this has accomplished
project_cols = projects.columns
projects = df[project_cols].drop_duplicates()

donations_train_df, donations_test_df = train_test_split(donations_full_df, test_size=0.20, random_state=42)

# Indexing by Donor ID to speed up the searches during evaluation
donations_full_indexed_df = donations_full_df.set_index('Donor ID')
donations_train__indexed_df = donations_train_df.set_index('Donor ID')
donations_test_indexed_df = donations_test_df.set_index('Donor ID')

# Preprocessing of text data
textfeats = ["Project Title", "Project Essay"]

for cols in textfeats:
    projects[cols] = projects[cols].astype(str)
    projects[cols] = projects[cols].astype(str).fillna('')
    projects[cols] = projects[cols].str.lower()

text = projects['Project Title'] + ' ' + projects['Project Essay']

vectorizer = TfidfVectorizer(strip_accents='unicode',
                             analyzer='word',
                             lowercase=True,
                             stop_words='english',
                             max_df=0.9)

project_ids = projects['Project ID'].tolist()
tfidf_matrix = vectorizer.fit_transform(text)

# tfidf_matrix will look in the form of. this is a scipy sparse matrix
#   (0, 86231)    0.18043116799737025
#   (0, 81767)    0.1246356155583376
#   (0, 22477)    0.2190395191964646
#   (0, 21249)    0.26066742427773676
#   (0, 81746)    0.4410626006376963
#   and so on

# read the sparse matrix as
# Assume general form: (A,B) C

# A: Document index B: Specific word-vector index C: TFIDF score for word B in document A     



tfidf_feature_names = vectorizer.get_feature_names()

def get_project_profile(project_id):
    idx = project_ids.index(project_id)
    project_profile = tfidf_matrix[idx:idx+1]

    # project_profile will look in the form of. this is a scipy sparse matrix
    # (0, 106686)    0.10452968447326941
    # (0, 135238)    0.05905238364194011
    # (0, 71476)     0.02192229196906363
    # and so on

    return project_profile


def get_project_profiles(ids):
    project_profiles_list = [get_project_profile(x) for x in np.ravel([ids])]
    # project_profiles_list should be a sparse matrix, which is why we use scipy.sparse.vstack below
    
    project_profiles = scipy.sparse.vstack(project_profiles_list)

    # project_profile will look in the form of 
    # (0, 106686)    0.10452968447326941
    # (0, 135238)    0.05905238364194011
    # (0, 71476)     0.02192229196906363
    # and so on

    return project_profiles

def build_donors_profile(donor_id, donations_indexed_df):
    donations_donor_df = donations_indexed_df.loc[donor_id] # get the group of rows of a given donor_id 
    # donations_donor_df will come in the form of something like
    # Project ID      ee8e7795....a60f
    # eventStrength            6.65821
    # Name          0014244b5a...cc2c7
    # essentially, it took a subset of the original dataframe and rotated it so that all the column headings are stacked on a vertical row to the left
    # and all of the values are stacked vertically on the right

    donor_project_profiles = get_project_profiles(donations_donor_df['Project ID'])

    donor_project_strengths = np.array(donations_donor_df['eventStrength']).reshape(-1, 1) # come in the form of something like [[5.6724]]
    # come in the form of [[5.6724]]

    donor_project_strengths_weighted_avg = np.sum(donor_project_profiles.multiply(donor_project_strengths), axis=0)/(np.sum(donor_project_strengths)+1) # come in the form of something like [[0, 0, 0, ... 0, 0, 0]]
    # come in the form of [[0, 0, 0, ...., 0, 0, 0]]
     
    donor_profile_norm = sklearn.preprocessing.normalize(donor_project_strengths_weighted_avg)
    # I think this step may be needed as you have different donations amounts (remember donation amounts is the donation eventStrength)
    # come in the form of [[0, 0, 0, ...., 0, 0, 0]]

    return donor_profile_norm

def build_donors_profiles(donations_full_df):
    donations_indexed_df = donations_full_df[donations_full_df['Project ID'].isin(projects['Project ID'])].set_index('Donor ID') # make sure that we filter out donations without a Project ID

    donor_profiles = {}
    for donor_id in donations_indexed_df.index.unique(): # return unique values in the index
        donor_profiles[donor_id] = build_donors_profile(donor_id, donations_indexed_df)

    return donor_profiles

donor_profiles = build_donors_profiles(donations_full_df)

print(donor_profiles)


# Creating a sparse pivot table with donors in rows and projects in columns
donors_projects_pivot_matrix_df = donations_full_df.pivot(index='Donor ID', 
                                                          columns='Project ID', 
                                                          values='eventStrength').fillna(0)
# shold take the form of 
# eventStrength       Project ID
#                         1       2          3    .......
# Donor ID            
#   1                   0.23    0.45        0.67  .......
#   2                    .       .            .   .......
#   3                    .       .            .   .......
#   .                    .       .            .   .......

# Transform the donor-project dataframe into a matrix
donors_projects_pivot_matrix = donors_projects_pivot_matrix_df.as_matrix()

# this will then look like
# [[0.23, 0.45, 0.67] .......]

# Get donor ids
donors_ids = list(donors_projects_pivot_matrix_df.index)

U, sigma, Vt = svds(donors_projects_pivot_matrix, k=20)
sigma = np.diag(sigma)

# Reconstruct the matrix by multiplying its factors 
all_donor_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

cf_preds_df = pd.DataFrame(all_donor_predicted_ratings, 
                           columns = donors_projects_pivot_matrix_df.columns,
                           index= donors_ids).transpose() # note the transpose here. so we have the transpose of donors_projects_pivot_matrix_df
# eventStrength       Donor ID
#                         1       2          3    .......
# Project ID            
#   1                   0.23    0.45        0.67  .......
#   2                    .       .            .   .......
#   3                    .       .            .   .......
#   .                    .       .            .   .......






# mydonor1 = "6d5b22d39e68c656071a842732c63a0c"
# mydonor2 = "0016b23800f7ea46424b3254f016007a"

# mydonor1_profile = pd.DataFrame(sorted(zip(tfidf_feature_names,
#                                 donor_profiles[mydonor1].flatten().tolist()),
#                                 key=lambda x: -x[1])[:10],
#                                 columns=['token', 'relevance'])

# mydonor2_profile = pd.DataFrame(sorted(zip(tfidf_feature_names,
#                                 donor_profiles[mydonor2].flatten(),tolist()),
#                                 key=lambda x: -x[1])[:10],
#                                 columns=['token', 'relevance'])




