import matplotlib.pyplot as plt
import numpy as np

def calculate_project_costs_by_metro_type(projects_df, schools_df):
    cost = schools_df[['School ID', 'School Metro Type']].merge(projects_df[['School ID', 'Project Cost']], left_on='School ID', right_on='School ID', how='left')
    cost.groupby('School Metro Type', as_index=False).agg(['median', 'mean', 'max', 'min'])

    return cost

def find_distribution_of_project_cost_by_metro_type(projects_df, schools_df):
    metro = projects_df[['School ID', 'Project Cost']].merge(schools_df[['School ID', 'School Metro Type']], left_on='School ID', right_on='School ID', how='left')

    rural = metro[metro['School Metro Type']=='rural']
    urban = metro[metro['School Metro Type']=='urban']
    suburb = metro[metro['School Metro Type']=='suburban']
    town = metro[metro['School Metro Type']=='town']

    f, ax = plt.subplots(2,2, figsize=(25,12))

    np.log(rural['Project Cost'].dropna()).hist(ax=ax[0,0], bins=30, edgecolor='black')
    ax[0,0].set_title('Project Cost of Rural Schools')

    np.log(urban['Project Cost'].dropna()).hist(ax=ax[0,1], bins=30, edgecolor='black')
    ax[0,1].set_title('Project Cost of Urban Schools')

    np.log(suburb['Project Cost'].dropna()).hist(ax=ax[1,0], bins=30, edgecolor='black')
    ax[1,0].set_title('Project Cost of Suburban Schools')

    np.log(town['Project Cost'].dropna()).hist(ax=ax[1,1], bins=30, edgecolor='black')
    ax[1,1].set_title('Project Cost of Town Schools')

    plt.savefig('Distribution of Project Cost by Metro Type.png')

def calculate_donations_from_home_state(donors_df, donations_df, projects_df, schools_df):
    home_don = donors_df[['Donor ID', 'Donor State']].merge(donations_df[['Donor ID', 'Project ID']], left_on='Donor ID', right_on='Donor ID', how='right')
    home_don = home_don.merge(projects_df[['Project ID', 'School ID']], left_on='Project ID', right_on='Project ID', how='right')
    home_don = home_don.merge(schools_df[['School ID', 'School State', 'School Zip']], left_on='School ID', right_on='School ID', how='left')
    home_don['Home State Donation'] = np.where(home_don['Donor State']==home_don['School State'], 'YES', 'NO')
    home_don['Home State Donation'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, shadow=True, explode=[0, 0.1])
    plt.gcf().set_size_inches(5,5)
    plt.savefig('Donations from home state.png')