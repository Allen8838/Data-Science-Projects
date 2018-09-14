import json
import pandas as pd 

def test_1():
    # file_business, file_checkin, file_review, file_tip, file_user = ['../Yelp Dataset/yelp_academic_dataset_business.json', 
    # '../Yelp Dataset/yelp_academic_dataset_checkin.json', '../Yelp Dataset/yelp_academic_dataset_review.json', 
    # '../Yelp Dataset/yelp_academic_dataset_tip.json', '../Yelp Dataset/yelp_academic_dataset_user.json']

    # with open(file_business, encoding='utf-8') as f:
    #     df_business = pd.DataFrame(json.loads(line) for line in f)

    # print(df_business.columns.values)
    # print(df_business.head(2))

    df_business = pd.read_csv(r'../Yelp Dataset/csv/yelp_academic_dataset_business.csv')
    print(df_business['attributes.RestaurantsReservations'])


if __name__ == '__main__':
    test_1()
