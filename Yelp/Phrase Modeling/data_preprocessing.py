import os
import codecs
import json

def create_frozen_set_of_business_id(businesses_filepath):
    restaurant_ids = set()

    # open the businesses file
    with codecs.open(businesses_filepath, encoding='utf_8') as f:
        
        # iterate through each line (json record) in the file
        for business_json in f:
            
            # convert the json record to a Python dict
            business = json.loads(business_json)
            
            try:
                # if this business is not a restaurant, skip to the next one
                if u'Restaurants' not in business[u'categories']:
                    continue
            
                # add the restaurant business id to our restaurant_ids set
                restaurant_ids.add(business[u'business_id'])
            except:
                continue

    # turn restaurant_ids into a frozenset, as we don't need to change it anymore
    restaurant_ids = frozenset(restaurant_ids)

    return restaurant_ids


def create_review_text_file(review_txt_filepath, review_json_filepath):
    review_count = 0  
    # create & open a new file in write mode
    with codecs.open(review_txt_filepath, 'w', encoding='utf_8') as review_txt_file:

        # open the existing review json file
        with codecs.open(review_json_filepath, encoding='utf_8') as review_json_file:

            # loop through all reviews in the existing file and convert to dict
            for review_json in review_json_file:
                review = json.loads(review_json)

                # if this review is not about a restaurant, skip to the next one
                if review[u'business_id'] not in restaurant_ids:
                    continue

                # write the restaurant review as a line in the new file
                # escape newline characters in the original review text
                review_txt_file.write(review[u'text'].replace('\n', '\\n') + '\n')
                review_count += 1

    return review_count
    


if __name__ == '__main__':
    #set path for data
    data_directory = os.path.join('..', 'Yelp Dataset')

    businesses_filepath = os.path.join(data_directory, 'yelp_academic_dataset_business.json')
    review_json_filepath = os.path.join(data_directory,'yelp_academic_dataset_review.json')

    restaurant_ids = create_frozen_set_of_business_id(businesses_filepath)
    # print the number of unique restaurant ids in the dataset
    print('{:,}'.format(len(restaurant_ids)), u'restaurants in the dataset.')
    #57,173 restaurants in the dataset.

    #create a new file that contains only the text from reviews about restaurants, with one review per line in the file
    intermediate_directory = os.path.join('..', 'Reviews')

    review_txt_filepath = os.path.join(intermediate_directory, 'review_text_all.txt')

    review_count = create_review_text_file(review_txt_filepath, review_json_filepath)

    print(u'''Text from {:,} restaurant review written to the new txt file.'''.format(review_count))
    #Text from 3,654,797 restaurant review written to the new txt file

    