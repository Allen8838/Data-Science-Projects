"""
filter for restaurants and collect restaurant reviews
"""

import codecs
import json

def create_frozen_set_of_business_id(businesses_filepath):
    """
    create restaurant dataset
    """
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


def create_review_text_file(review_txt_filepath, review_json_filepath, restaurant_ids):
    """
    collect restaurant reviews
    """
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
