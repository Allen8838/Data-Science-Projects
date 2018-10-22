
def create_union_of_keys(dict1, dict2):
    """
    finds the union of two dictionaries, where dict1 keys are placed first
    """
    union_of_all_keys = []
    # size of messages_received_by_recip (dict2) is larger than size of num_msgs_sent_by_sender (dict1).
    # need to union the keys of these two dictionaries. the top 5 senders will be
    # in dict1, so need to append the list with this dictionary first
    for key in dict1:
        # getting the union of the keys in both dictionaries, so will only append
        # if the key DOESN'T already exist in the list
        if key not in union_of_all_keys:
            union_of_all_keys.append(key)

    for key in dict2:
        if key not in union_of_all_keys:
            union_of_all_keys.append(key)

    return union_of_all_keys
