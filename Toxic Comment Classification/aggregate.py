import sys, os, re, csv, codecs, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"]  # this will be a pandas series
list_sentences_test = test["comment_text"]

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
# this looks like some kind of a hash object and when printed it says this
# <keras_preprocessing.text.Tokenizer object at 0x000001A9097A66A0>

tokenizer.fit_on_texts(list(list_sentences_train))
# fit on texts will provide 4 attributes
# 1. word_counts, which is a dictionary of words and their counts
# and it look like this
# [('benetti', 1)...(''automakers, 1)

# 2. word_docs, which is a dictionary of words and how many documents each appeared in
# and it looks like this
# ({'rothaus':1, ......, 'automakers':1})

# 3. word_index, which is a dictionary of words and their uniquely assigned integers
# and it looks like this
# {'submissively':158547, ....., 'ciu':210337}

# 4. a count of the total number of documents that were used to fit
# in our case it is
# 159571

list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
# list_tokenized_train will look like
# note that each inner list is not the same length to another inner list. we will need to pad the inner list
# [[4, 11, 574, 49, 11, 24, 210, 6, 62, 201, 15, 1, 254, 2, 18, 1, 113, 395, 136, 89, 9, 7, 151, 34, 11],.... 
# [4, 7, 134, 59, 67, 6, 252, 7, 587, 63, 4, 29, 414, 24, 344, 145, 506, 39, 444, 3, 498, 811, 6, 18, 344, 1231, 121, 506, 363, 3, 1627, 2056, 88]]
# the length of this list is 159571

list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
# list_tokenized_test will look like
# note that each inner list is not the same length to another inner list. we will need to pad the inner list
# [[13, 1005, 228, 95, 81, 786, 1884, 82, 16253, 5, 27], [7, 1031, 264, 13, 536, 8, 238, 26, 162, 250, 1435], ....
# [2648, 31, 79, 954, 2, 989, 6099, 103, 11, 96, 41, 4623, 32, 3984, 2, 1325, 212, 437, 25, 1093, 31, 79, 86, 1, 79, 10, 1, 15516, 2518, 18, 204, 2462, 3417]]
# the length of this list is 153164

# plot the distribution of the length of each comment
# this is to make sure that when we do the padding, we set the right amount of maxlen
totalNumWords = [len(one_comment) for one_comment in list_tokenized_train]
plt.hist(totalNumWords, bins=np.arange(0,410,10))
plt.savefig("Distribution of each comment.png")

maxlen = 200
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
# X_t will look like the following
# interesting how the zeroes are padded from the inside
# [[    0     0     0 ...  4583  2273   985]
#  [    0     0     0 ...   589  8377   182]
#  [    0     0     0 ...     1   737   468]
#  ...
#  [    0     0     0 ...  3509 13675  4528]
#  [    0     0     0 ...   151    34    11]
#  [    0     0     0 ...  1627  2056    88]]


X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

# X_te will look like the following
# [[   0    0    0 ...  145  493   84]
#  [   0    0    0 ...   11    8 2826]
#  [   0    0    0 ...  109   15  355]
#  ...
#  [   0    0    0 ...   12 1652  358]
#  [   0    0    0 ... 9844 3506  355]
#  [   0    0    0 ...  100 5220    6]]

inp = Input(shape=(maxlen, ))
# this will just create a tensor
# if printed out, it would look like the following
# Tensor("input_1:0", shape=(?, 200), dtype=float32)

embed_size = 128
x = Embedding(max_features, embed_size)(inp)
# this will have a shape of (None, 200, 128)

x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
# this will have a shape of (None, 200, 60)
# first number in (None, 200, 60) is the batch size
# followed by the time step and the output size
# according to https://www.kaggle.com/sbongo/for-beginners-tackling-toxic-using-keras
# this is the unrolled version of the LSTM because there are 60 hidden layers
# I suppose the rolled version would just have a 1 instead of 60

x = GlobalMaxPool1D()(x)

x = Dropout(0.1)(x)

x = Dense(50, activation="relu")(x)

x = Dropout(0.1)(x)

x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)

model.compile(loss="binary_crossentropy", 
                    optimizer='adam',
                    metrics=['accuracy'])

batch_size = 32
epochs = 2
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

