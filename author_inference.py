
import tensorflow as tf
import os
import re
import datetime
import numpy as np

from text_cnn import TextCNN


SAMPLE_LENGTH = 200


import gzip, pickle as pkl

with gzip.GzipFile("data/abridged_index2word.pkl.gz", "rb") as f:
    abridged_index2word = pkl.load(f)
with gzip.GzipFile("data/abridged_word2index.pkl.gz", "rb") as f:
    abridged_word2index = pkl.load(f)
with gzip.GzipFile("data/id_to_cat.pkl.gz", "rb") as f:
    id_to_cat = pkl.load(f)


cnn = TextCNN(
    sequence_length=SAMPLE_LENGTH,
    num_classes=3,
    vocab_size=len(abridged_index2word),
    embedding_size=300,
    filter_sizes=[3, 4, 5],
    num_filters=128)


sess = tf.InteractiveSession()



gensim_weights = [x for x in tf.global_variables() if "embedding" in str(x)]
other_weights = [x for x in tf.global_variables() if "embedding" not in str(x)]



saver = tf.train.Saver(other_weights)
saver_embedding = tf.train.Saver(gensim_weights)



saver_embedding.restore(sess,
                        "output/gensim_weights")



saver.restore(sess,
              'runs/checkpoints/model-100')



import pickle as pkl, gzip, pandas as pd
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+|\'')



def preprocess(text):
    tokens = tokenizer.tokenize(text)
    preprocessed = []
    for token in tokens:
        if len(token) > 2 and token == token[0].upper() + token[1:].lower():
            token = "_PROPERNAME_"
        preprocessed.append(token.lower())
    return preprocessed




def preprocess_to_token_id(text):
    tokens = preprocess(text)
    ids = []
    for t in tokens:
        if t in abridged_word2index:
            idx = abridged_word2index[t]
        else:  # _UNKNOWN_
            idx = 0
        ids.append(idx)
    return ids



# In[49]:


def execute_cnn(text_preprocessed_gensim):
    input_data = []
    if len(text_preprocessed_gensim) < SAMPLE_LENGTH:
        input_data.append([])
        for i in range(100):
            input_data[0].extend(text_preprocessed_gensim)
            if len(input_data[0]) > SAMPLE_LENGTH:
                input_data[0] = input_data[0][:SAMPLE_LENGTH]
                break
    else:
        for i in range(0, len(text_preprocessed_gensim) - SAMPLE_LENGTH - 1, 100):
            input_data.append(text_preprocessed_gensim[i:i + SAMPLE_LENGTH])

    result = sess.run(cnn.scores, {cnn.input_x: input_data, cnn.dropout_keep_prob: 1.0})

    proba_result = []
    for result_segment in result:
        probas = [np.exp(x) for x in result_segment]
        probas /= sum(probas)
        proba_result.append(probas)

    return np.asarray(proba_result)



def predict(text):
    token_ids = preprocess_to_token_id(text)

    probabilities = execute_cnn(token_ids)

    return list(zip(id_to_cat, probabilities[0]))