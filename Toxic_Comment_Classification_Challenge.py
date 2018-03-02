# Name: Toxic Comment Classification Challenge
# Hacker: Gu, Shijia
# Created on 02-12-2017

import re
import pandas as pd
import numpy as np
from nltk import pos_tag
from nltk.tokenize import casual_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, GRU, Dropout, Dense, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os
os.environ['OMP_NUM_THREADS'] = '4'

# Read data from files
print("Loading data from files ...", end=' ')
path_prefix = 'D:\data\kaggle\Toxic_Comment_Classification_Challenge\\'
train = pd.read_csv(path_prefix + 'train.csv')
test = pd.read_csv(path_prefix + 'test.csv')
print('Finished!')

# Prepare data
print("Pre-processing data for training and testing ...", end=' ')


def decontract(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"let\'s", "let us", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize(sentence):
    res = []
    lmtzr = WordNetLemmatizer()
    words = casual_tokenize(sentence, preserve_case=False, reduce_len=True, strip_handles=True)         # TweetTokenizer
    for word, pos in pos_tag(words):                                                                       # POS tagging
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lmtzr.lemmatize(word, pos=wordnet_pos))                                     # lemmatize based on tags
    return res


def pre_process(txt):
    # txt = txt.tolower()
    # txt = ' '.join(line.rstrip('\n') for line in txt)
    txt = re.sub(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]', 'url', txt)  # URL
    txt = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", "ip", txt)  # IP address
    txt = decontract(txt)

    stop_words = stopwords.words('english')

    words = lemmatize(txt)
    words = [w for w in words if w not in stop_words]
    words = [w for w in words if len(w) > 2 and not w.isdigit()]
    txt = " ".join(words)
    return txt


train_text = []
for t in train['comment_text'].fillna('foobar'):
    t = pre_process(t)
    train_text.append(t)
train_text = pd.Series(train_text).astype(str)

test_text = []
for t in test['comment_text'].fillna('foobar'):
    t = pre_process(t)
    test_text.append(t)
test_text = pd.Series(test_text).astype(str)

dict_size = 50000
max_len = 200

tokenizer = text.Tokenizer(num_words=dict_size)
tokenizer.fit_on_texts(list(train_text) + list(test_text))
train_seq = tokenizer.texts_to_sequences(train_text)
test_seq = tokenizer.texts_to_sequences(test_text)
train_X = sequence.pad_sequences(train_seq, maxlen=max_len)
train_Y = train[list(train)[2:]].values  # list to get column names
test_X = sequence.pad_sequences(test_seq, maxlen=max_len)

# Persistence
pd.DataFrame(train_X).to_csv(path_prefix + 'train_X.csv', header=False, index=False)
pd.DataFrame(test_X).to_csv(path_prefix + 'test_X.csv', header=False, index=False)
print('Finished!')

# The model
print("Building and Training a model ...", end=' ')
embed_size = 100
word_index = tokenizer.word_index
GLOVE_FILE = path_prefix + "glove.6B.100d.txt"


def get_coefs(wd, *arr): return wd, np.asarray(arr, dtype='float32')


glove_dict = dict(get_coefs(*o.strip().split()) for o in open(GLOVE_FILE, encoding='UTF-8'))
all_embed = np.stack(glove_dict.values())
embed_mean, embed_std = all_embed.mean(), all_embed.std()
embed_matrix = np.random.normal(embed_mean, embed_std, (dict_size, embed_size))

for word, i in word_index.items():
    if i >= dict_size:
        continue
    embedding_vector = glove_dict.get(word)
    if embedding_vector is not None:
        embed_matrix[i] = embedding_vector


model = Sequential()
model.add(Embedding(dict_size, embed_size, weights=[embed_matrix], trainable=True))
model.add(Bidirectional(GRU(max_len, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
model.add(Dropout(0.1))
model.add(Bidirectional(GRU(max_len, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.1))
model.add(Dense(6, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

weight_file = path_prefix + 'model_weights.hdf5'
checkpoint = ModelCheckpoint(filepath=weight_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=1)
model.fit(train_X, train_Y, batch_size=32, epochs=5, validation_split=0.1, callbacks=[checkpoint, early_stopping])
print('Finished!')

# Test the model
print("Predicting ...", end=' ')
test_Y = model.predict(test_X)
print('Finished!')
print("Generating submission file ...", end=' ')
sample_submission = pd.read_csv(path_prefix + 'sample_submission.csv')
sample_submission[list(train)[2:]] = test_Y
sample_submission.to_csv(path_prefix + "submission.csv", index=False)
print('Finished!')
