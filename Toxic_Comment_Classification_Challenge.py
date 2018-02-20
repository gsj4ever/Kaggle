# Name: Toxic Comment Classification Challenge
# Hacker: Gu, Shijia
# Date: 02-12-2017

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, GRU, Dropout, Dense, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Read data from files
print("Loading data from files ...")
path_prefix = 'D:\data\kaggle\Toxic_Comment_Classification_Challenge\\'
train = pd.read_csv(path_prefix + 'train.csv')
test = pd.read_csv(path_prefix + 'test.csv')

#Prepare data
print("Pre-processing data for training and testing ...")
stop = stopwords.words('english')
dict_size = 50000
max_len = 200

col_names = train.columns.values.tolist()  # just simply type list(Dataframe)
classes = col_names[2:]
train_text = train['comment_text'].fillna('foobar').str.lower().str.replace(r'[\W0-9]+', ' ').str.strip().str.split()
train_refine = train_text.apply(lambda x: [item for item in x if item not in stop]).str.join(' ')
# print(train_refine)
test_text = test['comment_text'].fillna("foobar").str.lower().str.replace(r'[\W0-9]+', ' ').str.strip().str.split()
test_refine = test_text.apply(lambda x: [item for item in x if item not in stop]).str.join(' ')
tokenizer = text.Tokenizer(num_words=dict_size)
tokenizer.fit_on_texts(train_refine)
train_seq = tokenizer.texts_to_sequences(train_refine)
test_seq = tokenizer.texts_to_sequences(test_refine)
train_pad = sequence.pad_sequences(train_seq, maxlen=max_len)
# print(train_pad)
test_pad = sequence.pad_sequences(test_seq, maxlen=max_len)
train_Y = train[classes]

# The model
print("Building and Training a model ...")
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
model.add(Embedding(dict_size, embed_size, weights=[embed_matrix]))
model.add(Bidirectional(GRU(max_len, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
model.add(Dropout(0.1))
model.add(Bidirectional(GRU(max_len, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.1))
model.add(Dense(6, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

weight_file = path_prefix + 'model_weights.hdf5'
checkpoint = ModelCheckpoint(filepath=weight_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=1)
model.fit(train_pad, train_Y, batch_size=32, epochs=3, validation_split=0.1, callbacks=[checkpoint, early_stopping])

# Test the model
print("Predicting ...")
test_Y = model.predict(test_pad)

print("Generating submission file ...")
sample_submission = pd.read_csv(path_prefix + 'sample_submission.csv')
sample_submission[classes] = test_Y
sample_submission.to_csv(path_prefix + "submission.csv", index=False)
