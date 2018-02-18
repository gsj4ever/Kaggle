# Name: Toxic Comment Classification Challenge
# Hacker: Gu, Shijia
# Date: 02-12-2017

import pandas as pd
from nltk.corpus import stopwords
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, GRU, Dropout, Dense, GlobalAveragePooling1D
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
max_len = 100

col_names = train.columns.values.tolist()  # just simply type list(Dataframe)
classes = col_names[2:]
train_text = train['comment_text'].fillna('foobar').str.lower().str.replace(r'[\W0-9]+', ' ').str.strip().str.split()
train_refine = train_text.apply(lambda x: [item for item in x if item not in stop]).str.join(' ')
test_text = test['comment_text'].fillna("foobar").str.lower().str.replace(r'[\W0-9]+', ' ').str.strip().str.split()
test_refine = test_text.apply(lambda x: [item for item in x if item not in stop]).str.join(' ')
tokenizer = text.Tokenizer(num_words=dict_size)
tokenizer.fit_on_texts(train_refine)
train_seq = tokenizer.texts_to_sequences(train_refine)
test_seq = tokenizer.texts_to_sequences(test_refine)
train_pad = sequence.pad_sequences(train_seq, maxlen=max_len, padding='post', truncating='post')
# print(train_pad)
test_pad = sequence.pad_sequences(test_seq, maxlen=max_len, padding='post', truncating='post')
train_Y = train[classes]

# The model
print("Building and Training a model ...")
model = Sequential()
model.add(Embedding(dict_size, 128))
model.add(Bidirectional(GRU(max_len, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(GRU(max_len, return_sequences=True)))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.1))
model.add(Dense(6, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

weight_file = path_prefix + 'model_weights.hdf5'
checkpoint = ModelCheckpoint(filepath=weight_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=0)
model.fit(train_pad, train_Y, batch_size=32, epochs=5, validation_split=0.1, callbacks=[checkpoint, early_stopping])

# Test the model
print("Predicting test result and generate submission file ...")
test_Y = model.predict(test_pad)

sample_submission = pd.read_csv(path_prefix + 'sample_submission.csv')
sample_submission[classes] = test_Y
sample_submission.to_csv(path_prefix + "submission.csv", index=False)
