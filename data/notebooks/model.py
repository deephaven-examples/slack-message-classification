import pandas as pd
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import save_model
from keras.layers import Input, LSTM, Embedding, Dense, Dropout
from keras.layers import GlobalMaxPool1D
from keras.models import Model

train = pd.read_csv('train.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
comment = train["comment_text"]

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(comment))

tokenizer_json_string = tokenizer.to_json()
with open('tokenizer.json', 'w') as outfile:
    outfile.write(tokenizer_json_string)


tokenized = tokenizer.texts_to_sequences(comment)

maxlen = 200
X = pad_sequences(tokenized, maxlen=maxlen)
input = Input(shape=(maxlen,)) 
x = Embedding(max_features, 128)(input)
x = LSTM(60, return_sequences=True, name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)  # 6 classification labels
model = Model(inputs=input, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32
epochs = 50
model.fit(X, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

save_model(model, "model.h5")