import pandas as pd
import numpy as np
import pickle, re
from os.path import exists
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

# Load CSV file
df = pd.read_csv('twcs.csv')
sample = df.sample(n=5000)

text_data = sample['text'].tolist()

text_data_processed = []
for text in text_data:
    text_no_handle= re.sub(r'@\w+', '', text)
    text_no_handle_or_url = re.sub(r'http\S+|www\S+', '',text_no_handle)
    text_data_processed.append(text_no_handle_or_url)

# print(f"num of samples: {len(text_data)}")

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text_data_processed)
total_words = len(tokenizer.word_index) + 1
print(f"total_words: {total_words}")
#
with open('nlg_tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Create input sequences
input_sequences = []
for line in text_data_processed:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # print(f"len(token_list): {len(token_list)}")
    # print(token_list)
    # print(f"line: {line}")
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad input sequences
max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
# for seq in input_sequences:
#     if seq[0] != 0:
#         for token_num in seq:
#             if token_num != 0:
#                 print(tokenizer.index_word[token_num])
print(f"input_sequences length: {len(input_sequences)}, {len(input_sequences[0])}")
# print(input_sequences)
# Create X and y
X, y = input_sequences[:,:-1],input_sequences[:,-1]
# print("X:")
# print(X)
# print(f"Y: {len(y)}")
# print(y)
# y = pd.get_dummies(y).values
y = to_categorical(y, num_classes=total_words)
# print("Y dummies:")
# print(y)

# Build RNN model
model = Sequential()
model.add(Embedding(total_words, 10, input_length=max_sequence_len-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))

# # Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# # Define early stopping
early_stop = EarlyStopping(monitor='loss', patience=5)
# print(f"X: {len(X)} , {len(X[0])}")
# print(X[0])
# print(f"Y: {len(y)} , {len(y[0])}")
# print(y[0])
model.summary()

# # Train model
model.fit(X, y, epochs=50, batch_size=32, callbacks=[early_stop])

model.save('nlg_trained_model.h5')

model.summary()
# # Generate new text
def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted_probs = model.predict(token_list, verbose=0)
        predicted_index = np.argmax(predicted_probs)
        output_word = tokenizer.index_word[predicted_index]
        seed_text += " " + output_word
    return seed_text

# # Example usage
seed_text = "The sky is"
generated_text = generate_text(seed_text, 10, model, max_sequence_len)
print("Generated Text: ", generated_text)
