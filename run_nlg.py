from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

def generate_text(model_file, tokenizer_file, seed_text, next_words=100):
    # Load the trained model
    model = load_model(model_file)

    # Load the tokenizer
    with open(tokenizer_file, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Tokenize the seed text
    seed_seq = tokenizer.texts_to_sequences([seed_text])[0]

    # Handle unknown words in the seed text
    seed_seq = [s if s < len(tokenizer.word_index) else tokenizer.word_index['<UNK>']
                for s in seed_seq]

    generated_text = seed_text

    # Generate text
    for _ in range(next_words):
        padded_seed = pad_sequences([seed_seq], maxlen=model.input_shape[1], padding='pre')
        predicted_word_index = np.argmax(model.predict(padded_seed), axis=-1)
        predicted_word = tokenizer.index_word[predicted_word_index[0]]
        generated_text += " " + predicted_word
        seed_seq.append(predicted_word_index[0])
        seed_seq = seed_seq[1:]

    return generated_text

# seed_text = "We're sorry, but"
# generated_text = generate_text('nlg_trained_model.h5', seed_text, 20)
# print("Generated Text: ", generated_text)
