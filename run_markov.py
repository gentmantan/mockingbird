import markovify, pickle

with open('markov_trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

def generate_text(seed_text):
    sentence = None
    for j in range(100):
        sentence = model.make_sentence(tries=10000, max_overlap_ratio=0.7)
        if sentence is not None and seed_text.lower() in sentence.lower():
            break
    if sentence is not None:
        return sentence

    return print(model.make_short_sentence(280))
