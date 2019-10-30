import pickle

with open('../Distance_Words/sentiments_embeddings/crawl-300d-2M-subword.vec.pickle', 'rb') as handle:
    b = pickle.load(handle)
