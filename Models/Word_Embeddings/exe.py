import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
import json #https://stackoverflow.com/questions/7100125/storing-python-dictionaries
import pickle #https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict
import os






def load_model(name): #(0,vec),(1,gloove),(2,bin)

    format_option = 0
    if name[-1] == 't':
        format_option = 1
    elif name[-1] == 'n':
        format_option = 2
    print("Loading " + name )
    model_name = "./models/" + name
    if format_option == 0:
        print("word-2-vec")
        model = KeyedVectors.load_word2vec_format(model_name)
    elif format_option == 1:
        print("gloove")
        tmp_file = get_tmpfile("tmp")
        _ = glove2word2vec(model_name, tmp_file)
        model = KeyedVectors.load_word2vec_format(tmp_file)
    else:
        print("word-2-vec binary")
        model = KeyedVectors.load_word2vec_format(model_name, binary=True)
    print("Model " + name + " Loaded")
    return model


def generate_model_embeddings(model, sentiments):
    model_embeddings = dict()
    for i,s in enumerate(sentiments):
        printPercentage(i,len(sentiments),10)
        top_similar = model.most_similar(positive=s,topn=50)
        sentiment_embeddings = dict()
        sentiment_embeddings["vectors"] = dict()
        sentiment_embeddings["key_list"] = top_similar
        sentiment_embeddings["vectors"][s] = model[s]
        for word,_ in top_similar:
            sentiment_embeddings["vectors"][word] = model[word]
        model_embeddings[s] = sentiment_embeddings
    if(len(model_embeddings.keys())!=len(sentiments)):
        print("erro no tamanho do embedding gerado")
    return model_embeddings



def write_embeddings_to_file(filename, model_embeddings):
    with open("./sentiments_embeddings/"+filename+'.pickle', 'wb') as handle:
        pickle.dump(model_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)


def printPercentage(i,size,steps):
    if i%(size//steps) == 0:
        print(str(round((i/size)*100,2)) + "%")
    if(i==size-1):
        print("finished")




files = os.listdir("models")
models_done = os.listdir("sentiments_embeddings")
sentiments = np.load("./sentiments_list.npy")



for file_name in files:
    try:
        for i in models_done:
            if file_name == i[0:-7]:
                print("Arquivo ja existe")
                raise

        model = load_model(file_name)
        emb = generate_model_embeddings(model, sentiments)
        write_embeddings_to_file(file_name,emb)
    except:
        print("#########################################")
        print("ERRO TENTANDO NO ARQUIVO " + file_name)
        print("#########################################")

print("End of script")






