

"""
word2vec embeddings start with a line with the number of lines (tokens?) and 
the number of dimensions of the file. This allows gensim to allocate memory 
accordingly for querying the model. Larger dimensions mean larger memory is 
held captive. Accordingly, this line has to be inserted into the GloVe 
embeddings file.
"""

import gensim

# Demo: Loads the newly created glove_model.txt into gensim API.
model=gensim.models.KeyedVectors.load_word2vec_format('glove_pt_wiki.model', binary=True) #GloVe Model
#model=gensim.models.KeyedVectors.load_word2vec_format('glove_pt_wiki.txt') #GloVe Model

print(model.most_similar(positive=['mulher', 'rei'], negative=['homem']))
print(model.most_similar(positive=['mulher']))
print(model.most_similar(positive=['homem']))
print(model.most_similar(positive=['austrália']))
print(model.similarity('mulher', 'homem'))
