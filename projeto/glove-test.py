

"""
word2vec embeddings start with a line with the number of lines (tokens?) and 
the number of dimensions of the file. This allows gensim to allocate memory 
accordingly for querying the model. Larger dimensions mean larger memory is 
held captive. Accordingly, this line has to be inserted into the GloVe 
embeddings file.
"""

import gensim

# Demo: Loads the newly created glove_model.txt into gensim API.
model=gensim.models.KeyedVectors.load_word2vec_format('glove_pt_wiki.txt',binary=False) #GloVe Model
model.init_sims(replace=True)
model.save_word2vec_format('glove_pt_wiki.model', binary=True)

#print(model.most_similar(positive=['woman', 'king'], negative=['man']))
#print(model.most_similar(positive=['australia']))
#print(model.similarity('woman', 'man'))
