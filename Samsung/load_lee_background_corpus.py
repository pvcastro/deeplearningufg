import os
import operator
import warnings
import gensim
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

import nltk
#nltk.download('stopwords') # Let's make sure the 'stopword' package is downloaded & updated
#nltk.download('wordnet') # Let's also download wordnet, which will be used for lemmatization

from gensim.utils import lemmatize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Analysing our corpus.
#     - The first document talks about a bushfire that had occured in New South Wales.
#     - The second talks about conflict between India and Pakistan in Kashmir.
#     - The third talks about road accidents in the New South Wales area.
#     - The fourth one talks about Argentina's economic and political crisis during that time.
#     - The last one talks about the use of drugs by midwives in a Sydney hospital.
# Our final topic model should be giving us keywords which we can easily interpret and make a small summary out of. Without this the topic model cannot be of much practical use.

def get_lee_train_file():
    test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])
    print(test_data_dir)
    lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
    return lee_train_file


def load_texts(fname):
    """
    Function to build tokenized texts from file
    
    Parameters:
    ----------
    fname: File to be read
    
    Returns:
    -------
    yields preprocessed line
    """
    with open(fname) as f:
        for line in f:
            yield line


def process_file(file, keep_all=True):
    """
    Function to process texts. Following are the steps we take:
    
    1. Stopword Removal.
    2. Collocation detection.
    3. Lemmatization (not stem since stemming can reduce the interpretability).
    
    Parameters:
    ----------
    texts: Tokenized texts.
    
    Returns:
    -------
    texts: Pre-processed tokenized texts.
    """
    texts = list(load_texts(file))
    
    if keep_all is False:
        tokens = [word_tokenize(text) for text in texts]
        tagged = [nltk.pos_tag(doc) for doc in tokens]
        texts = [[tag[0] for tag in doc if tag[1].startswith('NN')] for doc in tagged]
        texts = [gensim.utils.simple_preprocess(' '.join(text), deacc=True, min_len=3) for text in texts]
    else:
        texts = [gensim.utils.simple_preprocess(text, deacc=True, min_len=3) for text in texts]
    
    bigram = gensim.models.Phrases(texts)  # for bigram collocation detection
    texts = [bigram[line] for line in texts]
    stops = set(stopwords.words('english'))  # nltk stopwords list
    
    texts = [[word for word in line if word not in stops] for line in texts]
    
    lemmatizer = WordNetLemmatizer()

    texts = [[lemmatizer.lemmatize(word) for word in line] for line in texts]

    return texts


def get_raw_texts():
    lee_train_file = get_lee_train_file()
    raw_texts = list(load_texts(lee_train_file))
    return raw_texts


def get_train_texts(keep_all=False):
    lee_train_file = get_lee_train_file()
    train_texts = process_file(lee_train_file, keep_all=keep_all)
    print('Returning', len(train_texts), 'training texts')
    return train_texts
