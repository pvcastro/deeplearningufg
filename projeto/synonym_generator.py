from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from random import randint
import nltk.data
import gensim

#Based on https://gist.github.com/Ghost---Shadow/c361f2d6b4501f40648b
class SynonymGenerator():

    def __init__(self):
        self.word_vector_model = gensim.models.KeyedVectors.load_word2vec_format('glove_pt_wiki.model', binary=True)  # GloVe Model

    def generate_text_from_synonyms(self, text, lang='por'):
        output = ""

        # Get the list of words from the entire text
        #words = word_tokenize(text)
        words = gensim.utils.simple_preprocess(text)
        similars = self.word_vector_model.most_similar(positive=words)
        print(similars)

        # Identify the parts of speech
        tagged = nltk.pos_tag(words)

        for i in range(0,len(words)):
            print("word:", words[i])
            replacements = []

            # Only replace nouns with nouns, vowels with vowels etc.
            synonyms = self.get_synonyms(words[i], lang=lang)
            print("synonyms:", synonyms)

            [synonyms.remove(synonym) for synonym in synonyms if synonym not in self.word_vector_model.vocab]
            print("updated synonyms:", synonyms)

            similars = self.word_vector_model.most_similar(positive=synonyms)
            print("similars:", similars)

            for synonym in synonyms:
                # Do not attempt to replace proper nouns or determiners
                if tagged[i][1] == 'NNP' or tagged[i][1] == 'DT':
                    break
                else:
                    replacements.append(synonym)

            replacements = list(set(replacements))

            if len(replacements) > 1:
                # Choose a random replacement, different from the original word
                replacement = replacements[randint(0,len(replacements)-1)]
                while replacement == words[i]:
                    replacement = replacements[randint(0, len(replacements) - 1)]
                output = output + " " + replacement
            else:
                # If no replacement could be found, then just use the
                # original word
                output = output + " " + words[i]

        print(output)

    def get_synonyms(self, word, lang='por'):
        synonyms = set()
        synsets = wordnet.synsets(word, lang=lang)
        for synset in synsets:
            lemmas = synset.lemma_names(lang)
            [synonyms.add(lemma.lower()) for lemma in lemmas]
        if len(synonyms) == 0:
            synonyms = [word]
        return list(synonyms)

#generate_text_from_synonyms("Pete ate a large cake. Sam has a big mouth.")