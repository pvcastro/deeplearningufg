from portuguese_wiki_corpus import PortugueseWikiCorpus
from synonym_generator import SynonymGenerator

synonym_generator = SynonymGenerator()
corpus = PortugueseWikiCorpus()
text = None
for document in corpus:
    text = document
    print(document)
    break

generated_text = synonym_generator.generate_text_from_synonyms(text='Olá, tudo bem?')
print(generated_text)