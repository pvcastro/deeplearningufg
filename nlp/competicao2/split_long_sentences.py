# coding: utf-8

import fire
from pathlib import Path

from split_datalawyer import SentenceSplit
from split_datalawyer.modules import ForceDropDuplicatedModule, ReplaceModule, ReplaceLongWordsModule, \
    ReplaceConcatenatedDotModule


def load_sentences(path, separator=' '):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    print('')
    print('Loading sentences from %s' % path)
    print('')
    for line in path.open(mode='r', encoding='utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split(sep=separator)
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences, [path] * len(sentences)


def join_punctuation(sentence, annotations, characters='.?!'):
    characters = set(characters)

    delete_indexes = []

    for index, token in enumerate(sentence):
        if token in characters:
            sentence[index - 1] += token
            delete_indexes.append(index)

    fixed_sentence = [token for index, token in enumerate(sentence) if index not in delete_indexes]
    fixed_annotations = [annotation for index, annotation in enumerate(annotations) if index not in delete_indexes]

    return fixed_sentence, fixed_annotations


def split_long_sentences(in_path, out_path, split_by_semicolon=False, limit=200, join_punct=True):
    sentence_split = SentenceSplit(
        modules=[ForceDropDuplicatedModule(), ReplaceModule(), ReplaceLongWordsModule(),
                 ReplaceConcatenatedDotModule()])

    if type(in_path) == str:
        in_path = Path(in_path)
    if type(out_path) == str:
        out_path = Path(out_path)
    out_file = out_path.open(mode='w', encoding='utf8')
    sentences, _ = load_sentences(in_path)
    count = 0
    for sentence in sentences:
        count += 1
        if len(sentence) > limit:
            sentence_array = [w_tuple[0] for w_tuple in sentence]
            annotations_array = [' '.join(w_tuple[1:]) for w_tuple in sentence]

            if join_punct:
                fixed_sentence, fixed_annotations = join_punctuation(sentence=sentence_array,
                                                                     annotations=annotations_array)
            else:
                fixed_sentence, fixed_annotations = sentence_array, annotations_array

            printed_sentence = ' '.join(fixed_sentence)
            token_count = 0
            segmentos = sentence_split.get_sentences(printed_sentence, split_by_semicolon=split_by_semicolon)
            for segmento in segmentos:
                for token in segmento.split(' '):
                    try:
                        out_file.write(token + ' ' + fixed_annotations[token_count] + '\n')
                        token_count += 1
                    except IndexError:
                        print(segmentos)
                        break
                out_file.write('\n')
        else:
            for token in sentence:
                out_file.write(' '.join(token) + '\n')
        out_file.write('\n')
    out_file.close()
    print('Rewrote %d sentences after splitting with length greater than %d' % (count, limit))


if __name__ == '__main__': fire.Fire(split_long_sentences)
