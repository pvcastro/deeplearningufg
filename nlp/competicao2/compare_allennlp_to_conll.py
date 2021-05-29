# coding: utf-8

from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.util import lazy_groups_of, import_module_and_submodules
from typing import List, Iterator
from allennlp.data import Instance
import logging, os, fire
from pathlib import Path
import torch

log = logging.getLogger('allennlp')


def get_instance_data(predictor, document_path) -> Iterator[Instance]:
    yield from predictor._dataset_reader.read(document_path)


def predict_instances(predictor, batch_data: List[Instance]) -> Iterator[str]:
    yield predictor.predict_batch_instance(batch_data)


def predict_batch(batch, predictor, count, out_file, raise_oom):
    try:
        for _, results in zip(batch, predict_instances(predictor, batch)):
            for idx, result in enumerate(results):
                count['count'] += 1
                real_sentence = batch[idx]
                real_tags = real_sentence.fields['tags'].labels
                words = result['words']
                predicted_labels = result['tags']
                for word_idx, (word, tag) in enumerate(zip(words, predicted_labels)):
                    out_file.write(' '.join([word, real_tags[word_idx], tag]) + '\n')
                out_file.write('\n')
                if count['count'] % 200 == 0:
                    log.info('Predicted %d sentences' % count['count'])

    except RuntimeError as e:
        if 'out of memory' in str(e) and not raise_oom:
            new_batch_size = int(len(batch) / 2)
            print('| WARNING: ran out of memory, retrying with batch size %d' % new_batch_size)
            for p in predictor._model.parameters():
                if p.grad is not None:
                    del p.grad  # free some memory
            torch.cuda.empty_cache()
            for sub_batch in lazy_groups_of(iter(batch), new_batch_size):
                if new_batch_size == 1:
                    # Chegou no tamanho mínimo do batch, se não der certo desiste
                    predict_batch(sub_batch, predictor, count, out_file, raise_oom=True)
                else:
                    # Tenta novamente até que o tamanho do batch seja suficiente
                    predict_batch(sub_batch, predictor, count, out_file, raise_oom=False)
        else:
            raise e


def evaluate(model_path, document_path, cuda_device=-1, batch_size=64, predictions_file='predictions',
             scores_file='scores'):
    if not model_path.startswith('https'):
        model_path = Path(model_path)

    # model_path = '/media/discoD/models/elmo/ner_elmo_harem_pt/'
    # model_path = '/media/discoD/models/elmo/ner_elmo_harem_no_validation/'
    # document_path = '/home/pedro/repositorios/entidades/dataset/harem/harem_test_selective_conll2003.txt'

    import_module_and_submodules('allennlp_datalawyer')
    import_module_and_submodules('allennlp_models')

    if not str(model_path).endswith('.tar.gz'):
        model_path = model_path / 'model.tar.gz'

    archive = load_archive(archive_file=model_path, cuda_device=cuda_device,
                           overrides='{model:{verbose_metrics:true},dataset_reader:{type:"conll2003"}}')
    predictor = Predictor.from_archive(archive)

    count = {'count': 0}

    predictions_file = predictions_file + '.txt'
    scores_file = scores_file + '.txt'

    with open(predictions_file, mode='w', encoding='utf8') as out_file:
        for batch in lazy_groups_of(get_instance_data(predictor, document_path), batch_size):
            predict_batch(batch, predictor, count, out_file, raise_oom=False)
    out_file.close()
    log.info('Finished predicting %d sentences' % count['count'])
    os.system("./%s < %s > %s" % ('conlleval.perl', predictions_file, scores_file))
    print(open(scores_file, mode='r', encoding='utf8').read())


if __name__ == '__main__': fire.Fire(evaluate)
