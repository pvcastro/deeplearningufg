#!/usr/bin/env python
# coding: utf-8


from allennlp.predictors.predictor import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.util import lazy_groups_of
from typing import List, Iterator
from allennlp.data import Instance
from pathlib import Path
import json


def load_ids(document_path):
    lines = open(document_path, mode='r').readlines()
    return [json.loads(line)['Id'] for line in lines]


def predict(document_path, model_path, out_path, batch_size=4):
    def get_instance_data(document_path) -> Iterator[Instance]:
        yield from predictor._dataset_reader.read(Path(document_path))

    def predict_instances(batch_data: List[Instance]) -> Iterator[str]:
        yield predictor.predict_batch_instance(batch_data)

    print('Loading model from %s' % model_path)
    archive = load_archive(archive_file=model_path, cuda_device=0)
    predictor = Predictor.from_archive(archive, 'text_classifier')
    ids = load_ids(document_path)

    count = 0

    with open(out_path, mode='w', encoding='utf-8') as out_file:
        print('Loading batches from %s for prediction' % document_path)
        out_file.write('Id,Category\n')
        idx = 0
        for batch in lazy_groups_of(get_instance_data(document_path), batch_size):
            for _, results in zip(batch, predict_instances(batch)):
                for result in results:
                    count += 1
                    predicted_label = result['label']
                    out_file.write(str(ids[idx]) + ',' + predicted_label + '\n')
                    idx += 1
                    if count % 100 == 0:
                        print('Predicted %d sentences' % count)
    out_file.close()
    print('Finished predicting %d sentences' % count)
    print('Results saved in %s' % Path(out_path).absolute())


# predict(document_path='test_submission.json', model_path='snli-roberta-full/model.tar.gz',
#         out_path='submissions_snli_roberta_full.csv')
#
# predict(document_path='test_submission.json', model_path='snli-roberta-parameters-full/model.tar.gz',
#         out_path='submissions_snli_roberta_parameters_full.csv')

predict(document_path='test_submission.json', model_path='snli-bert-full/model.tar.gz',
        out_path='submissions_snli_bert_full.csv')
