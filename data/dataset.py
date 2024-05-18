import os
import numpy as np
import itertools
import collections
import torch
import json
import csv
import pandas as pd
from .example import Example
from .utils import nostdout
from underthesea import word_tokenize, text_normalize
from collections import defaultdict
# from pycocotools.coco import COCO as pyCOCO



class Dataset(object):
    def __init__(self, examples, fields):
        self.examples = examples
        self.fields = dict(fields)

    def collate_fn(self):
        def collate(batch):
            if len(self.fields) == 1:
                batch = [batch, ]
            else:
                batch = list(zip(*batch))

            tensors = []
            for field, data in zip(self.fields.values(), batch):
                tensor = field.process(data)
                if isinstance(tensor, collections.Sequence) and any(isinstance(t, torch.Tensor) for t in tensor):
                    tensors.extend(tensor)
                else:
                    tensors.append(tensor)

            if len(tensors) > 1:
                return tensors
            else:
                return tensors[0]

        return collate

    def __getitem__(self, i):
        example = self.examples[i]
        data = []
        for field_name, field in self.fields.items():
            data.append(field.preprocess(getattr(example, field_name)))

        if len(data) == 1:
            data = data[0]
        return data

    def __len__(self):
        return len(self.examples)

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)


class ValueDataset(Dataset):
    def __init__(self, examples, fields, dictionary):
        self.dictionary = dictionary
        super(ValueDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            value_batch_flattened = list(itertools.chain(*batch))
            value_tensors_flattened = super(ValueDataset, self).collate_fn()(value_batch_flattened)

            lengths = [0, ] + list(itertools.accumulate([len(x) for x in batch]))
            if isinstance(value_tensors_flattened, collections.Sequence) \
                    and any(isinstance(t, torch.Tensor) for t in value_tensors_flattened):
                value_tensors = [[vt[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])] for vt in value_tensors_flattened]
            else:
                value_tensors = [value_tensors_flattened[s:e] for (s, e) in zip(lengths[:-1], lengths[1:])]

            return value_tensors
        return collate

    def __getitem__(self, i):
        if i not in self.dictionary:
            raise IndexError

        values_data = []
        for idx in self.dictionary[i]:
            value_data = super(ValueDataset, self).__getitem__(idx)
            values_data.append(value_data)
        return values_data

    def __len__(self):
        return len(self.dictionary)


class DictionaryDataset(Dataset):
    def __init__(self, examples, fields, key_fields):
        if not isinstance(key_fields, (tuple, list)):
            key_fields = (key_fields,)
        for field in key_fields:
            assert (field in fields)

        dictionary = collections.defaultdict(list)
        key_fields = {k: fields[k] for k in key_fields}
        value_fields = {k: fields[k] for k in fields.keys() if k not in key_fields}
        key_examples = []
        key_dict = dict()
        value_examples = []

        for i, e in enumerate(examples):
            key_example = Example.fromdict({k: getattr(e, k) for k in key_fields})
            value_example = Example.fromdict({v: getattr(e, v) for v in value_fields})
            if key_example not in key_dict:
                key_dict[key_example] = len(key_examples)
                key_examples.append(key_example)

            value_examples.append(value_example)
            dictionary[key_dict[key_example]].append(i)

        self.key_dataset = Dataset(key_examples, key_fields)
        self.value_dataset = ValueDataset(value_examples, value_fields, dictionary)
        super(DictionaryDataset, self).__init__(examples, fields)

    def collate_fn(self):
        def collate(batch):
            key_batch, value_batch = list(zip(*batch))
            key_tensors = self.key_dataset.collate_fn()(key_batch)
            value_tensors = self.value_dataset.collate_fn()(value_batch)
            return key_tensors, value_tensors
        return collate

    def __getitem__(self, i):
        return self.key_dataset[i], self.value_dataset[i]

    def __len__(self):
        return len(self.key_dataset)


def unique(sequence):
    seen = set()
    if isinstance(sequence[0], list):
        return [x for x in sequence if not (tuple(x) in seen or seen.add(tuple(x)))]
    else:
        return [x for x in sequence if not (x in seen or seen.add(x))]


class PairedDataset(Dataset):
    def __init__(self, examples, fields):
        assert 'image' in fields
        assert 'question' in fields
        assert 'answer' in fields
        super(PairedDataset, self).__init__(examples, fields)
        self.image_field = self.fields['image']
        self.question_field = self.fields['question']
        self.answer_field = self.fields['answer']

    def question_set(self):
        question_list = [e.question for e in self.examples]
        question_list = unique(question_list)
        examples = [Example.fromdict({'question': q}) for q in question_list]
        dataset = Dataset(examples, {'question': self.question_field})
        return dataset

    def answer_set(self):
        answer_list = [e.answer for e in self.examples]
        answer_list = unique(answer_list)
        examples = [Example.fromdict({'answer': a}) for a in answer_list]
        dataset = Dataset(examples, {'answer': self.answer_field})
        return dataset

    def image_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='image')
        return dataset

    def question_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='question')
        return dataset

    def answer_dictionary(self, fields=None):
        if not fields:
            fields = self.fields
        dataset = DictionaryDataset(self.examples, fields, key_fields='answer')
        return dataset

    @property
    def splits(self):
        raise NotImplementedError


def convert_json_to_csv(input_file, folder_path,output_file):
    with open(input_file, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)

    rows = [["Anno ID", "Image ID", "Image Path", "Question", "Answer"]]

    for anno_id, annotation in data["annotations"].items():
        image_id = annotation["image_id"]
        question = annotation["question"]
        answer = annotation["answer"]
        image_name = data["images"].get(str(image_id), "")
        image_path = f"{folder_path}/{image_name}"
        rows.append([anno_id, image_id, image_path, question, answer])

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
def segment_text(text):
    result = word_tokenize(text)
    return result


class OpenViVQA(PairedDataset):
    def __init__(self,image_field, question_field, answer_field, img_root, ann_root, id_root=None):
        
        #path cac file trong dataset
        img_train_path = os.path.join(img_root,'training-images')
        json_train_path = os.path.join(ann_root,'training-annotations.json')
        img_test_path = os.path.join(img_root,'test-images')
        json_test_path = os.path.join(ann_root,'test-annotations.json')
        img_dev_path = os.path.join(img_root,'dev-images')
        json_dev_path = os.path.join(ann_root,'dev-annotations.json')
        train_csv = 'train.csv'
        test_csv = 'test.csv'
        dev_csv = 'dev.csv'
        
        #convert qua file csv tu json
        convert_json_to_csv(img_train_path,json_train_path,train_csv)
        convert_json_to_csv(img_test_path,json_test_path,test_csv)
        convert_json_to_csv(img_dev_path,json_dev_path,dev_csv)
        
        #gan cac file csv
        df_train = pd.read_csv('train.csv')
        df_test = pd.read_csv('test.csv')
        df_dev = pd.read_csv('dev.csv')
        
        
        #word_tokenize
        #ap dung tokenize cho dataset
        df_train['Question'] = [word_tokenize(text_normalize(x), format='text') for x in df_train['Question']]
        df_train['Answer'] = [word_tokenize(text_normalize(str(x)), format='text') for x in df_train['Answer']]

        df_dev['Question'] = [word_tokenize(text_normalize(x), format='text') for x in df_dev['Question']]
        df_dev['Answer'] = [word_tokenize(text_normalize(str(x)), format='text') for x in df_dev['Answer']]
        
        #example data
        train_examples = self.create_examples(df_train)
        dev_examples = self.create_examples(df_dev)
        test_examples = self.create_examples(df_test)
        
        examples = train_examples + dev_examples + test_examples
        
        fields = {'image': image_field, 'question': question_field, 'answer': answer_field}
        super(OpenViVQA, self).__init__(examples, fields)
        
        # return df_train,df_dev,df_test
    
    
    @property
    def splits(self):
        train_split = PairedDataset(self.train_examples, self.fields)
        val_split = PairedDataset(self.val_examples, self.fields)
        test_split = PairedDataset(self.test_examples, self.fields)
        return train_split, val_split, test_split

    @classmethod
    def create_examples(cls, df):
        examples = []
        for img, question, answer in zip(df['Image Path'], df['Question'], df['Answer']):
            example = Example.fromdict({
                'image': img,
                'question': question,
                'answer': answer
            })
            examples.append(example)
        return examples
        # roots = {}
        # roots['train'] = {
        #     'img': os.path.join(img_root, 'train2014'),
        #     'ann': os.path.join(ann_root, 'vlsp2023_train_data.json')
        # }
        # roots['val'] = {
        #     'img': os.path.join(img_root, 'val2014'),
        #     'ann': os.path.join(ann_root, 'vlsp2023_dev_data.json')
        # }
        # roots['test'] = {
        #     'img': os.path.join(img_root, 'test2014'),
        #     'ann': os.path.join(ann_root, 'vlsp2023_test_data.json')
        # }

        # if id_root is not None:
        #     ids = {}
        #     ids['train'] = np.load(os.path.join(id_root, 'vivqa_train_ids.npy'))
        #     ids['val'] = np.load(os.path.join(id_root, 'vivqa_dev_ids.npy'))
        #     ids['test'] = np.load(os.path.join(id_root, 'vivqa_test_ids.npy'))
        # else:
        #     ids = None

        # self.image_field = image_field
        # self.question_field = question_field
        # self.answer_field = answer_field
        # self.train_examples, self.val_examples, self.test_examples = self.get_samples(roots, ids)
        # examples = self.train_examples + self.val_examples + self.test_examples
        # super(OpenViVQA, self).__init__(examples, {'image': image_field, 'question': question_field, 'answer': answer_field})

    # @property
    # def splits(self):
    #     train_split = PairedDataset(self.train_examples, self.fields)
    #     val_split = PairedDataset(self.val_examples, self.fields)
    #     test_split = PairedDataset(self.test_examples, self.fields)
    #     return train_split, val_split, test_split

    # @classmethod
    # def get_samples(cls, roots, ids_dataset=None):
    #     train_samples = []
    #     val_samples = []
    #     test_samples = []

    #     for split in ['train', 'val', 'test']:
    #         if isinstance(roots[split]['ann'], tuple):
    #             vivqa_datasets = (cls.load_vivqa(roots[split]['ann'][0]), cls.load_vivqa(roots[split]['ann'][1]))
    #             root = roots[split]['img']
    #         else:
    #             vivqa_datasets = (cls.load_vivqa(roots[split]['ann']),)
    #             root = (roots[split]['img'],)

    #         if ids_dataset is None:
    #             ids = list(vivqa_datasets[0].keys())
    #         else:
    #             ids = ids_dataset[split]

    #         if isinstance(ids, tuple):
    #             bp = len(ids[0])
    #             ids = list(ids[0]) + list(ids[1])
    #         else:
    #             bp = len(ids)

    #         for index in range(len(ids)):
    #             if index < bp:
    #                 vivqa = vivqa_datasets[0]
    #                 img_root = root[0]
    #             else:
    #                 vivqa = vivqa_datasets[1]
    #                 img_root = root[1]

    #             ann_id = str(ids[index])
    #             question = vivqa[ann_id]['question']
    #             answer = vivqa[ann_id]['answer']
    #             img_id = vivqa[ann_id]['image_id']
    #             filename = f"{img_id}.jpg"

    #             example = Example.fromdict({'image': os.path.join(img_root, filename), 'text': f"Q: {question} A: {answer}"})

    #             if split == 'train':
    #                 train_samples.append(example)
    #             elif split == 'val':
    #                 val_samples.append(example)
    #             elif split == 'test':
    #                 test_samples.append(example)
