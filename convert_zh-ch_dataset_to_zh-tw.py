from typing import Union
from datasets import load_dataset, Dataset
from opencc import OpenCC

cc = OpenCC('s2twp')

dataset_name = 'shibing624/nli-zh-all'

zh_cn_dataset = load_dataset(dataset_name)

def convert(text: Union[str, list[str]]) -> Union[str, list[str]]:
    return cc.convert(text) if isinstance(text, str) else [cc.convert(s) for s in text]

def convert_example(example: dict[str, list[str]]):
    example['text1'] = convert(example['text1'])
    example['text2'] = convert(example['text2'])
    return example

zh_tw_dataset = zh_cn_dataset['train'].map(convert_example, batched=True, batch_size=100)

zh_tw_dataset.push_to_hub('asadfgglie/nli-zh-tw-all')