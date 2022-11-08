import os
from collections.abc import Iterable, Mapping
import torch
from datasets import load_dataset

def load(data_dir="/w/data/mkt/kobaco", split=None, tokenizer=None, num_proc=8, eos_token="[EOS]"):

    dataset = load_dataset('json', 
        split=split, 
        data_files={
            "train" : os.path.join(data_dir, "train.jsonl"),
            "valid" : os.path.join(data_dir, "valid.jsonl"),
        })

    dataset = dataset.shuffle(seed=42)

    def tokenizing(example):
        example['input'] = example['input'].strip()
        example['label'] = example['label'].strip()

        sequence = example['input'] + example['label'] + f"\n{eos_token}\n"

        num_tokens_input = len(tokenizer.encode(example['input']))
        num_tokens_label = len(tokenizer.encode(example['label'] + f"\n{eos_token}\n"))
        num_tokens_total = len(tokenizer.encode(sequence))

        example = tokenizer(sequence, max_length=256, truncation=True, padding=False)
        example['loss_mask'] = [0]*(num_tokens_total-num_tokens_label) + [1]*num_tokens_label

        return example

    if tokenizer:
        dataset = dataset.map(
            tokenizing,
            remove_columns=dataset.column_names,
            num_proc=num_proc,
            load_from_cache_file=True,
        )

    return dataset


def prepare_for_language_modeling(tokenized_dataset, block_size=1024, num_proc=8):

    def grouping(examples):
        # >> examples
        # { 'attention_mask': [ [1, 1, 1], [1, 1, 0, 0] ],
        #   'input_ids': [ [796, 569, 18354], [7496, 17740, 6711, 796] ] }

        # How about inserting a special token between sentences? ### works better
        examples = {k: sum(examples[k], []) for k in examples}
        # >> examples
        # { 'attention_mask': [1, 1, 1, 1, 1, 0, 0],
        #   'input_ids': [796, 569, 18354, 7496, 17740, 6711, 796] }

        examples = {
            key: list(split(sequence, block_size, drop_last=True)) 
            for key, sequence in examples.items()
        }

        examples["labels"] = [
            list(map( lambda x, y: x if y else -100, input_ids, loss_mask))
            for input_ids, loss_mask in zip(examples["input_ids"], examples["loss_mask"])
        ]

        # >> examples
        # { 'attention_mask': [ [1, ..(1024).., 0], [1, ..(1024).., 0], ... ],
        #   'input_ids': [ [796, ..(1024).., 569], [18354, ..(1024).., 17740], ... ],
        #   'labels': [ [796, ..(1024).., 569], [18354, ..(1024).., 17740], ... ] }

        #del examples["loss_mask"]

        return examples

    # >> dataset['train'][1]
    # { 'attention_mask': [1, ..(1024).., 0],
    #   'input_ids': [796, ..(1024).., 569],
    #   'labels': [796, ..(1024).., 569], }
    return tokenized_dataset.map(
        grouping,
        batched=True,
        remove_columns=['loss_mask'],
        num_proc=num_proc,
        load_from_cache_file=True,
    )


def load_feedback(data_dir="/w/mkt/data/kobaco", split=None, tokenizer=None, num_proc=8):

    dataset = load_dataset('json',
        split=split,
        data_files={
            'train': os.path.join(data_dir, 'comparisons_train.jsonl'),
            'valid': os.path.join(data_dir, 'comparisons_valid.jsonl'),
        }
    )

    dataset = dataset.shuffle(seed=42)

    def tokenizing(example):
        pos_example = f"{example['input'].strip()} {example['pos_label'].strip()}[EOS]\n"
        neg_example = f"{example['input'].strip()} {example['neg_label'].strip()}[EOS]\n"

        pos_sample = tokenizer(pos_example, max_length=256, truncation=True, padding=False)
        neg_sample = tokenizer(neg_example, max_length=256, truncation=True, padding=False)

        return {
            'input_ids': pos_sample['input_ids'],
            'attention_mask': pos_sample['attention_mask'],
            'input_ids_neg': neg_sample['input_ids'],
            'attention_mask_neg': neg_sample['attention_mask'],
        }

    dataset = dataset.map(
        tokenizing,
        #remove_columns=dataset.column_names,
        num_proc=num_proc,
        load_from_cache_file=True,
    )

    return dataset


def split(inputs, size, drop_last=False, shuffle=False):
    # The type of inputs is list, np.array, torch.tensor
    # or dictionary like {'input_ids', [233, 23, 345, 12], 'attention_mask': [1, 1, 1, 0]}

    if shuffle:
        inputs = shuffling(inputs)

    if isinstance(inputs, Mapping):
        the_first_value = next(iter(inputs.values()))
        length = len(the_first_value)
        slice_fn = lambda x, i: {key: x[key][i:i+size] for key in x}

    elif isinstance(inputs, Iterable):
        length = len(inputs)
        slice_fn = lambda x, i: x[i:i+size]

    else:
        raise Exception("Only dict-like Mapping and list-like Iterable are acceptable.")


    # dropping the last batch smaller than others so that each segment is the same length
    length = (length//size)*size if drop_last else length

    for i in range(0, length, size):
        yield slice_fn(inputs, i)


def shuffling(inputs): # only works on torch tensors

    if isinstance(inputs, Mapping): # dict of tensors
        batch_size = len(next(iter(inputs.values())))
        indexes = torch.randperm(batch_size)
        return {key: inputs[key][indexes] for key in inputs}

    elif torch.is_tensor(inputs):
        indexes = torch.randperm(len(inputs))
        return inputs[indexes]
    
    else:
        raise Exception("Only dict of tensors and a tensor are acceptable.")

