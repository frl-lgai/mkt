import json
import random
import itertools

with open('/w/data/mkt/Feedback-2022-11-06.json', 'r') as f:
    examples = json.load(f)

print(f"Total number of Examples : {len(examples)}")

examples = [example for example in examples 
    if not any([sample['rank'] is None for sample in example['models']])]

print(f"Total number of Examples without rank = None : {len(examples)}")


comparisons = []
for example in examples:
    for sample1, sample2 in itertools.combinations(example['models'], 2):
        if sample1['rank'] and sample2['rank'] and sample1['rank'] < sample2['rank']:
            comparisons.append({
                'input': example['prompt'],
                'pos_model': sample1['model'],
                'pos_temp': sample1['temperature'],
                'pos_label': sample1['response'],
                'neg_model': sample2['model'],
                'neg_temp': sample2['temperature'],
                'neg_label': sample2['response'],
            })

print(f"Total number of comparisons : {len(comparisons)}")

random.shuffle(comparisons)

with open('/w/mkt/data/comparisons_train.jsonl', 'w') as f_train:
    with open('/w/mkt/data/comparisons_valid.jsonl', 'w') as f_valid:

        for i, comparison in enumerate(comparisons):
            if i < 2000:
                f_valid.write(json.dumps(comparison, ensure_ascii=False)+"\n")
            else:
                f_train.write(json.dumps(comparison, ensure_ascii=False)+"\n")

