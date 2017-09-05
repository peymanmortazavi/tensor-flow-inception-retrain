import os
from collections import defaultdict


labels = open(os.environ.get('MODEL_LABELS'), 'rb').read().decode('utf-8').split('\n')
threshold = int(os.environ.get('threshold'))
print('LABELS: {}'.format(', '.join(labels)))
print('CONF THRESHOLD: {}'.format(threshold))

bad_prediction = set()
false_positives = set()
count_map = defaultdict(list)
recall_map = defaultdict(list)
precision_map = defaultdict(list)
with open(os.environ.get('RAW_FILE'), 'rb') as reader:
    for line in reader.read().decode('utf-8').split('\n'):
        parts = line.split(',')
        correct = parts[1]
        prediction = parts[2]
        confidence = parts[3]
        count_map[correct] += 1
        is_certain = int(confidence) >= threshold
        if correct not in labels:
            # make sure we don't have a false positive
            if prediction != 'none' and is_certain:
                false_positives.add(line)
        else:
            if prediction != correct and is_certain:
                bad_prediction.add(line)
                continue
            recall_map[correct] += 1 if is_certain else 0
            if prediction == correct and is_certain:
                precision_map[correct] += 1

    for label in labels:
        recall = recall_map[label] / float(count_map[label]) * 100.0
        precision = precision_map[label] / float(recall_map[labels]) * 100.0
        print('{} - Recall: {:.2f} ; Precision: {:.2f}'.format(label, recall, precision))

    print('Bad Prediction:\n{}\n\nFalse Positives:\n{}\n'.format('\n'.join(bad_prediction), '\n'.join(false_positives)))
