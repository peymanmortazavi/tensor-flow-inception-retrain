import os
from collections import defaultdict


labels = open(os.environ.get('MODEL_LABELS'), 'rb').read().decode('utf-8').split('\n')
threshold = float(os.environ.get('threshold'))
print('LABELS: {}'.format(', '.join(labels)))
print('CONF THRESHOLD: {}'.format(threshold))

false_positives = set()
count_map = defaultdict(int)
recall_map = defaultdict(int)
precision_map = defaultdict(int)
with open(os.environ.get('RAW_FILE'), 'rb') as reader:
    for line in reader.read().decode('utf-8').split('\n'):
        if not line:
            continue
        parts = line.split(',')
        correct = parts[1]
        prediction = parts[2].replace(' ', '_')
        confidence = parts[3]
        count_map[correct] += 1
        is_certain = float(confidence) >= threshold
        is_prediction = is_certain and prediction != 'none'
        if correct not in labels:
            # make sure we don't have a false positive
            if prediction != 'none' and is_prediction:
                false_positives.add(line)
                recall_map[prediction] += 1
        else:
            recall_map[prediction] += 1 if is_prediction else 0
            precision_map[prediction] += 1 if prediction == correct and is_prediction else 0
            if prediction != correct and is_prediction:
                false_positives.add(line)

    total_recall = 0
    total_precision = 0
    total_count = 0
    for label in labels:
        if label == 'none' or not label:
            continue
        if count_map[label] == 0:
            print('no tests were given for {}'.format(label))
            continue
        total_count += count_map[label]
        total_recall += recall_map[label]
        total_precision += precision_map[label]
        recall = recall_map[label] / float(count_map[label]) * 100.0
        try:
            precision = precision_map[label] / float(recall_map[label]) * 100.0
        except:
            precision = 0.00
        print('{} - Recall: {:.2f} ; Precision: {:.2f} ; Total Count: {}'.format(label, recall, precision, count_map[label]))

    print('\nTotal Recall: {} ; Total Precision: {}'.format(total_recall/float(total_count) * 100.0, total_precision/float(total_recall) * 100.0))
    print('\nFalse Positives:\n{}\n'.format('\n'.join(false_positives)))
