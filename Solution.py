import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import transformers
transformers.logging.set_verbosity_error()
import torch
from progress.bar import IncrementalBar
from joblib import dump, load


def relativize_dates_(df):
    '''
    :param df: a pandas dataframe
    :return: no return - operation works in place
    '''
    unique_matters = np.unique(df['Matter'])
    first_date_dict, last_date_dict, relative_dates, relative_fractions = {}, {}, [], []
    for matter in unique_matters:
        dates_sorted = np.sort(df[df['Matter'] == matter]['Date'])
        first_date_dict[matter] = datetime.strptime(dates_sorted[0], '%Y-%m-%d')
        last_date_dict[matter] = datetime.strptime(dates_sorted[-1], '%Y-%m-%d')
    bar = IncrementalBar('Relativizing', max=df.shape[0], suffix='%(percent)d%%')
    for row in range(df.shape[0]):
        row_date = datetime.strptime(df['Date'].iloc[row], '%Y-%m-%d')
        relative_dates.append(row_date - first_date_dict[df['Matter'].iloc[row]])
        relative_fractions.append(relative_dates[row] /
                                  (last_date_dict[df['Matter'].iloc[row]] - first_date_dict[df['Matter'].iloc[row]])
                                  )
        bar.next()
    print()
    df['Date_Relative'] = relative_dates
    df['Timeline_Completion'] = relative_fractions
    return True


def analyze_clusters(data, codeset):
    '''
    :param data: a 2D list or nparray with columns Cluster, Label, Score
    :param labels: target groups to assign to clusters
    :return: dictionary of the form {cluster_label: target_group}
    '''
    labels = codeset['Description'].to_numpy()
    codes = codeset['Code'].to_numpy()

    df = pd.DataFrame(data, columns=['Cluster', 'Label', 'Score'])
    redux = df.groupby(by=['Cluster', 'Label'], as_index=False).sum().sort_values('Score', ascending=False)
    searching, found_labels, found_clusters, best_clusters = True, [], [], {}
    while searching:
        best_clusters[redux['Cluster'].iloc[0]] = codes[labels == redux['Label'].iloc[0]][0]
        found_clusters.append(redux['Cluster'].iloc[0])
        found_labels.append(redux['Label'].iloc[0])
        redux = redux[~redux['Label'].isin(found_labels)]
        redux = redux[~redux['Cluster'].isin(found_clusters)]
        if len(found_labels) == len(labels):
            searching = False
    return best_clusters

data = pd.read_csv('data/ailbiz_challenge_data.csv')[:40]
codes = pd.read_csv('data/ailbiz_challenge_codeset.csv')

relativize_dates_(data)

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model = transformers.BertModel.from_pretrained('bert-base-uncased')
raw_features = []
bar = IncrementalBar('Embedding', max=data.shape[0], suffix='%(percent)d%%')
for ind, line in enumerate(data['Narrative']):
    embedded = model(
        torch.tensor(
            tokenizer.encode(
                re.compile("[^a-zA-Z]").sub('', line).lower()
            )
        ).unsqueeze(0))[0].detach().numpy()
    raw_features.append(np.append(np.average(embedded, axis=1).flatten(), data['Timeline_Completion'][ind]))
    bar.next()
print()

scaler = StandardScaler()
features = scaler.fit_transform(raw_features)
model = KMeans(n_clusters=len(codes))
labels = model.fit_predict(features)
raw_solutions = np.concatenate((data['UID'].to_numpy().reshape(-1, 1), labels.reshape(-1, 1)), axis=1)

classifier = transformers.pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

classifications = []
for i in range(len(codes)):
    matchind = np.where([labels == i])[1]
    occupation = len(matchind)
    size = np.minimum(66, occupation)
    cutind = np.random.choice(a=occupation, size=size, replace=False)
    testind = matchind[cutind]
    test_set = data['Narrative'][testind]
    bar = IncrementalBar(f'Classifying group {(i + 1):02d}/{len(codes)}', max=size, suffix='%(percent)d%%')
    for sequence in test_set:
        result = classifier(sequence, codes['Description'])
        for ind, guess in enumerate(result['labels']):
            score = result['scores'][ind]
            classifications.append([i, guess, score])
        bar.next()
    print()

dump(classifications, 'cluster_labels.joblib')
dump(raw_solutions, 'raw_solutions.joblib')
classifications = load('cluster_labels.joblib')
raw_solutions = load('raw_solutions.joblib')

cluster_map = analyze_clusters(classifications, codes)

solutions = pd.DataFrame([[raw_solutions[idx][0], cluster_map[raw_solutions[idx][1]]]
                          for idx in range(raw_solutions.shape[0])], columns=['UID', 'Prediction_Track1'])

solutions.to_csv('predictions_track1.zip', index=False,
                 compression={'method': 'zip', 'archive_name': 'predictions_track1.csv'})
