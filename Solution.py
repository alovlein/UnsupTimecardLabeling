import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import transformers
transformers.logging.set_verbosity_error()
import torch
from concurrent import futures
from progress.bar import IncrementalBar
from progress.spinner import PieSpinner
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


def embed_one(index, sequence, model):
    embedded = model(
        torch.tensor(
            tokenizer.encode(
                re.compile("[^a-zA-Z]").sub('', sequence).lower()
            )
        ).unsqueeze(0))[0].detach().numpy()
    return np.append(np.average(embedded, axis=1).flatten(), data['Timeline_Completion'][index])


def classifier(cluster_label, samples, targets, model=None, target_descriptions=None):
    assert (model is None and targets.shape[1] and target_descriptions is not None) \
           or (model is not None and not targets.shape[1]), 'Incompatible options selected'
    classifications = []
    if model:
        bar = IncrementalBar(f'Classifying group {(i + 1):02d}/{len(codes)}', max=size, suffix='%(percent)d%%')
        for sample in samples:
            result = model(sample, targets)
            for ind, guess in enumerate(result['labels']):
                score = result['scores'][ind]
                classifications.append([cluster_label, guess, score])
            bar.next()
        print()
    else:
        sim = cosine_similarity(samples, targets)
        col_0 = np.array([cluster_label] * sim.shape[0] * sim.shape[1]).reshape(-1, 1)
        col_1 = np.repeat(np.arange(targets.shape[0]).reshape(1, -1), sim.shape[0], axis=0).reshape(-1, 1)
        col_2 = sim.reshape(-1, 1)
        for ind in range(col_0.shape[0]):
            classifications.append([col_0[ind][0], target_descriptions[col_1[ind][0]], col_2[ind][0]])
        #classifications = np.concatenate((np.concatenate((col_0, col_1), axis=1, dtype=int), col_2), axis=1)
    return classifications


print('Loading data...')
codes = pd.read_csv('data/ailbiz_challenge_codeset.csv')
data = pd.read_csv('data/ailbiz_challenge_data.csv')
bart_model = transformers.pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

relativize_dates_(data)

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')
raw_features, embedded_code_descriptions = [], []
bar = IncrementalBar('Embedding', max=data.shape[0], suffix='%(percent)d%%')
for ind, line in enumerate(codes['Description']):
    embedded_code_descriptions.append(embed_one(ind, line, bert_model))
embedded_code_descriptions = np.array(embedded_code_descriptions)
'''
for ind, line in enumerate(data['Narrative']):
    raw_features.append(embed_one(ind, line, bert_model))
    bar.next()
print()

raw_features = np.array(raw_features)

dump(raw_features, 'raw_features.joblib')
'''
raw_features = load('raw_features.joblib')

scaler = StandardScaler()
features = scaler.fit_transform(raw_features)
model = KMeans(n_clusters=len(codes))
spin = PieSpinner('Clustering ')
with futures.ThreadPoolExecutor(max_workers=2) as executor:
    future = executor.submit(model.fit_predict, features)
    while not future.done():
        executor.submit(spin.next())
    labels = future.result()
print()
raw_solutions = np.concatenate((data['UID'].to_numpy().reshape(-1, 1), labels.reshape(-1, 1)), axis=1)

cluster_labels = []
for i in range(len(codes)):
    matchind = np.where([labels == i])[1]
    occupation = len(matchind)
    print(f'\nGroup {(i + 1):02d} has {occupation} timecards.')
    size = np.minimum(350, occupation)
    cutind = np.random.choice(a=occupation, size=size, replace=False)
    testind = matchind[cutind]
    test_set = data['Narrative'][testind]
    cluster_labels.extend(classifier(i, features[testind], embedded_code_descriptions,
                                     target_descriptions=codes['Description']))

dump(cluster_labels, 'cluster_labels.joblib')
dump(raw_solutions, 'raw_solutions.joblib')
cluster_labels = load('cluster_labels.joblib')
raw_solutions = load('raw_solutions.joblib')

cluster_map = analyze_clusters(cluster_labels, codes)

solutions = pd.DataFrame([[raw_solutions[idx][0], cluster_map[raw_solutions[idx][1]]]
                          for idx in range(raw_solutions.shape[0])], columns=['UID', 'Prediction_Track1'])

solutions.to_csv('predictions_track1.zip', index=False,
                 compression={'method': 'zip', 'archive_name': 'predictions_track1.csv'})
