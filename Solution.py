import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel, pipeline
import torch
from progress.bar import IncrementalBar
from joblib import dump


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


data = pd.read_csv('data/ailbiz_challenge_data.csv')
codes = pd.read_csv('data/ailbiz_challenge_codeset.csv')

relativize_dates_(data)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
raw_features = []
bar = IncrementalBar('Embedding', max=data.shape[0], suffix='%(percent)d%%')
for ind, line in enumerate(data['Narrative']):
    embedded = model(
        torch.tensor(
            tokenizer.encode(
                re.compile("[^a-zA-Z]").sub('', line).lower()
            )
        ).unsqueeze(0))[0]
    raw_features.append(np.append(np.average(embedded.detach().numpy(), axis=1).flatten(),
                                  data['Timeline_Completion'][ind]))
    bar.next()
print()

scaler = StandardScaler()
features = scaler.fit_transform(raw_features)
model = KMeans(n_clusters=len(codes))
labels = model.fit_predict(features)

classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
cluster_labels = []
bar = IncrementalBar('Classifying', max=data.shape[0], suffix='%(percent)d%%')
for ind, sequence in enumerate(data['Narrative']):
    suggestion = classifier(sequence, codes['Description'])
    cluster_labels.append([labels[ind], suggestion['labels'][0], suggestion['scores'][0]])
    bar.next()
print()
dump(cluster_labels)
