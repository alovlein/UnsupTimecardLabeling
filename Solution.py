import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import transformers
transformers.logging.set_verbosity_error()
import torch
from concurrent import futures
from progress.bar import IncrementalBar
from progress.spinner import PixelSpinner
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


def embed_one(index, sequence, tokenizer, model):
    embedded = model(
        torch.tensor(
            tokenizer.encode(
                re.compile("[^a-zA-Z]").sub('', sequence).lower()
            )
        ).unsqueeze(0))[0].detach().numpy()
    return np.append(np.average(embedded, axis=1).flatten(), data['Timeline_Completion'][index])


def classifier(samples, targets, ids, target_descriptions):
    spin = PixelSpinner('Clustering ')
    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(cosine_similarity, samples, targets)
        while not future.done():
            executor.submit(spin.next())
        sim = future.result()
    best = np.argsort(sim)[:, -1]
    codes = np.array([target_descriptions[x] for x in best]).reshape(-1, 1)
    return np.concatenate((ids.to_numpy().reshape(-1, 1), codes), axis=1)


print('Loading data...')
codes = pd.read_csv('data/ailbiz_challenge_codeset.csv')
data = pd.read_csv('data/ailbiz_challenge_data.csv')
relativize_dates_(data)

hf_tokenizer = transformers.DebertaTokenizer.from_pretrained('microsoft/deberta-large')
hf_model = transformers.DebertaModel.from_pretrained('microsoft/deberta-large')
raw_features, embedded_code_descriptions = [], []
bar = IncrementalBar('Embedding Targets ', max=codes.shape[0], suffix='%(percent)d%%')
for ind, line in enumerate(codes['Description']):
    embedded_code_descriptions.append(embed_one(ind, line, hf_tokenizer, hf_model))
    bar.next()
print()
embedded_code_descriptions = np.array(embedded_code_descriptions)

bar = IncrementalBar('Embedding Features ', max=data.shape[0], suffix='%(percent)d%%')
for ind, line in enumerate(data['Narrative']):
    raw_features.append(embed_one(ind, line, hf_tokenizer, hf_model))
    bar.next()
print() 
raw_features = np.array(raw_features)
dump(raw_features, 'raw_features.joblib')


spin = PixelSpinner('Standardizing ')
scaler = StandardScaler()
with futures.ThreadPoolExecutor(max_workers=2) as executor:
    future = executor.submit(scaler.fit_transform, load('raw_features.joblib'))
    while not future.done():
        executor.submit(spin.next())
    features = future.result()
print()
sol = classifier(features, embedded_code_descriptions, data['UID'], codes['Code'])
solutions = pd.DataFrame(sol, columns=['UID', 'Prediction_Track1'])

solutions.to_csv('predictions_track1.zip', index=False,
                 compression={'method': 'zip', 'archive_name': 'predictions_track1.csv'})
