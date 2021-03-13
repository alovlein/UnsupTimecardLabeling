import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Simple Baseline System Implementation (Track 1)
print('Simple Baseline (Track 1)')
datafile = 'ailbiz_challenge_data.csv'
codefile = 'ailbiz_challenge_codeset.csv'
outputfile = 'predictions_track1.csv'

print('1. loading data...')
data = pd.read_csv(datafile)
code = pd.read_csv(codefile)

print('2. computing features...')
tfidfmodel = TfidfVectorizer()
tfidfmodel.fit_transform(data['Narrative'].values)
codemtx = tfidfmodel.transform(code['Description'].values)
datamtx = tfidfmodel.transform(data['Narrative'].values)

print('3. making predictions...') 
outputs = []
for k in range(datamtx.shape[0]):
    rank = cosine_similarity(datamtx[k,:],codemtx)[0].argsort()
    predictedcodes = {'UID': data.iloc[k]['UID'],\
                      'Prediction_Track1': code['Code'].values[rank[-1:]][0]}
    outputs.append(predictedcodes)

print('4. saving results...')
outputs = pd.DataFrame(outputs)[['UID','Prediction_Track1']]
outputs.to_csv(outputfile,index=False)
