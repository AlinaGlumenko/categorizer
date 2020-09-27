#%%
import pickle
import pandas as pd

#%%
# need to get the performance metrics generated in every model
path_pickles = "/home/lnc/0. Latest News Classifier/04. Model Training/Models/"

list_pickles = [
    "df_models_gbc.pickle",
    "df_models_knnc.pickle",
    "df_models_lrc.pickle",
    "df_models_mnbc.pickle",
    "df_models_rfc.pickle",
    "df_models_svc.pickle"
]

df_summary = pd.DataFrame()

for pickle_ in list_pickles:
    
    path = path_pickles + pickle_
    
    with open(path, 'rb') as data:
        df = pickle.load(data)

    df_summary = df_summary.append(df)

df_summary = df_summary.reset_index().drop('index', axis=1)

#%%
# see the summary
df_summary

#%%
# And sort it by Test Set Accuracy
df_summary.sort_values('Test Set Accuracy', ascending=False)

#%%