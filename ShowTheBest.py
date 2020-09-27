#%%
import pickle


#%%
path_mnbc = os.getcwd() + "\\Models\\df_models_mnbc.pickle"
with open(path_mnbc, 'rb') as data:
    df_models_mnbc = pickle.load(data)

df_models_mnbc


#%%
path_knnc = os.getcwd() + "\\Models\\df_models_knnc.pickle"
with open(path_knnc, 'rb') as data:
    df_models_knnc = pickle.load(data)

df_models_knnc


#%%
path_lrc = os.getcwd() + "\\Models\\df_models_lrc.pickle"
with open(path_lrc, 'rb') as data:
    df_models_lrc = pickle.load(data)

df_models_lrc


#%%
path_svc = os.getcwd() + "\\Models\\df_models_svc.pickle"
with open(path_svc, 'rb') as data:
    df_models_svc = pickle.load(data)

df_models_svc

#%%
path_rfc = os.getcwd() + "\\Models\\df_models_rfc.pickle"
with open(path_rfc, 'rb') as data:
    df_models_rfc = pickle.load(data)

df_models_rfc


#%%