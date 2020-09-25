#%%
import pickle
import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np

#%%
# load the dataset
path_df = os.getcwd() + '\\Entries_dataset.pickle'
data = open(path_df, 'rb')
df = pickle.load(data)
data.close()
df.head()

#%%
# visualize one sample news content
df.loc[1]['Content']

# === 1. Text cleaning and preparation
# Special character cleaning
# \r and \n
df['Content_Parsed_1'] = df['Content'].str.replace("\r", " ")
df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("\n", " ")
df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace("    ", " ")
# " when quoting text
df['Content_Parsed_1'] = df['Content_Parsed_1'].str.replace('"', '')

# Upcase/downcase
# Lowercasing the text
df['Content_Parsed_2'] = df['Content_Parsed_1'].str.lower()
# Punctuation signs
punctuation_signs = list("?:!.,;")
df['Content_Parsed_3'] = df['Content_Parsed_2']

for punct_sign in punctuation_signs:
    df['Content_Parsed_3'] = df['Content_Parsed_3'].str.replace(punct_sign, '')

# Possessive pronouns
df['Content_Parsed_4'] = df['Content_Parsed_3'].str.replace("'s", "")

df['Content_Parsed_4'] = df['Content_Parsed_4'].fillna("")

# Stemming and Lemmatization
#%% 
# Downloading punkt and wordnet from NLTK
nltk.download('punkt')
print("------------------------------------------------------------")
nltk.download('wordnet')

# Saving the lemmatizer into an object
wordnet_lemmatizer = WordNetLemmatizer()

# In order to lemmatize, we have to iterate through every word
nrows = len(df)
lemmatized_text_list = []

for row in range(0, nrows):
    
    # Create an empty list containing lemmatized words
    lemmatized_list = []
    
    # Save the text and its words into an object
    text = df.loc[row]['Content_Parsed_4']

    # remove links
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    
    # remove numbers
    text = re.sub(r'[0-9]+', '', text)

    # remove tags in <>
    text = re.sub(r'<[^>]*>', '', text)

    # remove tags <style>...</style>
    text = re.sub(r'<style[^>]*>(.*?)<\/style>', '', text)

    # remove tags <script>...</script>
    text = re.sub(r'<script[^>]*>(.*?)<\/script>', '', text)

    text_words = text.split(" ")

    # Iterate through every word to lemmatize
    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
    # Join the list
    lemmatized_text = " ".join(lemmatized_list)
    
    # Append to the list containing the texts
    lemmatized_text_list.append(lemmatized_text)

df['Content_Parsed_5'] = lemmatized_text_list

# Stop words
#%% 
# Downloading the stop words list
nltk.download('stopwords')

#%% 
# Loading the stop words in english
stop_words = list(stopwords.words('english'))
print(stop_words)

# loop through all the stop words
df['Content_Parsed_6'] = df['Content_Parsed_5']

for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b"
    df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_stopword, '')

# Useless words
useless_words = ['wwwskillsharecom', 'staticskillsharecom', 'https', \
                'undefined', 'yahoocom', 'sender', 'time', 'sameas', \
                'message', 'http', 'schemaorg', 'uploads', 'logo', \
                'aolcom', 'video', 'assets', 'images', 'nundefined', \
                'skillshare', '252undefined', 'image', 'user']

for useless_word in useless_words:
    regex_uselessword = r"\b" + useless_word + r"\b"
    df['Content_Parsed_6'] = df['Content_Parsed_6'].str.replace(regex_uselessword, '')

#%%
# original entity article
df.loc[54]['Content']
#%%
# 1. Special character cleaning
df.loc[54]['Content_Parsed_1']
#%%
# 2. Upcase/downcase
df.loc[54]['Content_Parsed_2']
#%%
# 3. Punctuation signs
df.loc[54]['Content_Parsed_3']
#%%
# 4. Possessive pronouns
df.loc[54]['Content_Parsed_4']
#%%
# 5. Stemming and Lemmatization
df.loc[54]['Content_Parsed_5']
#%% 
# 6. Stop words
df.loc[54]['Content_Parsed_6']

#%% 
df.head(1)

#%%
# delete the intermediate columns
list_columns = ["File name", "Category", "Content", "Content_Parsed_6"]
df = df[list_columns]

df = df.rename(columns={'Content_Parsed_6': 'Content_Parsed'})

df.head()


# === 2. Label coding
# create a dictionary with the label codification
category_codes = {
    'info': 0,
    'support': 1,
    'unknown': 2
}

#%% 
# Category mapping
df['Category_Code'] = df['Category']
df = df.replace({'Category_Code':category_codes})
df.head()

#%% 
print(df.loc[150:170])

# === 3. Train - test split
# Cross Validation in the train set in order to tune the hyperparameters and then test performance on the unseen data of the test set
# choose a test set size of 15% of the full dataset
X_train, X_test, y_train, y_test = train_test_split(df['Content_Parsed'], 
                                                    df['Category_Code'], 
                                                    test_size=0.15, 
                                                    random_state=8)

# === 4. Text representation
# Parameter selection
ngram_range = (1,2)
min_df = 10
max_df = 1.
max_features = 300

#%%
# 
tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                        
features_train = tfidf.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf.transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)

#%% 
# use the Chi squared test in order to see what unigrams and bigrams are most correlated with each category
from sklearn.feature_selection import chi2
import numpy as np

for Product, category_id in sorted(category_codes.items()):
    features_chi2 = chi2(features_train, labels_train == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}' category:".format(Product))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-10:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))
    print("")

#%% 
# save the files
# X_train
with open('Pickles/X_train.pickle', 'wb') as output:
    pickle.dump(X_train, output)
    
# X_test    
with open('Pickles/X_test.pickle', 'wb') as output:
    pickle.dump(X_test, output)
    
# y_train
with open('Pickles/y_train.pickle', 'wb') as output:
    pickle.dump(y_train, output)
    
# y_test
with open('Pickles/y_test.pickle', 'wb') as output:
    pickle.dump(y_test, output)
    
# df
with open('Pickles/df.pickle', 'wb') as output:
    pickle.dump(df, output)
    
# features_train
with open('Pickles/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('Pickles/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('Pickles/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('Pickles/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)
    
# TF-IDF object
with open('Pickles/tfidf.pickle', 'wb') as output:
    pickle.dump(tfidf, output)

#%%
