#%%
import os
import xlrd
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
sns.set_style("whitegrid")
import altair as alt
alt.renderers.enable('html')

# Code for hiding seaborn warnings
import warnings
warnings.filterwarnings("ignore")


df_path = os.getcwd()
df_path2 = df_path + '\\data.xlsx'
dfdata = pd.read_excel(df_path2)
columnsNames = ['File name', 'Content', 'Category']
df = pd.DataFrame(dfdata)
df.head()

#%%
# Number of entries in each category
bars = alt.Chart(df).mark_bar(size=50).encode( 
    x = alt.X("Category"),
    y = alt.Y("count():Q", axis=alt.Axis(title='Number of entries')),
    tooltip=[alt.Tooltip('count()', title='Number of entries'), 'Category'],
    color='Category'
)

text = bars.mark_text(
    align='center',
    baseline='bottom',
).encode(
    text='count()'
)

(bars + text).interactive().properties(
    height=300, 
    width=700,
    title = "Number of entries in each category",
)

#%%
# % of entries in each category
df['id'] = 1
df2 = pd.DataFrame(df.groupby('Category').count()['id']).reset_index()

bars = alt.Chart(df2).mark_bar(size=50).encode(
    x=alt.X('Category'),
    y=alt.Y('PercentOfTotal:Q', axis=alt.Axis(format='.0%', title='% of Entries')),
    color='Category'
).transform_window(
    TotalEntries='sum(id)',
    frame=[None, None]
).transform_calculate(
    PercentOfTotal="datum.id / datum.TotalEntries"
)

text = bars.mark_text(
    align='center',
    baseline='bottom',
    #dx=5  # Nudges text to right so it doesn't appear on top of the bar
).encode(
    text=alt.Text('PercentOfTotal:Q', format='.1%')
)

(bars + text).interactive().properties(
    height=300, 
    width=700,
    title = "% of data in each category",
)

#%%
# Entries length by category
df['Entries_length'] = df['Content'].str.len()
plt.figure(figsize=(12.8,6))
sns.distplot(df['Entries_length']).set_title('Entries length distribution');

#%%
df['Entries_length'].describe()

#%%
# remove from the 95% percentile onwards to better appreciate the histogram
quantile_95 = df['Entries_length'].quantile(0.95)
df_95 = df[df['Entries_length'] < quantile_95]
plt.figure(figsize=(12.8,6))
sns.distplot(df_95['Entries_length']).set_title('Entries length distribution');

#%%
# get the number of news entries with more than 10,000 characters
df_more10k = df[df['Entries_length'] > 10000]
print(len(df_more10k))

# see one
df_more10k['Content'].iloc[0]

#%%
# plot a boxplot
plt.figure(figsize=(12.8,6))
sns.boxplot(data=df, x='Category', y='Entries_length', width=.5);

#%%
# remove the larger documents for better comprehension
plt.figure(figsize=(12.8,6))
sns.boxplot(data=df_95, x='Category', y='Entries_length');

#%%
# save the dataset
with open('Entries_dataset.pickle', 'wb') as output:
    pickle.dump(df, output)

#%%