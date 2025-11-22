
from ucimlrepo import fetch_ucirepo, list_available_datasets
# fetch dataset 
nursery = fetch_ucirepo(id=76) 


## Imporat relevant packages
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_val_score


# Get full dataset (features + target together)
df = nursery.data.original

print(df.shape)
df

# ### EDA


df.info()

# %%
df.shape

# %%
### Check missing values
df.isnull().sum()

# %%
## Unique values per column
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

# %%
df['class'].value_counts(normalize=True)

# %%
sns.countplot(data=df, x='class', order=df['class'].value_counts().index)
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.show()

# %%
df['class'] = df['class'].replace({'recommend': 'very_recom'}) # Merge recommend and very_recom

# %%
df['class'].value_counts(normalize=True)

# %%
for col in df.columns:
    print(f"Column: {col}")
    print(f"Unique values ({df[col].nunique()}): {df[col].unique()}")
    print(df[col].value_counts())  # proportion of each category
    print("-"*40)


# %%
cols = df.columns
n_cols = 3
n_rows = (len(cols) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))

for i, col in enumerate(cols):
    ax = axes[i // n_cols, i % n_cols]
    sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax)
    ax.set_title(f"{col}")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)

# Hide empty subplots if any
for j in range(i + 1, n_rows * n_cols):
    fig.delaxes(axes.flatten()[j])

plt.tight_layout()
plt.show()


# %%
# Train /test/validation split
from sklearn.model_selection import train_test_split
df_full_train , df_test = train_test_split(df, test_size = 0.2, random_state = 11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state =11)

# %%
df_train = df_train.reset_index(drop =True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# %%
len(df_train), len(df_val), len(df_test)

# %%
# Separate features 
df_train_f = df_train.drop('class', axis=1)
df_val_f = df_val.drop('class', axis=1)
df_test_f = df_test.drop('class', axis=1)

# %%
## Target Features
y_train = df_train['class']
y_val = df_val['class']
y_test = df_test['class']

# %%
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

print(le.classes_)   # shows mapping like ['not_recom', 'priority', 'spec_prior', 'very_recom']

# %% [markdown]
# ### Logistic Regression Model

# %%
dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(df_train_f.to_dict(orient='records'))
X_val = dv.transform(df_val_f.to_dict(orient='records'))
X_test = dv.transform(df_test_f.to_dict(orient='records'))

# %%
log = LogisticRegression(max_iter=1000)
log.fit(X_train, y_train)

# %%
val_dicts = df_val.to_dict(orient = 'records')
X_val = dv.transform(val_dicts)

# %%
y_pred = log.predict_proba(X_val)
roc_auc = roc_auc_score(y_val, y_pred, multi_class='ovr')


print("ROC-AUC score:", roc_auc)



#import pickle, joblib
# save the model using pickle
#joblib.dump(log, 'models/logistic_model.pkl')
#joblib.dump(dv, 'models/dv.pkl')

#print("Model and vectorizer saved successfully.")



output = f'logisticmodel.bin'
output

with open(output, 'wb') as f_out:
    pickle.dump((dv, log), f_out)

print(f'the model is saved to {output}')








