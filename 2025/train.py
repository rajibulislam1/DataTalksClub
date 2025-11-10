
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, roc_curve

from ucimlrepo import fetch_ucirepo, list_available_datasets

nursery = fetch_ucirepo(id=76) 


# Get full dataset (features + target together)
df = nursery.data.original

print(df.shape)
df


# ### EDA


df.info()


### Check missing values
df.isnull().sum()


sns.countplot(data=df, x='class', order=df['class'].value_counts().index)
plt.title('Class Distribution')
plt.xticks(rotation=45)
plt.show()


df['class'] = df['class'].replace({'recommend': 'very_recom'}) # Merge recommend and very_recom

df['class'].value_counts(normalize=True)

for col in df.columns:
    print(f"Column: {col}")
    print(f"Unique values ({df[col].nunique()}): {df[col].unique()}")
    print(df[col].value_counts())  # proportion of each category
    print("-"*40)

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


# Train /test/validation split
from sklearn.model_selection import train_test_split
df_full_train , df_test = train_test_split(df, test_size = 0.2, random_state = 11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state =11)

df_train = df_train.reset_index(drop =True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Separate features and target
df_train_f = df_train.drop('class', axis=1)
df_val_f = df_val.drop('class', axis=1)
df_test_f = df_test.drop('class', axis=1)



y_train = df_train['class']
y_val = df_val['class']
y_test = df_test['class']


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

print(le.classes_)   # shows mapping like ['not_recom', 'priority', 'spec_prior', 'very_recom']


# ### Decision Tree
dv = DictVectorizer(sparse = False)
X_train = dv.fit_transform(df_train_f.to_dict(orient='records'))
X_val = dv.transform(df_val_f.to_dict(orient='records'))
X_test = dv.transform(df_test_f.to_dict(orient='records'))

dt = DecisionTreeClassifier(random_state=11)
dt.fit(X_train, y_train)


val_dicts = df_val.to_dict(orient = 'records')
X_val = dv.transform(val_dicts)


y_pred = dt.predict_proba(X_val)
roc_auc = roc_auc_score(y_val, y_pred, multi_class='ovr', average='macro')

print("ROC-AUC score:", roc_auc)


### Save objects using Pickle
import pickle
with open('decision_tree_model.pkl', 'wb') as f:
    pickle.dump(dt, f)

# save feature encoder
with open('dict_vectorizer.pkl', 'wb') as f:
    pickle.dump(dv, f)

# save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# ### Load the model

import pickle

with open('decision_tree_model.pkl', 'rb') as f:
    dt_loaded = pickle.load(f)

with open('dict_vectorizer.pkl', 'rb') as f:
    dv_loaded = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le_loaded = pickle.load(f)

new_data = {
    'parents': 'usual',
    'has_nurs': 'proper',
    'form': 'complete',
    'children': '1',
    'housing': 'convenient',
    'finance': 'convenient',
    'social': 'nonprob',
    'health': 'recommended'
}


# Transform features into numeric vector
X_new = dv_loaded.transform([new_data])


# Predict class index
y_pred_idx = dt_loaded.predict(X_new)



# Convert numeric label back to original class
y_pred_class = le_loaded.inverse_transform(y_pred_idx)
print("Predicted class:", y_pred_class[0])


