import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics




def normalize(x):
        for column in x.columns:
            x[column] = pd.to_numeric(x[column], errors="coerce")

# Load the dataset using pandas
# Replace 'your_dataset.csv' with the actual name of your CSV file
dataset = pd.read_csv('datasets/autism_screening.csv')


dataset = dataset.replace({'yes':1, 'no':0, '?':'Others', 'others':'Others'})

print(dataset.head())

for column in dataset.columns:
    unique_values = dataset[column].unique()
    print(f"Unique values in '{column}' column: {unique_values}")


removal = ['age_desc', 'used_app_before', 'austim','Class/ASD']
features = dataset.drop(removal, axis=1)
target = dataset['Class/ASD']

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size = 0.3, random_state=10)

# As the data was highly imbalanced we will balance it by adding repetitive rows of minority class.
ros = RandomOverSampler(sampling_strategy='minority',random_state=0)
X, Y = ros.fit_resample(X_train,Y_train)
print(X)

X.shape, Y.shape

# Normalizing the features for stable and fast training.
scaler = StandardScaler()
normalize(X)
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

model =  MultinomialNB()
model.fit(X, Y)

print(f'{model} : ')
print('Training Accuracy : ', metrics.roc_auc_score(Y, model.predict(X)))
print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, model.predict(X_val)))

