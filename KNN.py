import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier\

data = pd.read_csv("datasets/csvAdam.csv")
data.describe()
data.dropna()
data.drop(["Class/ASD"],axis=1)
for column in data.columns:
   encoder = LabelBinarizer()
   data[column] = encoder.fit_transform(data[column])
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(data, data["Class/ASD"])