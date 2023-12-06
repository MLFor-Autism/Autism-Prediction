import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import train_test_split
import os

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ML:
    def __init__(self, filePath):
        self.model = None
        self.data = pd.read_csv(filePath)
        self.features = self.data.drop("Class/ASD", axis=1)
        self.target = self.data["Class/ASD"]
        self.target = self.target.replace({'YES': 1, 'NO': 0, '?': 'Others', 'others': 'Others'})
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.features, self.target, test_size=0.3,
                                                                            random_state=42)

    def train(self):
        self.xTrain = self.normalize(self.xTrain)
        self.model.fit(self.xTrain, self.yTrain)

    def predict(self):
        self.xTest = self.normalize(self.xTest)
        return self.model.predict(self.xTest)

    def showMetrics(self):
        accuracy = accuracy_score(self.yTest, self.predict())
        precision = precision_score(self.yTest, self.predict(), zero_division=1)
        recall = recall_score(self.yTest, self.predict())
        f1 = f1_score(self.yTest, self.predict())
        cm = confusion_matrix(self.yTest, self.predict())
        report = classification_report(self.yTest, self.predict(), zero_division=1)

        print(f"Confusion Matrix: \n{cm}")
        print(f"Classification Report: \n{report}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")

        return {
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
        }

    def normalize(self, x):
        for column in x.columns:
            x[column] = pd.to_numeric(x[column], errors="coerce")
        x = pd.get_dummies(x, columns=x.select_dtypes(include=['object']).columns)
        x = x.apply(pd.to_numeric, errors='coerce')
        x = x.fillna(0)
        return x

    def createModel(self, model):
        if model == "catBoost":
            self.model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, loss_function='Logloss')
        elif model == "Logistic Regression":
            self.model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1500)
        elif model == "RandomForest":
            self.model = RandomForestClassifier(random_state=42)
        elif model == "SVC":
            self.model = SVC(degree=10, probability=True, random_state=42)
        elif model == "Decision Tree":
            self.model = DecisionTreeClassifier(random_state=42)
        elif model == "KNN":
            self.model = KNeighborsClassifier(n_neighbors=5)
        elif model == "Naive Bayes":
            self.model = GaussianNB()
        else:
            print("\nUnknown model")
            exit(0)
        return self.model


models = ["catBoost", "Logistic Regression", "RandomForest", "SVC", "Decision Tree", "KNN", "Naive Bayes"]
csvFiles = [os.path.join("datasets", filename) for filename in os.listdir("datasets")
            if filename.endswith(('csvAdam.csv', 'train.csv', 'autism_screening.csv'))]

results = {}
for file in csvFiles:
    print(f"\nDataset file: {file}")
    testing = ML(file)
    datasetResults = {}

    for model in models:
        print(f"\nModel: {model}")
        testing.createModel(model)
        testing.train()
        metrics = testing.showMetrics()
        datasetResults[model] = metrics

    results[file] = datasetResults

print("\nSummary of Results:")
for dataset, metricsDict in results.items():
    print(f"\nDataset: {dataset}")
    for model, metrics in metricsDict.items():
        print(f"\nModel: {model}")
        print(metrics)
    bestModel = max(metricsDict, key=lambda k: metricsDict[k]["accuracy"])
    print(f"\nMost accurate algorithm for {dataset} is {bestModel} with accuracy {metricsDict[bestModel]['accuracy']}")

