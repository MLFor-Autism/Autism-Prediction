import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.utils import resample


class ML:
    def __init__(self, filePath):
        self.model = None
        self.data = pd.read_csv(filePath)
        if not filePath.endswith('train.csv'):
            self.preprocess()
            self.balance()
        self.features = self.data.drop("Class/ASD", axis=1)
        self.target = self.data["Class/ASD"]
        self.target = self.target.replace({'YES': 1, 'NO': 0})
        self.features = self.features.replace({True: 1, False: 0})
        self.xTrain = self.xTest = self.yTrain = self.yTest = None

    def trainWithSplitting(self):
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.features, self.target, test_size=0.3,
                                                                            random_state=42)
        self.xTrain = self.normalize(self.xTrain)
        self.model.fit(self.xTrain, self.yTrain)

    def crossValidation(self):
        scoring_metrics = {
            'accuracy': 'accuracy',
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'f1': make_scorer(f1_score)
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for metric_name, scoring_metric in scoring_metrics.items():
            scores = cross_val_score(self.model, self.normalize(self.features), self.normalize(self.target), cv=cv,
                                     scoring=scoring_metric)
            print(f"Cross-validated {metric_name}: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")

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
        x = x.apply(pd.to_numeric, errors='coerce')
        x = x.fillna(0)
        return x

    def preprocess(self):
        categorical_columns = ['ethnicity', 'jaundice', 'autism','age_desc', 'relation']
        self.data['Class/ASD'] = self.data['Class/ASD'].replace({"YES": 1, "NO": 0})
        for column in ['ethnicity', 'country_of_res','age_desc', 'relation']:
            success_rates = self.data.groupby(column)['Class/ASD'].mean()
            weights = 1 / success_rates.replace(0, np.inf)
            weights.replace(np.inf, 1e10, inplace=True)
            self.data[column + '_weight'] = self.data[column].map(weights)
        self.data = pd.get_dummies(self.data, columns=categorical_columns, prefix=categorical_columns)
        self.data['age'] = (self.data['age'] - min(self.data['age'])) / (max(self.data['age']) - min(self.data['age']))
        self.data['result'] = (self.data['result'] - min(self.data['result'])) / (
                max(self.data['result']) - min(self.data['result']))
        self.data = self.data.drop(['jaundice_no',"gender",'autism_no','used_app_before','country_of_res',], axis=1)

    def createModel(self, model):
        if model == "catBoost":
            self.model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, loss_function='Logloss')
        elif model == "Logistic Regression":
            self.model = LogisticRegression(random_state=42, solver='liblinear', max_iter=1500)
        elif model == "RandomForest":
            self.model = RandomForestClassifier(random_state=42, criterion="entropy", )
        elif model == "SVC":
            self.model = SVC(degree=10, probability=True, random_state=42)
        elif model == "Decision Tree":
            self.model = DecisionTreeClassifier(random_state=42, criterion="entropy")
        elif model == "KNN":
            self.model = KNeighborsClassifier(n_neighbors=5)
        elif model == "Naive Bayes":
            self.model = GaussianNB()
        else:
            print("\nUnknown model")
            exit(0)
        return self.model

    def plotTarget(self):
        sns.countplot(x='Class/ASD', data=self.data)
        plt.show()

    def balance(self):
        majority = self.data[self.data['Class/ASD'] == 0]
        minority = self.data[self.data['Class/ASD'] == 1]
        majorityDownsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
        self.data = pd.concat([majorityDownsampled, minority])


models = ["catBoost", "Logistic Regression", "RandomForest", "SVC", "Decision Tree", "KNN", "Naive Bayes"]
csvFiles = [os.path.join("datasets", filename) for filename in os.listdir("datasets")
            if filename.endswith(( 'train.csv', 'autism_screening.csv'))]

results = {}
for file in csvFiles:
    print(f"\nDataset file: {file}")
    testing = ML(file)
    datasetResults = {}

    for model in models:
        print(f"\nModel: {model}")
        testing.createModel(model)
        testing.trainWithSplitting()
        metrics = testing.showMetrics()
        datasetResults[model] = metrics
    print(testing.xTrain.info())
    print(testing.xTest.info())
    results[file] = datasetResults
print("\nSummary of Results:")
for dataset, metricsDict in results.items():
    print(f"\nDataset: {dataset}")
    for model, metrics in metricsDict.items():
        print(f"\nModel: {model}")
        print(metrics)
    bestModel = max(metricsDict, key=lambda k: metricsDict[k]["accuracy"])
    print(f"\nMost accurate algorithm for {dataset} is {bestModel} with accuracy {metricsDict[bestModel]['accuracy']}")
