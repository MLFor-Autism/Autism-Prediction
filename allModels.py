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
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, make_scorer, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.utils import resample


class ML:
    def __init__(self, filePath):
        self.datasetName = filePath[9:-8]
        self.model = None
        self.data = pd.read_csv(filePath)
        if filePath.endswith('Generic_ASD.csv'):
            self.data['ASD'] = self.data['ASD'].replace({"YES": 1, "NO": 0})
        else:
            self.preprocess()
        self.balance()
        self.features = self.data.drop("ASD", axis=1)
        self.target = self.data["ASD"]
        self.target = self.target.replace({'YES': 1, 'NO': 0})
        self.features = self.features.replace({True: 1, False: 0})
        self.xTrain = self.xTest = self.yTrain = self.yTest = None

    def trainWithSplitting(self):
        print('\nTrain-Test Spliting')
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.features, self.target, test_size=0.3,
                                                                            random_state=42)
        self.xTrain = self.normalize(self.xTrain)
        self.model.fit(self.xTrain, self.yTrain)

    def crossValidation(self):
        print("\nCross Validation")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        prediction = cross_val_predict(self.model, self.normalize(self.features), self.normalize(self.target), cv=cv)
        report = classification_report(self.normalize(self.target), prediction, output_dict=True)
        cm = confusion_matrix(self.normalize(self.target), prediction)
        print(f'Confusion Matrix: \n{cm}')
        print(f"Classification Report: \n{classification_report(self.normalize(self.target), prediction)}")
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']
        accuracy = report['accuracy']
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
        return {
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
        }

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
        self.plotConfusionMatrix(cm)
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
        categorical_columns = ['Ethnicity', 'Jaundice', 'testTaker', 'ASDInFamily']
        self.data['ASD'] = self.data['ASD'].replace({"YES": 1, "NO": 0})
        for column in ['Ethnicity', 'testTaker']:
            success_rates = self.data.groupby(column)['ASD'].mean()
            weights = 1 / success_rates.replace(0, np.inf)
            weights.replace(np.inf, 1e10, inplace=True)
            self.data[column + '_weight'] = self.data[column].map(weights)
        self.data = pd.get_dummies(self.data, columns=categorical_columns, prefix=categorical_columns)
        self.data['Age'] = (self.data['Age'] - min(self.data['Age'])) / (max(self.data['Age']) - min(self.data['Age']))
        if 'result' in self.data:
            self.data['result'] = (self.data['result'] - min(self.data['result'])) / (
                    max(self.data['result']) - min(self.data['result']))
        self.data = self.data.drop(['Jaundice_No', "Gender", 'ASDInFamily_No'], axis=1)

    def balance(self):
        majority = self.data[self.data['ASD'] == 0]
        minority = self.data[self.data['ASD'] == 1]
        majorityDownsampled = resample(majority, replace=True, n_samples=len(minority), random_state=42)
        self.data = pd.concat([majorityDownsampled, minority])

    def createModel(self, model):
        if model == "catBoost":
            self.model = CatBoostClassifier(iterations=500, verbose=False, depth=10, learning_rate=0.05,
                                            loss_function='Logloss')
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
        asd = self.data.copy()
        asd['ASD'] = asd['ASD'].replace({0: "Isn't afflicted with ASD", 1: "Is afflicted with ASD"})
        sns.countplot(x='ASD', data=asd, hue='ASD', palette={"Isn't afflicted with ASD": "indianred",
                                                             "Is afflicted with ASD": "lightseagreen"}).set_facecolor(
            "floralwhite")
        plt.xlabel('Autsim Spectrum Disorder Classification')
        plt.ylabel('Questionaire Subjects')
        plt.title(self.datasetName + " Data")
        plt.gcf().set_facecolor("oldlace")
        plt.show()

    def plotTree(self):
        if isinstance(self.model, DecisionTreeClassifier):
            plt.figure(figsize=(12, 8))
            plot_tree(self.model, filled=True, feature_names=self.features.columns)
            plt.title(f'Decision Tree for the {self.datasetName} subjects')
            plt.gcf().set_facecolor("oldlace")
            plt.show()

    def plotCorrelations(self):
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.data[
            ['ASD'] + ['A1'] + ['A2'] + ['A3'] + ['A4'] + ['A5'] + ['A6'] + ['A7'] + ['A8'] + ['A9'] + ['A10']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='twilight', fmt=".2f")
        plt.title(self.datasetName + " Feature Correlation Matrix")
        plt.gcf().set_facecolor("oldlace")
        plt.show()

    def plotConfusionMatrix(self, cm):
        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, annot=True, cmap=["cadetblue", "brown"], cbar=False, fmt='g',
                    xticklabels=["Predicted to not have ASD", "Predicted to have ASD"],
                    yticklabels=["Doesn't actually have ASD", "Actually has ASD"])
        plt.title("Confusion Matrix for " + self.datasetName + " Subjects using " + self.model.__class__.__name__)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.gcf().set_facecolor("oldlace")
        plt.show()


models = ["catBoost", "Logistic Regression", "RandomForest", "SVC", "Decision Tree", "KNN", "Naive Bayes"]
csvFiles = [os.path.join("datasets", filename) for filename in os.listdir("datasets")
            if filename.endswith(('.csv'))]

trainTestSplittingResults = {}
crossValidationResults = {}
for file in csvFiles:
    print(f"\nDataset file: {file}")
    testing = ML(file)
    datasetTestResults = {}
    datasetValidationResults = {}
    testing.plotTarget()
    testing.plotCorrelations()
    for model in models:
        print(f"\nModel: {model}")
        testing.createModel(model)
        testing.trainWithSplitting()
        metrics = testing.showMetrics()
        datasetValidationResults[model] = testing.crossValidation()
        datasetTestResults[model] = metrics
        testing.plotTree()
    print(testing.xTest.info())
    trainTestSplittingResults[file] = datasetTestResults
    crossValidationResults[file] = datasetValidationResults
    input("Press Enter to continue...")


def print_results(results, result_type):
    for dataset, metricsDict in results.items():
        print(f"\nDataset: {dataset}")
        for model, metrics in metricsDict.items():
            print(f"\nModel: {model}")
            print(metrics)
        bestModel = max(metricsDict, key=lambda k: metricsDict[k]["accuracy"])
        print(
            f"\nMost accurate algorithm for {dataset} using {result_type} is {bestModel} with accuracy {metricsDict[bestModel]['accuracy']}")


print("\nSummary of Results:")
print_results(trainTestSplittingResults, "train-test splitting")
print_results(crossValidationResults, "k-folds cross validation")
