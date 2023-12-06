import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import os


class catBoost:
    def __init__(self, filePath):
        self.model = None
        self.data = pd.read_csv(filePath)
        self.features = self.data.drop("Class/ASD", axis=1)
        self.target = self.data["Class/ASD"]
        self.target = self.target.replace({'YES': 1, 'NO': 0, '?': 'Others', 'others': 'Others'})
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.features, self.target, test_size=0.3,
                                                                            random_state=42)

    def train(self):
        self.normalize(self.xTrain)
        self.model.fit(self.xTrain, self.yTrain)

    def predict(self):
        self.normalize(self.xTest)
        return self.model.predict(self.xTest)

    def showMetrics(self):
        accuracy = accuracy_score(self.yTest, self.predict())
        precision = precision_score(self.yTest, self.predict())
        recall = recall_score(self.yTest, self.predict())
        f1 = f1_score(self.yTest, self.predict())
        cm = confusion_matrix(self.yTest, self.predict())
        report = classification_report(self.yTest, self.predict())

        print(f"Confusion Matrix: {cm}")
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


    def createModel(self,model):
        if model == "catBoost":
            self.model = CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, loss_function='Logloss')



csvFiles = []
for filename in os.listdir("datasets"):
    if filename.endswith(('csvAdam.csv', 'train.csv', 'autism_screening.csv')):  # Add more extensions if needed
        path = os.path.join("datasets", filename)
        csvFiles.append(path)

for file in csvFiles:
    print(f"Dataset file: {file}")
    testing = catBoost(file)
    testing.createModel("catBoost")
    testing.train()
    testing.showMetrics()
