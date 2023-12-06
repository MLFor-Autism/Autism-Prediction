import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class RandomForestModel:
    def __init__(self, file_path, target_column):
        self.file_path = file_path
        self.target_column = target_column
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.clf = None
        self.predictions = None
        self.accuracy = None


    def convert_value(value):
        if value.lower() == "yes":
            return 1
        elif value.lower() == "no":
            return 0
        elif value.lower() in ["?", "other"]:
            return 1
        else:
            return 0
    

            
    def convert_value(value):
        if value.lower == "yes":
            return 1
    elif value.lower == "no":
        return 0
    elif value.lower in ["?", "other"]:
        return 1
    else:
        return 0



    def load_data(self):
        file_path = "E:\Autism-Prediction\datasets\csvAdam.csv"
        self.data = pd.read_csv(file_path)
        for col in self.data.columns:
            self.data[col] = self.data[col].apply(convert_value)




    def preprocess_data(self):
        X = self.data.drop(self.target_column, axis=1)
        y = self.data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.clf.fit(self.X_train, self.y_train)

    def predict(self):
        self.predictions = self.clf.predict(self.X_test)

    def evaluate(self):
        self.accuracy = accuracy_score(self.y_test, self.predictions)
        print(f"Accuracy: {self.accuracy:.2f}")

        print("Classification Report:")
        print(classification_report(self.y_test, self.predictions))

# Example usage:
random_forest = RandomForestModel('csvAdam.csv', 'Class/ASD')
random_forest.load_data()
random_forest.preprocess_data()
random_forest.train_model()
random_forest.predict()
random_forest.evaluate()
