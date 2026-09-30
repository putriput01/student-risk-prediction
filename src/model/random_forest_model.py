from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self):
           self.model = RandomForestClassifier(random_state=42)

    def train(self, X, y):
           self.model.fit(X, y)
           return self.model

    def predict(self, X):
           return self.model.predict(X)

    def predict_proba(self, X):
           return self.model.predict_proba(X)