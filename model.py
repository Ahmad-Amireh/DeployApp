from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle 

iris = load_iris()
columns=iris.feature_names
print(columns)

X,y = iris.data, iris.target 
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
score = classification_report(y_test, y_pred)

print(score)

with open ("model.pkl", "wb") as f: 
    pickle.dump(model, f)
