from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import pickle

app = Flask(__name__)

# Load data and train the models
data = pd.read_csv('bank_notes.csv', header=None)
data.columns = ['var', 'skew', 'curt', 'entr', 'auth']
x = data.loc[:, data.columns != 'auth']
y = data.loc[:, data.columns == 'auth']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Model 1: Logistic Regression
scalar_lr = StandardScaler()
scalar_lr.fit(x_train)
clf_lr = LogisticRegression(solver='lbfgs', random_state=42, multi_class='auto')
clf_lr.fit(x_train, y_train.values.ravel())

# Model 2: Support Vector Machine
scalar_svm = StandardScaler()
scalar_svm.fit(x_train)
clf_svm = SVC(kernel='linear', random_state=42, probability=True)
clf_svm.fit(x_train, y_train.values.ravel())

# Model 3: K Nearest Neighbours
scalar_knn = StandardScaler()
scalar_knn.fit(x_train)
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(x_train, y_train)

# Model 4: Decision Tree
scalar_dt = StandardScaler()
scalar_dt.fit(x_train)
clf_dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, criterion='entropy')
clf_dt.fit(x_train, y_train)

# Save the models and scalers
pickle.dump(clf_lr, open("model_lr.pkl", "wb"))
pickle.dump(scalar_lr, open("scaler_lr.pkl", "wb"))
pickle.dump(clf_svm, open("model_svm.pkl", "wb"))
pickle.dump(scalar_svm, open("scaler_svm.pkl", "wb"))
pickle.dump(clf_knn, open("model_knn.pkl", "wb"))
pickle.dump(scalar_knn, open("scaler_knn.pkl", "wb"))
pickle.dump(clf_dt, open("model_dt.pkl", "wb"))
pickle.dump(scalar_dt, open("scaler_dt.pkl", "wb"))

# Load the models and scalers
model_lr = pickle.load(open("model_lr.pkl", "rb"))
scaler_lr = pickle.load(open("scaler_lr.pkl", "rb"))
model_svm = pickle.load(open("model_svm.pkl", "rb"))
scaler_svm = pickle.load(open("scaler_svm.pkl", "rb"))
model_knn = pickle.load(open("model_knn.pkl", "rb"))
scaler_knn = pickle.load(open("scaler_knn.pkl", "rb"))
model_dt = pickle.load(open("model_dt.pkl", "rb"))
scaler_dt = pickle.load(open("scaler_dt.pkl", "rb"))

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    var = float(request.form['var'])
    skew = float(request.form['skew'])
    curt = float(request.form['curt'])
    entr = float(request.form['entr'])
    new_banknote = np.array([var, skew, curt, entr], ndmin=2)

    selected_model = request.form['model']

    if selected_model == 'Logistic Regression':
        new_banknote = scaler_lr.transform(new_banknote)
        model = model_lr
        scaler = scaler_lr
    elif selected_model == 'Support Vector Machine':
        new_banknote = scaler_svm.transform(new_banknote)
        model = model_svm
        scaler = scaler_svm
    elif selected_model == 'KNN':
        new_banknote = scaler_knn.transform(new_banknote)
        model = model_knn
        scaler = scaler_knn
    elif selected_model == 'DTree':
        new_banknote = scaler_dt.transform(new_banknote)
        model = model_dt
        scaler = scaler_dt
    else:
        return "Invalid model selection."

    prediction = int(model.predict(new_banknote)[0])
    probabilities = model.predict_proba(new_banknote)[0].tolist()
    return render_template("index.html", prediction=prediction, probabilities=probabilities)

if __name__ == '__main__':
    app.run(debug=True)
