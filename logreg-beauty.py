import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
np.random.seed(32)

X = pd.read_csv('./data/processed/beauty.csv')
y = pd.read_csv('./data/processed/beautylabels.csv')

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.75)
num_epochs = 10
avg_acc = []
avg_f1 = []
c_values = [0.01, 0.05, 0.25, 0.5, 1, 1.2, 1.5, 1.8, 2]
for c in c_values:
    accs = []
    f1 = []
    for i in range(num_epochs):
        lr = LogisticRegression(C=c)
        lr.fit(X_train, np.ravel(y_train)) 
        accs.append(accuracy_score(np.ravel(y_train), lr.predict(X_train)))
        f1.append(f1_score(np.ravel(y_train), lr.predict(X_train)))
    avg_acc.append(np.mean(accs))
    avg_f1.append(np.mean(f1))

for i in range(len(c_values)):
    print(f'c : {c_values[i]}, accuracy : {avg_acc[i]}, f1 : {avg_f1[i]}')

final_model = LogisticRegression(C=2)
final_model.fit(X, y)
test_acc = accuracy_score(y_val, final_model.predict(X_val))
test_f1 = f1_score(y_val, final_model.predict(X_val))
print(f'Test accuracy : {test_acc}, Test f1_score : {test_f1}')

feature_to_coef = {
    word: coef for word, coef in zip(
        X.columns, final_model.coef_[0]
    )
}
for best_positive in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1], 
    reverse=True)[:5]:
    print (best_positive)
    
for best_negative in sorted(
    feature_to_coef.items(), 
    key=lambda x: x[1])[:5]:
    print (best_negative)
X_test = X
y_test = y
X_test = pd.read_csv ('./data/processed/tools.csv')
y_test = pd.read_csv('./data/processed/toollabels.csv')
X_test = X_test.reindex(columns = X.columns)
y_test = y_test.reindex(columns = y.columns)
X_test = X_test.fillna(0)
y_test = y_test.fillna(0)
print(X_test.shape)
print(y_test.shape)

test_acc = accuracy_score(y_test, final_model.predict(X_test))
test_f1 = f1_score(y_test, final_model.predict(X_test))
print(f'Test accuracy : {test_acc}, Test f1_score : {test_f1}')
