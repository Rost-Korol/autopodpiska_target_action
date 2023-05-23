import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import dill
import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score


with open('model/models/logreg_pipe.pkl', 'rb') as file:
    pipe = dill.load(file)

with open('data/join_data.pkl', 'rb') as file:
    df = pickle.load(file)


x = df.drop(columns='target')
y = df.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
grid_params = {'model__C': [0.01, 0.1, 1, 10, 100]}

start = time.time()

log_grid = GridSearchCV(pipe['model'], param_grid=grid_params, cv=5, scoring='roc_auc')
log_grid.fit(x_train, y_train)

print(f"Лучшее значение AUC: {round(log_grid.best_score_, 4)}")
print(f"Значение AUC на тестовом наборе: {round(log_grid.score(x_test, y_test), 4)}")
print(f"Наилучшие параметры: {log_grid.best_params_}")

print(f'Time of work: {time.time() - start}')







# # Параметры для Logregression model
# LogisticRegression(max_iter=10000, class_weight='balanced', C = 0.1))
# Лучшее значение AUC: 0.686
# Значение AUC на тестовом наборе: 0.683
# Наилучшие параметры: {'logreg__C': 0.1}







