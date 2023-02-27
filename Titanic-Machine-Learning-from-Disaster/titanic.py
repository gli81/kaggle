# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os

"""
LOAD DATA
"""
cwd = os.getcwd()
# print(cwd)
train_data = pd.read_csv(os.path.join(
    cwd, "Titanic-Machine-Learning-from-Disaster/input/train.csv"
))
# test_data = pd.read_csv("Titanic-Machine-Learning-from-Disaster\\input\\test.csv")
test_data = pd.read_csv(os.path.join(
    cwd, "Titanic-Machine-Learning-from-Disaster/input/test.csv"
))

combined = [train_data, test_data]


"""
## A LOOK AT DATA
# print(train_data.columns.values)
# print(train_data.head())
## categorical ones?
## numerical ones? continuous or discrete?
some points:
Ticket is a mix of numeric and alphanumeric data types.
Cabin is alphanumeric.
## which features may contain errors or typos?
## which features contain NAs? Cabin, Age, Embarked
"""
train_data.info()
print('=' * 50)
test_data.info()



"""
## DESCRIBE DATA DISTRIBUTION
include=["O"] selects non-numeric columns
EMBARKED => where the passenger got on board?
"""
print(train_data.describe())
print(train_data.describe(include=["O"]))


"""
Women (Sex=female) were more likely to have survived.
Children (Age<?) were more likely to have survived.
The upper-class passengers (Pclass=1) were more likely to 
have survived.
"""
## choose only the grouping column and the aggregated column
print(train_data[["Pclass", "Survived"]]
        .groupby("Pclass")
        .mean()
        .sort_values(by="Survived", ascending=False))
print(train_data[["Sex", "Survived"]]
        .groupby("Sex")
        .mean()
        .sort_values(by="Survived", ascending=False))
print(train_data[["SibSp", "Survived"]]
        .groupby("SibSp")
        .mean()
        .sort_values(by="Survived", ascending=False))
print(train_data[["Parch", "Survived"]]
        .groupby("Parch")
        .mean()
        .sort_values(by="Survived", ascending=False))


import seaborn as sns
import matplotlib.pyplot as plt

"""
## VISUALIZING NUMERICAL VARIABLE (Age)
"""
g = sns.FacetGrid(train_data, col="Survived")
g.map(plt.hist, "Age", bins=20)
plt.show()
"""
## VISUALIZING ORDINAL VARIABLE (Pclass)
"""
grid = sns.FacetGrid(train_data, col="Survived", row="Pclass")
grid.map(plt.hist, "Age", bins=20)
plt.show()
"""
## VISUALIZING CATEGORICAL VARIABLE (Embarked)
"""
grid = sns.FacetGrid(train_data, col="Embarked")
grid.map(sns.pointplot,
        "Pclass", "Survived", "Sex",
        palette="deep"
)
grid.add_legend()
plt.show()

import random

from sklearn.linear_model import LogisticRegression
