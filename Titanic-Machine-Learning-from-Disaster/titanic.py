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


import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
