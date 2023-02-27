# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import os

## LOAD DATA
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


import random
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.linear_model import LogisticRegression
