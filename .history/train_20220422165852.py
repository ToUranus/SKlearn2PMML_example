#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2022/04/22 16:25:14
@Author      :ShiFei
@version      :1.0
'''

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

# data
iris = load_iris()
X = iris.data
y = iris.target

# Training and Testing split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=10)

# pipeline
classfier = GradientBoostingClassifier(random_state=10)
pipe = Pipeline([('Standar',StandardScaler()),('Classfier',classfier)])

# fit
pipe.fit(x_train,y_train)

# evalute
print(pipe.score(x_test,y_test))

# converting pipelines to PMML:you must have a lib named sklearn2pmml,github:https://github.com/jpmml/sklearn2pmml
from sklearn

