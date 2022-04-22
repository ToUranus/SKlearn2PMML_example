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


# data
iris = load_iris()
X = iris.data
y = iris.target

print(X)
