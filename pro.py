import random
import pandas 
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_array

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import learn


train1 = pandas.read_csv('C:\\Users\\admin\\Desktop\\TV_gen_beta_1.csv',error_bad_lines=False)
train = train1.dropna()
y = train.pop('fee')
categorical_vars = ['colab', 'dis','loc', 'phase','req','study_type']
continues_vars = ['cfactor']
X = train[categorical_vars + continues_vars].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
