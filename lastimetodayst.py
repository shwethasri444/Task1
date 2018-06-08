from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
train1 = pd.read_csv('C:\\Users\\admin\\Desktop\\TV_gen_beta_1.csv',error_bad_lines=False)
train = train1.dropna()
df_train_ori, df_test_ori = train_test_split(train, test_size=0.2, random_state=42)
y_test = df_test_ori.pop('fee')

train_df = df_train_ori.head(1000)
evaluate_df = df_train_ori.tail(500)
test_df = df_test_ori.head(4000)

MODEL_DIR = "tf_model_full"

print("train_df.shape = ", train_df.shape)
print("evaluate_df.shape = ", evaluate_df.shape)
print("test_df.shape = ", test_df.shape)

features = train_df.columns
categorical_features = ['colab', 'dis','loc', 'phase','req','study_type']
continuous_features = ['cfactor']
LABEL_COLUMN = 'fee'


# Converting Data into Tensors
def input_fn(df, training = True):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values)
                       for k in continuous_features}

    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(df[k].size)],
        values=df[k].values,
        dense_shape=[df[k].size, 1])
        for k in categorical_features}

    # Merges the two dictionaries into one.
    feature_cols = dict(list(continuous_cols.items()) +
                        list(categorical_cols.items()))

    if training:
        # Converts the label column into a constant Tensor.
        label = tf.constant(df[LABEL_COLUMN].values)

        # Returns the feature columns and the label.
        return feature_cols, label
    
    # Returns the feature columns.
    return feature_cols


"""

#redefine input function AAAGGGGHHHHH I DONT KNOW HOW TO MAKE THAT STUPID FUNCTION ACCEPT THIS STUPID FEATURE AND LABEL COMBO WHAT IS ITS PROBLEM IN LIFE
def train_input_fn():
#returns data infinitely where shuffle buffer=1000 and batch_size=128
#find a replacement for shuffle coz thats not accepted by this version of tensorflow
    return input_fn(
        train_df.shuffle(1000).batch(128).repeat().make_one_shot_iterator().get_next())


def eval_input_fn():
    return input_fn(evaluate_df)


def test_input_fn():
     return input_fn(test_df.shuffle(1000).batch(128).make_one_shot_iterator().get_next(), False)
"""

def train_input_fn():
    return input_fn(train_df)

def eval_input_fn():
    return input_fn(evaluate_df)

def test_input_fn():
    return input_fn(test_df, False)    

engineered_features = []

for continuous_feature in continuous_features:
    engineered_features.append(
        tf.contrib.layers.real_valued_column(continuous_feature))


for categorical_feature in categorical_features:
    sparse_column = tf.contrib.layers.sparse_column_with_hash_bucket(
        categorical_feature, hash_bucket_size=1000)

    engineered_features.append(tf.contrib.layers.embedding_column(sparse_id_column=sparse_column, dimension=16,
                                                                  combiner="sum"))

"""
#DNNregressor does not generate error
model = tf.estimator.DNNRegressor(feature_columns=engineered_features, hidden_units=[10, 10], model_dir=MODEL_DIR)
#generates that huge ass bias,kernel,weights,adagrad(default optimiser of tf.estimator) error which can only be sorted with
#https://stackoverflow.com/questions/46697662/tensorflow-notfounderror-key-not-found-in-checkpoint
model.train(input_fn=train_input_fn,steps=500)

results = model.evaluate(input_fn=eval_input_fn)
for key in sorted(results):
  print('%s: %s' % (key, results[key]))

predicted_output = model.predict(input_fn=test_input_fn)
"""

#print(y_test)
#labels=tf.constant(y_test.values,dtype=np.float64)
labels=tf.constant(y_test.values)


regressor = tf.contrib.learn.DNNRegressor(feature_columns = engineered_features, 
                                          activation_fn = tf.nn.relu, hidden_units=[200, 100, 50, 25, 12])
print("learnt")
errorlist = []
#numberOfIterations = 5;
#for i in range(numberOfIterations):

regressor.fit(input_fn = train_input_fn , steps=500)
ev = regressor.evaluate(input_fn=lambda: input_fn(evaluate_df, training = True), steps=100)
for key in sorted(ev):
    print('%s: %s' % (key, ev[key]))
loss_score4 = ev["loss"]
print("Final Loss on the testing set: {0:f}".format(loss_score4))
predicted_output=regressor.predict(input_fn=test_input_fn)
predicted_output = list(predicted_output)
#shape1=list(tf.shape(predicted_output))
#print(shape1)
print(predicted_output[:10])
error=tf.metrics.mean_squared_error(tf.cast(labels, tf.float64),tf.cast(predicted_output, tf.float64))
tf.Print(error, [error])  # This does nothing
error = tf.Print(error, [error])  # Here we are using the value returned by tf.Print
result = error + 1
#shape=list(tf.shape(error))
#print(shape)

#try printing with tensor.eval() coz that returns numpy array when tf.session() is running
#error_t=tf.slice(error, [1, 0, 0], [1, 3, 3])
#errorarray= list(error_t)
#print(errorarray)
#error = mean_squared_error(y_test,predicted_output)
errorlist = np.append(errorlist,error)
#print("Remaining:",numberOfIterations-i)



