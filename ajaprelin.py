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
train2 = train1.dropna()
train=train2.reindex(np.random.permutation(train2.index))
df_train_ori, df_test_ori = train_test_split(train, test_size=0.2, random_state=42)
y_test = df_test_ori.pop('fee')
y_test_100 = y_test.head(30)

train_df = df_train_ori.head(100)
evaluate_df = df_train_ori.tail(50)
test_df = df_test_ori.head(30)

MODEL_DIR = "tf_model_full"

print("train_df.shape = ", train_df.shape)
print("evaluate_df.shape = ", evaluate_df.shape)
print("test_df.shape = ", test_df.shape)

features = train_df.columns
categorical_features = ['colab', 'dis','loc', 'phase','req','study_type']
continuous_features = ['cfactor']
LABEL_COLUMN = 'fee'

tf.reset_default_graph()
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

def train_input_fn():
    return input_fn(train_df)
    #return input_fn(train_df.batch(128).repeat().make_one_shot_iterator().get_next())


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

STEPS = 500
#PRICE_NORM_FACTOR = 1000

# def get_variable_scope(scope, var, shape=None):
#     with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#         v = tf.get_variable(var, shape)
#     return v
"""
with tf.variable_scope(get_variable_scope("foo",model), reuse=False):
  model=tf.estimator.LinearRegressor(
    feature_columns=engineered_features,
    model_dir=None,
    label_dimension=1,
    weight_column=None,
    optimizer='Ftrl',
    # config=None,
    # partitioner=None,
    # warm_start_from=None,
    #loss_reduction=losses.Reduction.SUM
  )
with tf.variable_scope(get_variable_scope("foo",model), reuse=True):
  """
# def foo():
#   with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
#     model = tf.get_variable("model", [1])
#   return model

# model=tf.estimator.LinearRegressor(
#     feature_columns=engineered_features,
#     model_dir=None,
#     label_dimension=1,
#     weight_column=None,
#     optimizer='Ftrl',
#     # config=None,
#     # partitioner=None,
#     # warm_start_from=None,
#     #loss_reduction=losses.Reduction.SUM
#   )


  # Train the model
#tf.reset_default_graph()
#with tf.variable_scope(tf.get_variable_scope(), reuse=False):

#error coz an existing variable is being got in a non reusing environment


#capturing a scope and setting reuse
# with tf.variable_scope("foo") as scope:
#     v = tf.get_variable("v", [1])
#     scope.reuse_variables()
#     v1 = tf.get_variable("v", [1])
# assert v1 == v

#with tf.variable_scope("def_model") as scope:
  #tf.get_variable_scope().reuse_variables()
with tf.variable_scope("def_model") as scope:
  
  model=tf.estimator.LinearRegressor(
    feature_columns=engineered_features,
    model_dir=None,
    label_dimension=1,
    weight_column=None,
    optimizer='Ftrl',
    # config=None,
    # partitioner=None,
    # warm_start_from=None,
    #loss_reduction=losses.Reduction.SUM
  )
  #scope.reuse_variables()

# def foo():
#   with tf.variable_scope("foo",reuse=True):
#     model=tf.estimator.LinearRegressor(
#     feature_columns=engineered_features,
#     model_dir=None,
#     label_dimension=1,
#     weight_column=None,
#     optimizer='Ftrl',
#     # config=None,
#     # partitioner=None,
#     # warm_start_from=None,
#     #loss_reduction=losses.Reduction.SUM
#   )
#   return model

# model=tf.estimator.LinearRegressor(
#     feature_columns=engineered_features,
#     model_dir=None,
#     label_dimension=1,
#     weight_column=None,
#     optimizer='Ftrl',
    # config=None,
    # partitioner=None,
    # warm_start_from=None,
    #loss_reduction=losses.Reduction.SUM
  # )

#print(model)

#with tf.variable_scope("def_model", reuse=True):
# with tf.variable_scope("train_model") as scope:
  #scope.reuse_variables()
# model1=foo()
# model2=foo()
# assert model1==model2
#model2.train(input_fn=train_input_fn, steps=10)


# model=tf.estimator.LinearRegressor(
#     feature_columns=engineered_features,
#     model_dir=None,
#     label_dimension=1,
#     weight_column=None,
#     optimizer='Ftrl',
#     # config=None,
#     # partitioner=None,
#     # warm_start_from=None,
#     #loss_reduction=losses.Reduction.SUM
#   )
#with tf.variable_scope("def_model", reuse=True):

with tf.variable_scope(tf.get_variable_scope(), reuse=True):
# model.train(input_fn=train_input_fn, steps=1)
  model.train(input_fn=train_input_fn, steps=1)
#with tf.variable_scope(get_variable_scope(tf.get_variable_scope(), model), reuse=True):

  # Evaluate how the model performs on data it has not yet seen.
#tf.reset_default_graph()
eval_result = model.evaluate(input_fn=eval_input_fn,steps=1)


#   # The evaluation returns a Python dictionary. The "average_loss" key holds the
#   # Mean Squared Error (MSE).
average_loss = eval_result["average_loss"]

#   # Convert MSE to Root Mean Square Error (RMSE).
print("\n" + 80 * "*")
print("\nRMS error for the test set: ${:.0f}"
        .format(PRICE_NORM_FACTOR * average_loss**0.5))
