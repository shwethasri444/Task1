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

def predict_input_fn():
    return input_fn(new_df, False)

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
print("Enter your option :\n 1. Train\n2. Evaluate\n 3. Test\n 4. Predict\n")

def Train_Func():
  model=tf.estimator.LinearRegressor(
    feature_columns=engineered_features,
    model_dir='',
    label_dimension=1,
    weight_column=None,
    optimizer='Ftrl',
    config=None,
    partitioner=None,
    warm_start_from=None,
    loss_reduction=losses.Reduction.SUM
  )
  model.train(input_fn=train_input_fn, steps=1)

  # Evaluate how the model performs on data it has not yet seen.
def Evaluate_Func():
  eval_result = model.evaluate(input_fn=eval_input_fn,steps=1)
  average_loss = eval_result["average_loss"]
  # Convert MSE to Root Mean Square Error (RMSE).
  print("\n" + 80 * "*")
  print("\nRMS error for the test set: ${:.0f}"
        .format(PRICE_NORM_FACTOR * average_loss**0.5))


def Predict_Func():
  'colab', 'dis','loc', 'phase','req','study_type'
  colab_in = input("colab")
  dis_in = input("dis")
  loc_in = input("loc")
  phase_in = input("phase")
  req_in = input("req")
  study_type_in = input("study_type")
  new_df = [idno,colab,dis,loc,phase,req,study_type,patients,fee,total_score,cfactor,clinic,third_party,invest,pass_through,data_manag,monitor,regulatory,pm,budget]
  predict(
    predict_input_fn,
    predict_keys="fee",
    hooks=None,
    checkpoint_path=None,
    yield_single_examples=False
  )

  # The evaluation returns a Python dictionary. The "average_loss" key holds the
  # Mean Squared Error (MSE).
average_loss = eval_result["average_loss"]

  # Convert MSE to Root Mean Square Error (RMSE).
print("\n" + 80 * "*")
print("\nRMS error for the test set: ${:.0f}"
        .format(PRICE_NORM_FACTOR * average_loss**0.5))
