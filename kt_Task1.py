
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf
import tflearn

try:
  import pandas as pd  
except ImportError:
  pass
tf.reset_default_graph()

defaults = collections.OrderedDict([("idno",[0]),
                         ("colab",[""]),
                         ("dis",[""]),
                         ("loc",[""]),
                         ("phase",[""]),
                         ("req",[""]),
                         ("study_type",[""]),
                         ("patients",[0]),
                         ("fee",[0.0]),
                         ("total_score",[0]),
                         ("cfactor",[0.0]),
                         ("clinic",[0.0]),
                         ("third_party",[0.0]),
                         ("invest",[0.0]),
                         ("pass_through",[0.0]),
                         ("data_manag",[0.0]),
                         ("monitor",[0.0]),
                         ("regulatory",[0.0]),
                         ("pm",[0.0]),
                         ("budget",[0.0])])

types = collections.OrderedDict((key, type(value[0]))
                                for key, value in defaults.items())

def dataset(y_name="fee", train_fraction=0.7):
  
  def decode_line(line):
    items = tf.decode_csv(line, list(defaults.values()))
    pairs = zip(defaults.keys(), items)
    features_dict = dict(pairs)
    label = features_dict.pop(y_name)

    return features_dict, label

  def has_no_question_marks(line):

    chars = tf.string_split(line[tf.newaxis], "").values
    is_question = tf.equal(chars, "?")
    any_question = tf.reduce_any(is_question)
    no_question = ~any_question

    return no_question

  def in_training_set(line):

    num_buckets = 1000000
    bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
   
    return bucket_id < int(train_fraction * num_buckets)

  def in_test_set(line):
    
    return tf.logical_not(in_training_set(line))

  base_dataset = (
      tf.data
      .TextLineDataset('C:\\Users\\admin\\Desktop\\shwetha\\sample.csv')
      .filter(has_no_question_marks))

  train = (base_dataset.filter(in_training_set).map(decode_line).cache())

  test = (base_dataset.filter(in_test_set).map(decode_line).cache())

  return train, test

STEPS = 2000
PRICE_NORM_FACTOR = 100


def main(argv):

  assert len(argv) == 1
  (train, test) = dataset()

  def normalize_price(features, labels):
    return features, labels / PRICE_NORM_FACTOR

  train = train.map(normalize_price)
  test = test.map(normalize_price)

  def input_train():
    return (
        train.shuffle(2000).batch(256).repeat().make_one_shot_iterator().get_next())

  def input_test():
    return (test.shuffle(2000).batch(256)
            .make_one_shot_iterator().get_next())

  colab_column = tf.feature_column.categorical_column_with_hash_bucket(
      key="colab", hash_bucket_size=50)
  dis_column = tf.feature_column.categorical_column_with_hash_bucket(
      key="dis", hash_bucket_size=50)
  loc_column = tf.feature_column.categorical_column_with_hash_bucket(
      key="loc", hash_bucket_size=50)
  phase_column = tf.feature_column.categorical_column_with_hash_bucket(
      key="phase", hash_bucket_size=50)
  req_column = tf.feature_column.categorical_column_with_hash_bucket(
      key="req", hash_bucket_size=50)
  study_type_column = tf.feature_column.categorical_column_with_hash_bucket(
      key="study_type", hash_bucket_size=50)

  df = pd.read_csv('C:\\Users\\admin\\Desktop\\shwetha\\sample.csv', names=types.keys(), dtype=types, na_values="?")
  df = df.dropna()
  MEAN = df["fee"].mean()

  fee_df=list(df["fee"].values)
  fee_tf = tf.convert_to_tensor(fee_df, dtype=tf.float32)

  cfactor_df=list(df["cfactor"].values)
  cfactor_tf = tf.convert_to_tensor(cfactor_df, dtype=tf.float32)

  corr_cfactor=tf.contrib.metrics.streaming_pearson_correlation(
    fee_tf,
    cfactor_tf,
    weights=None,
    metrics_collections=None,
    updates_collections=None,
    name=None
  )

  
  corr_saneformat_cfactor = tf.get_variable("corr",[1,2])
  sess = tf.InteractiveSession()  
  sess.run(tf.global_variables_initializer())
  print(sess.run(corr_saneformat_cfactor))
  sess.close()

  feature_columns = [
      tf.feature_column.numeric_column(key="cfactor"),
      tf.feature_column.indicator_column(colab_column),
      tf.feature_column.indicator_column(dis_column),
      tf.feature_column.indicator_column(loc_column),
      tf.feature_column.indicator_column(phase_column),
      tf.feature_column.indicator_column(req_column),
      tf.feature_column.indicator_column(study_type_column),
  ]

  model = tf.estimator.DNNRegressor(
      hidden_units=[7, 7], 
      feature_columns= feature_columns,
      optimizer=tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08))

  model.train(input_fn=input_train, steps=2000)

  eval_result_test = model.evaluate(input_fn=input_test,steps=2000)
  eval_result_train = model.evaluate(input_fn=input_train,steps=2000)

  average_loss_test = eval_result_test["average_loss"]
  average_loss_train = eval_result_train["average_loss"]

  rms_error_test=PRICE_NORM_FACTOR * average_loss_test**0.5
  rms_error_train=PRICE_NORM_FACTOR * average_loss_train**0.5

  print("\nMEAN fee: ${:.0f}"
        .format(MEAN))

  print("\n" + 80 * "*")
  print("\nRMS error for the training set: ${:.0f}"
        .format(1-rms_error_train))
  print("\nTraining accuracy: ${:.0f}"
        .format((1-rms_error_train/MEAN)*100))

  rms_error_test=PRICE_NORM_FACTOR * average_loss_test**0.5
  # Convert MSE to Root Mean Square Error (RMSE).
  print("\n" + 80 * "*")
  print("\nRMS error for the test set: ${:.0f}"
        .format(rms_error_test))
  print("\nTesting accuracy: ${:.0f}"
        .format((1-rms_error_test/MEAN)*100))

if __name__ == "__main__":

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)