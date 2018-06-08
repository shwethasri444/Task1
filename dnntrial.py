
"""Linear regression with categorical features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf
import tflearn


try:
  import pandas as pd  # pylint: disable=g-import-not-at-top
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
  """Load the imports85 data as a (train,test) pair of `Dataset`.
  Each dataset generates (features_dict, label) pairs.
  Args:
    y_name: The name of the column to use as the label.
    train_fraction: A float, the fraction of data to use for training. The
        remainder will be used for evaluation.
  Returns:
    A (train,test) pair of `Datasets`
  """
  # Download and cache the data
  #path = _get_imports85()

  # Define how the lines of the file should be parsed
  def decode_line(line):
    """Convert a csv line into a (features_dict,label) pair."""
    # Decode the line to a tuple of items based on the types of
    # csv_header.values().
    items = tf.decode_csv(line, list(defaults.values()))

    # Convert the keys and items to a dict.
    pairs = zip(defaults.keys(), items)
    features_dict = dict(pairs)

    # Remove the label from the features_dict
    label = features_dict.pop(y_name)

    return features_dict, label

  def has_no_question_marks(line):
    """Returns True if the line of text has no question marks."""
    # split the line into an array of characters
    chars = tf.string_split(line[tf.newaxis], "").values
    # for each character check if it is a question mark
    is_question = tf.equal(chars, "?")
    any_question = tf.reduce_any(is_question)
    no_question = ~any_question

    return no_question

  def in_training_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # If you randomly split the dataset you won't get the same split in both
    # sessions if you stop and restart training later. Also a simple
    # random split won't work with a dataset that's too big to `.cache()` as
    # we are doing here.
    num_buckets = 1000000
    bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
    # Use the hash bucket id as a random number that's deterministic per example
    return bucket_id < int(train_fraction * num_buckets)

  def in_test_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # Items not in the training set are in the test set.
    # This line must use `~` instead of `not` because `not` only works on python
    # booleans but we are dealing with symbolic tensors.
    return tf.logical_not(in_training_set(line))

  base_dataset = (
      tf.data
      # Get the lines from the file.
      .TextLineDataset('C:\\Users\\admin\\Desktop\\sample.csv')
      # drop lines with question marks.
      .filter(has_no_question_marks))

  train = (base_dataset
           # Take only the training-set lines.
           .filter(in_training_set)
           # Decode each line into a (features_dict, label) pair.
           .map(decode_line)
           # Cache data so you only decode the file once.
           .cache())

  # Do the same for the test-set.
  test = (base_dataset.filter(in_test_set).map(decode_line).cache())

  return train, test

STEPS = 2000
PRICE_NORM_FACTOR = 100


def main(argv):
  """Builds, trains, and evaluates the model."""
  assert len(argv) == 1
  (train, test) = dataset()

  # Switch the labels to units of thousands for better convergence.
  def normalize_price(features, labels):
    return features, labels / PRICE_NORM_FACTOR

  train = train.map(normalize_price)
  test = test.map(normalize_price)

  # Build the training input_fn.
  def input_train():
    return (
        # Shuffling with a buffer larger than the data set ensures
        # that the examples are well mixed.
        train.shuffle(2000).batch(256)
        # Repeat forever
        .repeat().make_one_shot_iterator().get_next())

  # Build the validation input_fn.
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

  df = pd.read_csv('C:\\Users\\admin\\Desktop\\sample.csv', names=types.keys(), dtype=types, na_values="?")
  df = df.dropna()
  MEAN = df["fee"].mean()

  fee_df=list(df["fee"].values)
  fee_tf = tf.convert_to_tensor(fee_df, dtype=tf.float32)

  cfactor_df=list(df["cfactor"].values)
  cfactor_tf = tf.convert_to_tensor(cfactor_df, dtype=tf.float32)
  
  # corr_cfactor=tf.contrib.metrics.streaming_pearson_correlation(
  #   fee_tf,
  #   tf.feature_column.embedding_column(colab_column),
  #   weights=None,
  #   metrics_collections=None,
  #   updates_collections=None,
  #   name=None
  # )


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
      # This model uses the same two numeric features as `linear_regressor.py`
      tf.feature_column.numeric_column(key="cfactor"),
      
      tf.feature_column.indicator_column(colab_column),
      tf.feature_column.indicator_column(dis_column),
      tf.feature_column.indicator_column(loc_column),
      tf.feature_column.indicator_column(phase_column),
      tf.feature_column.indicator_column(req_column),
      tf.feature_column.indicator_column(study_type_column),
  ]

  
  
  # Build the Estimator.
  # model = tf.estimator.DNNRegressor(
  #     hidden_units=[6, 6], 
  #     feature_columns= feature_columns,
  #     optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.01,l1_regularization_strength=0.005))

  model = tf.estimator.DNNRegressor(
      hidden_units=[10, 10], 
      feature_columns= feature_columns,
      optimizer=tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9,beta2=0.999,epsilon=1e-08))

  # Train the model.
  model.train(input_fn=input_train, steps=2000)

  # Evaluate how the model performs on data it has not yet seen.
  eval_result_test = model.evaluate(input_fn=input_test,steps=2000)
  eval_result_train = model.evaluate(input_fn=input_train,steps=2000)
  # The evaluation returns a Python dictionary. The "average_loss" key holds the
  # Mean Squared Error (MSE).
  average_loss_test = eval_result_test["average_loss"]
  average_loss_train = eval_result_train["average_loss"]

  rms_error_test=PRICE_NORM_FACTOR * average_loss_test**0.5
  rms_error_train=PRICE_NORM_FACTOR * average_loss_train**0.5
  # Convert MSE to Root Mean Square Error (RMSE).
  

  # tf.feature_column.indicator_column(colab_column),
  #     tf.feature_column.indicator_column(dis_column),
  #     tf.feature_column.indicator_column(loc_column),
  #     tf.feature_column.indicator_column(phase_column),
  #     tf.feature_column.indicator_column(req_column),
  #     tf.feature_column.indicator_column(study_type_column),
  #     body_style_column,
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
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)