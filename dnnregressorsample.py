"""Regression using the DNNRegressor Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import collections

import numpy as np
import tensorflow as tf

try:
  import pandas as pd  # pylint: disable=g-import-not-at-top
except ImportError:
  pass

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"

# Order is important for the csv-readers, so we use an OrderedDict here.
defaults = collections.OrderedDict([
    ("symboling", [0]),
    ("normalized-losses", [0.0]),
    ("make", [""]),
    ("fuel-type", [""]),
    ("aspiration", [""]),
    ("num-of-doors", [""]),
    ("body-style", [""]),
    ("drive-wheels", [""]),
    ("engine-location", [""]),
    ("wheel-base", [0.0]),
    ("length", [0.0]),
    ("width", [0.0]),
    ("height", [0.0]),
    ("curb-weight", [0.0]),
    ("engine-type", [""]),
    ("num-of-cylinders", [""]),
    ("engine-size", [0.0]),
    ("fuel-system", [""]),
    ("bore", [0.0]),
    ("stroke", [0.0]),
    ("compression-ratio", [0.0]),
    ("horsepower", [0.0]),
    ("peak-rpm", [0.0]),
    ("city-mpg", [0.0]),
    ("highway-mpg", [0.0]),
    ("price", [0.0])
])  # pyformat: disable


types = collections.OrderedDict((key, type(value[0]))
                                for key, value in defaults.items())


def _get_imports85():
  path = tf.contrib.keras.utils.get_file(URL.split("/")[-1], URL)
  return path


def dataset(y_name="price", train_fraction=0.7):
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
  path = _get_imports85()

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
    return ~in_training_set(line)

  base_dataset = (
      tf.data
      # Get the lines from the file.
      .TextLineDataset(path)
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
  test = (base_dataset.filter(in_test_set).cache().map(decode_line))

  return train, test


def raw_dataframe():
  """Load the imports85 data as a pd.DataFrame."""
  # Download and cache the data
  path = _get_imports85()

  # Load it into a pandas dataframe
  df = pd.read_csv(path, names=types.keys(), dtype=types, na_values="?")

  return df


def load_data(y_name="price", train_fraction=0.7, seed=None):
  """Get the imports85 data set.
  A description of the data is available at:
    https://archive.ics.uci.edu/ml/datasets/automobile
  The data itself can be found at:
    https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data
  Args:
    y_name: the column to return as the label.
    train_fraction: the fraction of the dataset to use for training.
    seed: The random seed to use when shuffling the data. `None` generates a
      unique shuffle every run.
  Returns:
    a pair of pairs where the first pair is the training data, and the second
    is the test data:
    `(x_train, y_train), (x_test, y_test) = get_imports85_dataset(...)`
    `x` contains a pandas DataFrame of features, while `y` contains the label
    array.
  """
  # Load the raw data columns.
  data = raw_dataframe()

  # Delete rows with unknowns
  data = data.dropna()

  # Shuffle the data
  np.random.seed(seed)

  # Split the data into train/test subsets.
  x_train = data.sample(frac=train_fraction, random_state=seed)
  x_test = data.drop(x_train.index)

  # Extract the label from the features dataframe.
  y_train = x_train.pop(y_name)
  y_test = x_test.pop(y_name)

  return (x_train, y_train), (x_test, y_test) # pylint: disable=g-bad-import-order
 # pylint: disable=g-bad-import-order

STEPS = 5000
PRICE_NORM_FACTOR = 1000


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
        train.shuffle(1000).batch(128)
        # Repeat forever
        .repeat().make_one_shot_iterator().get_next())

  # Build the validation input_fn.
  def input_test():
    return (test.shuffle(1000).batch(128)
            .make_one_shot_iterator().get_next())

  # The first way assigns a unique weight to each category. To do this you must
  # specify the category's vocabulary (values outside this specification will
  # receive a weight of zero). Here we specify the vocabulary using a list of
  # options. The vocabulary can also be specified with a vocabulary file (using
  # `categorical_column_with_vocabulary_file`). For features covering a
  # range of positive integers use `categorical_column_with_identity`.
  body_style_vocab = ["hardtop", "wagon", "sedan", "hatchback", "convertible"]
  body_style = tf.feature_column.categorical_column_with_vocabulary_list(
      key="body-style", vocabulary_list=body_style_vocab)
  make = tf.feature_column.categorical_column_with_hash_bucket(
      key="make", hash_bucket_size=50)

  feature_columns = [
      tf.feature_column.numeric_column(key="curb-weight"),
      tf.feature_column.numeric_column(key="highway-mpg"),
      # Since this is a DNN model, convert categorical columns from sparse
      # to dense.
      # Wrap them in an `indicator_column` to create a
      # one-hot vector from the input.
      tf.feature_column.indicator_column(body_style),
      # Or use an `embedding_column` to create a trainable vector for each
      # index.
      tf.feature_column.embedding_column(make, dimension=3),
  ]

  # Build a DNNRegressor, with 2x20-unit hidden layers, with the feature columns
  # defined above as input.
  model = tf.estimator.DNNRegressor(
      hidden_units=[20, 20], feature_columns=feature_columns)

  # Train the model.
  model.train(input_fn=input_train, steps=STEPS)

  eval_result_test = model.evaluate(input_fn=input_test,steps=500)
  eval_result_train = model.evaluate(input_fn=input_train,steps=500)
  # The evaluation returns a Python dictionary. The "average_loss" key holds the
  # Mean Squared Error (MSE).
  average_loss_test = eval_result_test["average_loss"]
  average_loss_train = eval_result_train["average_loss"]

  rms_error_test=PRICE_NORM_FACTOR * average_loss_test**0.5
  rms_error_train=PRICE_NORM_FACTOR * average_loss_train**0.5
  # Convert MSE to Root Mean Square Error (RMSE).
  # df = pd.read_csv('C:\\Users\\admin\\Desktop\\TV_gen_beta_1.csv', names=types.keys(), dtype=types, na_values="?")
  df=raw_dataframe()
  df = df.dropna()
  MEAN = df["price"].mean()
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