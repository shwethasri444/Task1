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


train1 = pandas.read_csv('C:\\Users\\admin\\Downloads\\titanic3.csv',error_bad_lines=False)
train = train1.dropna()
y = train.pop('survived')
# Drop all unique columns. List all variables for future reference.
categorical_vars = ['pclass', 'sex', 'embarked']
continues_vars = ['age', 'sibsp', 'parch', 'fare']
X = train[categorical_vars + continues_vars].fillna(0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def input_fn():
    dataset= tf.contrib.data.make_csv_dataset(x,128,num_epochs=num_epochs)
    return 

# Pandas input function
"""
def pandas_input_fn(x, y=None, batch_size=128, num_epochs=None):
  def input_fn():
    if y is not None:
      x['target'] = y
    queue = learn.dataframe.queues.feeding_functions.enqueue_data(
      x, 1000, shuffle=num_epochs is None, num_epochs=num_epochs)
    if num_epochs is None:
      features = queue.dequeue_many(batch_size)
    else:
      features = queue.dequeue_up_to(batch_size)
    features = dict(zip(['index'] + list(x.columns), features))
    if y is not None:
      target = features.pop('target')
      return features, target
    return features
  return input_fn
"""

# Process categorical variables into ids.
X_train = X_train.copy()
X_test = X_test.copy()
categorical_var_encoders = {}
for var in categorical_vars:
  le = LabelEncoder().fit(X_train[var])
  X_train[var + '_ids'] = le.transform(X_train[var].astype(str))
  X_test[var + '_ids'] = le.transform(X_test[var].astype(str))
  X_train.pop(var)
  X_test.pop(var)
  categorical_var_encoders[var] = le


CATEGORICAL_EMBED_SIZE = 10 # Note, you can customize this per variable.


# 3 layer neural network with hyperbolic tangent activation.
def dnn_tanh(features, target):
    target = tf.one_hot(target, 2, 1.0, 0.0)
    # Organize continues features.
    final_features = [tf.expand_dims(tf.cast(features[var], tf.float32), 1) for var in continues_vars]
    # Embed categorical variables into distributed representation.
    for var in categorical_vars:
        feature = learn.ops.categorical_variable(
            features[var + '_ids'], len(categorical_var_encoders[var].classes_),
            embedding_size=CATEGORICAL_EMBED_SIZE, name=var)
        final_features.append(feature)
    # Concatenate all features into one vector.
    features = tf.concat(1, final_features)
    # Deep Neural Network
    logits = layers.stack(features, layers.fully_connected, [10, 20, 10],
        activation_fn=tf.tanh)
    prediction, loss = learn.models.logistic_regression(logits, target)
    train_op = layers.optimize_loss(loss,
        tf.contrib.framework.get_global_step(), optimizer='SGD', learning_rate=0.05)
    return tf.argmax(prediction, dimension=1), loss, train_op

random.seed(42)
classifier = learn.Estimator(model_fn=dnn_tanh)
# Note: not training this alomst at all.
classifier.fit(input_fn=pandas_input_fn(X_train, y_train), steps=100)
preds = list(classifier.predict(input_fn=pandas_input_fn(X_test, num_epochs=1), as_iterable=True))
print(accuracy_score(y_test, preds))
