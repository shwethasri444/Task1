import pandas as pd
import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from sklearn.cross_validation import train_test_split

import warnings
warnings.filterwarnings("ignore")
#df_train_ori = pd.read_csv('../input/train.csv')
#df_test_ori = pd.read_csv('../input/test.csv')
train1 = pd.read_csv('C:\\Users\\admin\\Desktop\\TV_gen_beta_1.csv',error_bad_lines=False)
train = train1.dropna()
#X=train.fillna(0)
df_train_ori, df_test_ori = train_test_split(train, test_size=0.2, random_state=42)
y_test = df_test_ori.pop('fee')
print(y_test[10:1])

train_df = df_train_ori.head(15000)
evaluate_df = df_train_ori.tail(500)

test_df = df_test_ori.head(1000)

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
    
    # Returns the feature columns    
    return feature_cols

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

model = tf.contrib.learn.DNNRegressor(
    feature_columns=engineered_features, hidden_units=[10, 10], model_dir=MODEL_DIR)

model = tf.estimator.DNNRegressor(feature_columns=engineered_features, hidden_units=[10, 10], model_dir=MODEL_DIR)
model.train(input_fn=train_input_fn,steps=500)

results = model.evaluate(input_fn=eval_input_fn)
for key in sorted(results):
  print('%s: %s' % (key, results[key]))

predicted_output = model.predict(input_fn=test_input_fn)


"""
# Training Our Model
wrap = model.fit(input_fn=train_input_fn, steps=500)


evaluate(
    x=None,
    y=None,
    input_fn=None,
    feed_fn=None,
    batch_size=None,
    steps=None,
    metrics=None,
    name=None,
    checkpoint_path=None,
    hooks=None
)
# Evaluating Our Model
print('Evaluating ...')
results = model.evaluate(input_fn=eval_input_fn, steps=1)
for key in results:
    print("%s: %s" % (key, results[key]))
def my_auc(features, labels, predictions):
    return {'auc': tf.metrics.auc(
      labels, predictions['logistic'], weights=features['weight'])}

  estimator = tf.estimator.DNNClassifier(...)
  estimator = tf.contrib.estimator.add_metrics(estimator, my_auc)
  estimator.train(...)
  estimator.evaluate(...)


predicted_output = model.predict(input_fn=test_input_fn)
predicted_output = list(predicted_output)
print(predicted_output[:10])
count=0
for v in y_test:
    if count<10 :
        print(v)
        count=count+1
    else :
        break
    
"""
