
# Define the training algorithm for the Trainer module file
import os
from typing import List, Text

import tensorflow as tf
from tensorflow import keras

from tfx import v1 as tfx
from tfx_bsl.public import tfxio

from tensorflow_metadata.proto.v0 import schema_pb2

# Features used for classification - culmen length and depth, flipper length,
# body mass, and species.

_LABEL_KEY = 'Regime'

_FEATURE_KEYS = [
    'high', 'low', 'open', 'Volume','close'
]


def _input_fn(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema, batch_size: int) -> tf.data.Dataset:
  return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=_LABEL_KEY), schema).repeat()


def _build_keras_model():
  inputs = [keras.layers.Input(shape=(1,), name=f) for f in _FEATURE_KEYS]
  d = keras.layers.concatenate(inputs)
  d = keras.layers.Dense(8, activation='relu')(d)
  d = keras.layers.Dense(8, activation='relu')(d)
  outputs = keras.layers.Dense(3)(d)
  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(1e-2),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[keras.metrics.SparseCategoricalAccuracy()])
  return model


def run_fn(fn_args: tfx.components.FnArgs):
  schema = schema_pb2.Schema()
  tfx.utils.parse_pbtxt_file(fn_args.schema_path, schema)
  train_dataset = _input_fn(
      fn_args.train_files, fn_args.data_accessor, schema, batch_size=10)
  eval_dataset = _input_fn(
      fn_args.eval_files, fn_args.data_accessor, schema, batch_size=10)
  model = _build_keras_model()
  model.fit(
      train_dataset,
      epochs=int(fn_args.train_steps / 20),
      steps_per_epoch=20,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps)
  model.save(fn_args.serving_model_dir, save_format='tf')
