# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 
# ==============================================================================

# CREDITS: https://github.com/tensorflow/models/blob/master/research/object_detection/model_main_tf2.py
# I edited that file to work when imported rather than when executed within a terminal

import tensorflow.compat.v2 as tf
from object_detection import model_lib_v2


def model_main_tf2(
    pipeline_config_path=None,
    num_train_steps=None,
    eval_on_train_data=False,
    sample_1_of_n_eval_examples=None,
    sample_1_of_n_eval_on_train_examples=5,
    model_dir=None,
    checkpoint_dir=None,
    eval_timeout=3600,
    use_tpu=False,
    tpu_name=None,
    num_workers=1,
    checkpoint_every_n=1000,
    record_summaries=True):
  tf.config.set_soft_device_placement(True)

  if checkpoint_dir:
    model_lib_v2.eval_continuously(
        pipeline_config_path=pipeline_config_path,
        model_dir=model_dir,
        train_steps=num_train_steps,
        sample_1_of_n_eval_examples=sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(sample_1_of_n_eval_on_train_examples),
        checkpoint_dir=checkpoint_dir,
        wait_interval=300, timeout=eval_timeout)
  else:
    if use_tpu:
      # TPU is automatically inferred if tpu_name is None and
      # we are running under cloud ai-platform.
      resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu_name)
      tf.config.experimental_connect_to_cluster(resolver)
      tf.tpu.experimental.initialize_tpu_system(resolver)
      strategy = tf.distribute.experimental.TPUStrategy(resolver)
    elif num_workers > 1:
      strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    else:
      strategy = tf.compat.v2.distribute.MirroredStrategy()

    with strategy.scope():
      model_lib_v2.train_loop(
          pipeline_config_path=pipeline_config_path,
          model_dir=model_dir,
          train_steps=num_train_steps,
          use_tpu=use_tpu,
          checkpoint_every_n=checkpoint_every_n,
          record_summaries=record_summaries)
