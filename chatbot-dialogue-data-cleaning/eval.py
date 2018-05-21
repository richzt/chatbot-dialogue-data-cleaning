#! /usr/bin/env python
# encoding:utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import data_helpers

# Parameters
# ==================================================
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_string("ckpt_path", ".\\runs\\1526312711", "checkpoint_time part of Checkpoint directory")
tf.flags.DEFINE_integer('hidden_neural_size', 100, 'LSTM hidden neural size')
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

checkpoint_dir = os.path.join(FLAGS.ckpt_path, "checkpoints")

print("Loading data...")

# Load data from files
x, xp = data_helpers.load_predict_sentences("./data/origin5.csv")

# Build vocabulary
print(os.path.join(FLAGS.ckpt_path, "vocab_processor.bin"))
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
    os.path.join(FLAGS.ckpt_path, "vocab_processor.bin"))

x_dev, xp_dev, x_ch, xp_ch = data_helpers.filter_predict_list(vocab_processor, x, xp)
dev_data = data_helpers.load_predict_data(x_dev, xp_dev)
print("\nEvaluating...\n")
# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():

        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        Context = graph.get_operation_by_name("bi_rnn_1/context").outputs[0]
        Context_len = graph.get_operation_by_name("bi_rnn_1/context_len").outputs[0]
        Utterance = graph.get_operation_by_name("bi_rnn_1/utterance").outputs[0]
        Utterance_len = graph.get_operation_by_name("bi_rnn_1/utterance_len").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("bi_rnn_1/prediction/prediction").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter_eval(dev_data, FLAGS.batch_size, shuffle=False)
        # Collect the predictions here
        all_predictions = []

        for batch in batches:
            context_len = []
            for line in batch[0]:
                counter = data_helpers.clear_zero_in_list(line.tolist())
                context_len.append(counter)
            utterance_len = []
            for line in batch[1]:
                counter = data_helpers.clear_zero_in_list(line.tolist())
                utterance_len.append(counter)
            feed_dict = {
                Context: batch[0],
                Context_len: context_len,
                Utterance: batch[1],
                Utterance_len: utterance_len
            }
            batch_predictions = sess.run(predictions, feed_dict)
            batch_predictions = batch_predictions[:, 0]
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Save the evaluation to a csv
predictions_human_readable = pd.DataFrame({'0context': x_ch, '1utterance': xp_ch, '2probablity': all_predictions})
predictions_human_readable.rename(columns={'0context': "Context", '1utterance': "Utterance", '2probablity': "Label"}, inplace=True)
out_path = os.path.join("./data/origin_prob.csv")
print("Saving evaluation to {0}".format(out_path))
predictions_human_readable.to_csv(out_path, index=False, encoding="utf-8")
