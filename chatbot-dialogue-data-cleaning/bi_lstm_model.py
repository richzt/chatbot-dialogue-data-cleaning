# encoding:utf-8
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers


class Bi_Lstm_Model(object):

    def __init__(self, config, num_step, num_classes, vocab_size, is_training=0):

        self.context = tf.placeholder(tf.int32, [None, num_step], name="context")
        self.context_len = tf.placeholder(tf.int32, [None], name="context_len")
        self.utterance = tf.placeholder(tf.int32, [None, num_step], name="utterance")
        self.utterance_len = tf.placeholder(tf.int32, [None], name="utterance_len")

        self.global_step = tf.Variable(0, trainable=False)
        self.initializer = initializers.xavier_initializer()

        with tf.device("/cpu:0"), tf.name_scope("embedding_layer"):
            # embedding layer
            # Initialize embedidngs randomly or with pre-trained vectors if available
            self.embeddings = tf.get_variable(shape=[vocab_size, config.embed_dim],
                                              name="embeddings", initializer=self.initializer)  # (Vocabulary size),100
            # Embed the context and the utterance
            context_embedded = tf.nn.embedding_lookup(
                self.embeddings, self.context, name="embed_context")
            utterance_embedded = tf.nn.embedding_lookup(
                self.embeddings, self.utterance, name="embed_utterance")

        # with tf.name_scope("dropout_f"):
        #     if is_training:
        #         inputs = tf.nn.dropout(inputs, config.keep_prob)

        # Build the RNN
        with tf.variable_scope("rnn") as vs:
            # We use an LSTM Cell
            cell = tf.nn.rnn_cell.LSTMCell(
                config.hidden_neural_size,
                forget_bias=2.0,
                use_peepholes=True,
                state_is_tuple=True)

            aa = tf.concat([context_embedded, utterance_embedded], 0)
            # Run the utterance and context through the RNN
            rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                cell,
                tf.concat([context_embedded, utterance_embedded], 0),
                sequence_length=tf.concat([self.context_len, self.utterance_len], 0),
                dtype=tf.float32)
            encoding_context, encoding_utterance = tf.split(rnn_states.h, 2, 0)

        with tf.variable_scope("prediction") as vs:
            M = tf.get_variable("M", shape=[config.hidden_neural_size, config.hidden_neural_size],
                                initializer=tf.truncated_normal_initializer())

            with tf.name_scope("output"):
                # "Predict" a  response: c * M
                generated_response = tf.matmul(encoding_context, M)
                generated_response = tf.expand_dims(generated_response, 2)
                encoding_utterance = tf.expand_dims(encoding_utterance, 2)

                # Dot product between generated response and actual response
                # (c * M) * r
                logits = tf.matmul(generated_response, encoding_utterance, True)
                logits = tf.squeeze(logits, [2])

            # Apply sigmoid to convert logits to probabilities
            self.prediction = tf.sigmoid(logits, name="prediction")

            # saver of the model
            self.saver = tf.train.Saver(tf.global_variables(), reshape=True, max_to_keep=5)

            if is_training == 2:
                return

            self.input_y = tf.placeholder(tf.int32, [None, num_classes], name="target")  # target

            # add summary
            with tf.name_scope("accuracy"):
                # threshold with prediction
                y = tf.constant(config.threshold)
                t = tf.fill(tf.shape(self.prediction), 1)
                f = tf.fill(tf.shape(self.prediction), 0)
                prediction_normal = tf.where(tf.greater(self.prediction, y), t, f)
                correct_prediction = tf.equal(prediction_normal, self.input_y)
                self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

            with tf.name_scope("loss"):
                # Calculate the binary cross-entropy loss
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.to_float(self.input_y))

                # Mean loss across the batch of examples
                self.cost = tf.reduce_mean(losses, name="mean_loss")

        self.loss_summary = tf.summary.scalar("loss", self.cost)
        # add summary
        self.accuracy_summary = tf.summary.scalar("accuracy_summary", self.accuracy)

        if is_training == 1:
            return

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), config.max_grad_norm)
        grad_summaries = []
        for g, v in zip(grads, tvars):
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        self.grad_summaries_merged = tf.summary.merge(grad_summaries)

        self.train_summary_op = tf.summary.merge(
            [self.loss_summary, self.accuracy_summary, self.grad_summaries_merged])
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        # optimizer = tf.train.AdamOptimizer(self.lr)
        optimizer.apply_gradients(zip(grads, tvars))

        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self.lr, self._new_lr)

    def assign_new_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})
