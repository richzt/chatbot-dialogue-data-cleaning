# encoding:utf-8
import tensorflow as tf
import numpy as np
import os
import time
import logging
from bi_lstm_model import Bi_Lstm_Model
import data_helpers
from tensorflow.contrib import learn
import pickle


tf.flags.DEFINE_integer('batch_size', 50, 'the batch_size of the training procedure')
tf.flags.DEFINE_float('valid_sample_percentage', 0.95, 'valid_data divide from data_set')
tf.flags.DEFINE_float('lr', 0.5, 'the learning rate')
tf.flags.DEFINE_float('threshold', 0.5, 'the learning rate')
tf.flags.DEFINE_float('lr_decay', 1/1.15, 'the learning rate decay')
tf.flags.DEFINE_integer('num_step', 20, 'sentence len')
tf.flags.DEFINE_integer('emdedding_dim', 100, 'embedding dim')
tf.flags.DEFINE_integer('iteration', 1, 'iteration times')
tf.flags.DEFINE_integer('hidden_neural_size', 256, 'LSTM hidden neural size')
tf.flags.DEFINE_integer('hidden_layer_num', 1, 'LSTM hidden layer num')
tf.flags.DEFINE_integer('num_checkpoints', 5, 'epoch num of checkpoint')
tf.flags.DEFINE_integer('check_point_every', 1000, 'checkpoint every num epoch')   # 1000
tf.flags.DEFINE_float('init_scale', 0.04, 'init scale')
tf.flags.DEFINE_integer("l2_reg_lambda", 0.01, "l2 regulation")
tf.flags.DEFINE_float('keep_prob', 0.5, 'dropout rate')
tf.flags.DEFINE_integer('num_epoch', 10, 'num epoch')
tf.flags.DEFINE_integer("evaluate_every", 100, "run evaluation")
tf.flags.DEFINE_integer('max_decay_epoch', 15, 'num epoch')
tf.flags.DEFINE_integer('max_grad_norm', 10, 'max_grad_norm')
tf.flags.DEFINE_string('log_file', "train.log", 'log_file')
tf.flags.DEFINE_string('word2vec_path', ".\\data\\word2vec.txt", 'ckpt_path')
# tf.flags.DEFINE_string('ckpt_path', None, 'ckpt_path')
tf.flags.DEFINE_string('ckpt_path', ".\\runs\\1526312711", 'ckpt_path')

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


class Config(object):
    valid_sample_percentage = FLAGS.valid_sample_percentage
    hidden_neural_size = FLAGS.hidden_neural_size
    num_step = FLAGS.num_step
    embed_dim = FLAGS.emdedding_dim
    iteration = FLAGS.iteration
    hidden_layer_num = FLAGS.hidden_layer_num
    keep_prob = FLAGS.keep_prob
    threshold = FLAGS.threshold
    lr = FLAGS.lr
    lr_decay = FLAGS.lr_decay
    init_scale = FLAGS.init_scale
    l2_reg_lambda = FLAGS.l2_reg_lambda
    batch_size = FLAGS.batch_size
    max_grad_norm = FLAGS.max_grad_norm
    num_epoch = FLAGS.num_epoch
    max_decay_epoch = FLAGS.max_decay_epoch
    checkpoint_every = FLAGS.check_point_every
    evaluate_every = FLAGS.evaluate_every
    num_checkpoints = FLAGS.num_checkpoints
    allow_soft_placement = True
    log_device_placement = False
    sentense_length = 50
    ckpt_path = FLAGS.ckpt_path
    word2vec_path = FLAGS.word2vec_path
    log_file = FLAGS.log_file


def create_model(session, y, vocab, config, path, logger):
    # create model, reuse parameters if exists
    initializer = tf.random_uniform_initializer(-1 * config.init_scale, 1 * config.init_scale)  # between(-1, 1)
    with tf.variable_scope("bi_rnn", reuse=None, initializer=initializer):
        bi_rnn = Bi_Lstm_Model(config=config,
                               num_step=config.num_step,
                               num_classes=1,
                               vocab_size=len(vocab),
                               is_training=0)   # 0 train, 1 valid, 2 predict
    with tf.variable_scope("bi_rnn", reuse=True, initializer=initializer):
        valid_bi_rnn = Bi_Lstm_Model(config=config,
                                     num_step=config.num_step,
                                     num_classes=1,
                                     vocab_size=len(vocab),
                                     is_training=1)
        test_bi_rnn = Bi_Lstm_Model(config=config,
                                    num_step=config.num_step,
                                    num_classes=1,
                                    vocab_size=len(vocab),
                                    is_training=1)
    if path:
        ckpt = tf.train.get_checkpoint_state(os.path.join(path, "checkpoints"))
        if tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            bi_rnn.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        emb_weights = session.run(bi_rnn.embeddings.read_value())
        emb_weights = data_helpers.load_word2vec(config.word2vec_path, vocab, config.embed_dim, emb_weights)
        session.run(bi_rnn.embeddings.assign(emb_weights))
        logger.info("Load pre-trained embedding.")
    return bi_rnn, valid_bi_rnn, test_bi_rnn


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def print_config(FLAGS, logger):
    """
    Print configuration of the model
    """
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        logger.info("{}={}".format(attr.upper(), value))


def run_epoch(bi_rnn, session, data, train_summary_writer):
    context_len = []
    for line in data[0]:
        counter = data_helpers.clear_zero_in_list(line.tolist())
        context_len.append(counter)
    utterance_len = []
    for line in data[1]:
        counter = data_helpers.clear_zero_in_list(line.tolist())
        utterance_len.append(counter)
    feed_dict = {
        bi_rnn.context: data[0],
        bi_rnn.context_len: context_len,
        bi_rnn.utterance: data[1],
        bi_rnn.utterance_len: utterance_len,
        bi_rnn.input_y: data[2]
    }

    # bi_rnn.assign_new_batch_size(session, len(data[0]))
    fetches = [bi_rnn.global_step, bi_rnn.cost, bi_rnn.accuracy, bi_rnn.train_op, bi_rnn.train_summary_op]

    global_step, cost, accuracy, _, train_summary = session.run(fetches, feed_dict)
    # print("step {}, loss {:g}, acc {:g}".format(steps, cost, accuracy))

    train_summary_writer.add_summary(train_summary, global_step)
    train_summary_writer.flush()
    return global_step, cost, accuracy


def evaluate(bi_rnn, session, data, steps, dev_summary_writer, dev_summary_op):
    costs = []
    accuracy_total = []
    for (x, xp, y) in data_helpers.batch_iter(data, batch_size=FLAGS.batch_size):
        fetches = [bi_rnn.cost, bi_rnn.accuracy, dev_summary_op]
        context_len = []
        for line in x:
            counter = data_helpers.clear_zero_in_list(line.tolist())
            context_len.append(counter)
        utterance_len = []
        for line in xp:
            counter = data_helpers.clear_zero_in_list(line.tolist())
            utterance_len.append(counter)

        feed_dict = {
            bi_rnn.context: x,
            bi_rnn.context_len: context_len,
            bi_rnn.utterance: xp,
            bi_rnn.utterance_len: utterance_len,
            bi_rnn.input_y: y
        }

        cost, accuracy, dev_summary = session.run(fetches, feed_dict)
        costs.append(cost)
        accuracy_total.append(accuracy)

        if dev_summary_writer:
            dev_summary_writer.add_summary(dev_summary, steps)
            dev_summary_writer.flush()
    return np.average(costs), np.average(accuracy_total)


def test_evaluate(bi_rnn, session, data):
    costs = []
    accuracy_total = []

    for (x, xp, y) in data_helpers.batch_iter(data, batch_size=FLAGS.batch_size):
        fetches = [bi_rnn.cost, bi_rnn.accuracy]
        context_len = []
        for line in x:
            counter = data_helpers.clear_zero_in_list(line.tolist())
            context_len.append(counter)
        utterance_len = []
        for line in xp:
            counter = data_helpers.clear_zero_in_list(line.tolist())
            utterance_len.append(counter)

        feed_dict = {
            bi_rnn.context: x,
            bi_rnn.context_len: context_len,
            bi_rnn.utterance: xp,
            bi_rnn.utterance_len: utterance_len,
            bi_rnn.input_y: y
        }

        cost, accuracy = session.run(fetches, feed_dict)
        costs.append(cost)
        accuracy_total.append(accuracy)

    return np.average(costs), np.average(accuracy_total)


def train_step():
    print("loading the dataset...")
    config = Config()
    log_path = os.path.join("logs", config.log_file)
    logger = get_logger(log_path)
    print_config(FLAGS, logger)

    x, xp, y = data_helpers.load_train_sentences("./data/train.csv")
    x_test, xp_test, y_test = data_helpers.load_train_sentences("./data/test.csv")

    # Build vocabulary
    if config.ckpt_path:
        print(os.path.join(config.ckpt_path, "vocab_processor.bin"))
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(
            os.path.join(config.ckpt_path, "vocab_processor.bin"))
    else:
        vocab_processor = learn.preprocessing.VocabularyProcessor(config.num_step, min_frequency=2)
        sentences = x + xp
        vocab_processor.fit_transform(sentences)  # word to id
        if not os.path.exists(config.word2vec_path):
            word2vec_model = data_helpers.dataMatrix(sentences, config.embed_dim, config.iteration)
            word2vec_model.wv.save_word2vec_format(os.path.join(config.word2vec_path), binary=False)

    # word to id
    x_train, xp_train, y = data_helpers.filter_train_list(list(vocab_processor.transform(x)),
                                                          list(vocab_processor.transform(xp)), y)
    x_test, xp_test, y_test = data_helpers.filter_train_list(list(vocab_processor.transform(x_test)),
                                                             list(vocab_processor.transform(xp_test)), y_test)

    vocab_id2w = dict()
    for index in range(len(vocab_processor.vocabulary_)):
        vocab_id2w[index] = vocab_processor.vocabulary_._reverse_mapping[index]
    logger.info("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))

    # test_data
    test_data = data_helpers.load_train_data(x_test, xp_test, y_test)
    # Training
    # ==================================================
    print("begin training")
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=config.allow_soft_placement,
            log_device_placement=config.log_device_placement)
        session_conf.gpu_options.allow_growth=True
        with tf.Session(config=session_conf).as_default() as session:
            if config.ckpt_path:
                out_dir = config.ckpt_path
            else:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            bi_rnn, valid_bi_rnn, test_bi_rnn = create_model(session, y, vocab_id2w, config, config.ckpt_path, logger)

            # add summary
            dev_summary_op = tf.summary.merge([valid_bi_rnn.loss_summary, valid_bi_rnn.accuracy_summary])
            # dev_summary_op = tf.summary.merge([valid_bi_rnn.loss_summary])

            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)

            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, session.graph)

            # add checkpoint
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

            checkpoint_prefix = os.path.join(checkpoint_dir, "bi_rnn")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            # Write vocabulary
            if config.ckpt_path is None:    # not exist
                vocab_processor.save(os.path.join(out_dir, "vocab_processor.bin"))
            begin_time = int(time.time())

            for epoch in range(config.num_epoch):
                logger.info("epoch in:" + str(epoch+1) + "/" + str(config.num_epoch))
                # learning_rate衰减
                # 在遍数小于max_epoch时,lr_decay = 1;> max_epoch时,lr_decay = 0.5^(epoch-max_epoch)
                lr_decay = config.lr_decay ** max(epoch-config.max_decay_epoch, 0.0)
                bi_rnn.assign_new_lr(session, config.lr*lr_decay)

                shuffle_indices = np.random.permutation(np.arange(len(y)))  # 随机打乱索引
                x_shuffled = x_train[shuffle_indices]
                xp_shuffled = xp_train[shuffle_indices]
                y_shuffled = y[shuffle_indices]
                # Split train/test set
                # TODO: This is very crude, should use cross-validation
                dev_sample_index = int(config.valid_sample_percentage * float(len(y)))
                x_train1, x_valid1 = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]  # attention circle
                xp_train1, xp_valid1 = xp_shuffled[:dev_sample_index], xp_shuffled[dev_sample_index:]
                y_train1, y_valid1 = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
                logger.info("Train/Valid split: {:d}/{:d}".format(len(y_train1), len(y_valid1)))

                train_data = data_helpers.load_train_data(x_train1, xp_train1, y_train1)
                valid_data = data_helpers.load_train_data(x_valid1, xp_valid1, y_valid1)

                batches = data_helpers.batch_iter(train_data, batch_size=config.batch_size)
                for batch in batches:
                    global_steps, train_cost, train_accuracy = run_epoch(bi_rnn, session, batch, train_summary_writer)
                    if global_steps % config.evaluate_every == 0:
                        valid_cost, valid_accuracy = evaluate(valid_bi_rnn, session, valid_data, global_steps,
                                                              dev_summary_writer,
                                                              dev_summary_op)
                        logger.info("step {:d}, train_cost {:g}, train_accuracy {:g}, valid_cost {:g}, "
                                    "valid_accuracy {:g}".format(global_steps, train_cost, train_accuracy,
                                                                 valid_cost, valid_accuracy))

                    if global_steps % config.checkpoint_every == 0:  # 多少轮保存一次
                        path = bi_rnn.saver.save(session, checkpoint_prefix, global_step=global_steps)
                        logger.info("Saved bi_rnn chechpoint to:{}\n".format(path))
                embeddings = session.run(bi_rnn.embeddings)
                pickle.dump(embeddings.tolist(), open(os.path.join(out_dir, "vocab.pkl"), "wb"))
                test_costs, test_accuracy = test_evaluate(test_bi_rnn, session, test_data)
                logger.info("the test data, test_costs{:g}, acc {:g}".format(test_costs, test_accuracy))
            end_time = int(time.time())
            logger.info("training takes %d seconds already\n" % (end_time-begin_time))
            print("program end!")


if __name__ == "__main__":
    train_step()
