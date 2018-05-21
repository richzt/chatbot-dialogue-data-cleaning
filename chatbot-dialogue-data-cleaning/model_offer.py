#! /usr/bin/env python
# encoding:utf-8
import jieba
import tensorflow as tf
import os

out_dir = os.path.abspath(os.path.curdir)


class tensor_model():
    def __init__(self):
        # Parameters
        self.ct = "1499238237"

        # print("\nEvaluating...\n")
        # Evaluation
        # ==================================================
        checkpoint_dir = out_dir + "/runs/" + self.ct + "/checkpoints"
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
                # Get the placeholders from the graph by name
                self.input_x = graph.get_operation_by_name("bi_rnn_1/input_x_1").outputs[0]
                self.predictions = graph.get_operation_by_name("bi_rnn_1/output_1/predictions").outputs[0]

    def check_chinese(self, check_str):               # unicode码，仅限汉字和大写字母
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False

    def segment(self, x):
        stopwords = [line.strip() for line in open(out_dir+"/data/sp.txt").readlines()]  # 停用词集
        jieba.load_userdict(out_dir+"/data/dl_user.dic")
        origin_item = x.replace("\n", "").replace(" ", "")  # 去掉句子中原本的空格和换行符
        cut_list = jieba.cut(origin_item, cut_all=False)
        word_list = []
        for w in cut_list:
            if self.check_chinese(w) or w.isupper():
                if w.encode('utf-8') not in stopwords:
                    word_list.append(w)
        if len(word_list) > 50:
            word_list = word_list[:50]
        return ' '.join(word_list)
