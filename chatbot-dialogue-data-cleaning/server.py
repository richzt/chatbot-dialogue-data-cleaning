#!/usr/bin/env python
# -*-coding: utf-8-*-

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.httpclient
import tornado.gen
import tornado.escape
import model_offer
import pandas as pd
import numpy as np
import os
from tensorflow.contrib import learn

from tornado.options import define, options
define("port", default=9000, help="run on the given port", type=int)

out_dir = os.path.abspath(os.path.curdir)
checkpoint_dir = out_dir + "/runs/" + "1499238237" + "/checkpoints"
rule_path = out_dir + "/data/rule_data_8.xlsx"
rule_data = pd.read_excel(rule_path, 'Sheet1', encoding='gbk')
t_label = rule_data['category']
label_dic = {}.fromkeys(t_label).keys()

tm = model_offer.tensor_model()


class MainHandler(tornado.web.RequestHandler):

    def get(self):
        sentence = self.get_argument('q', " ")
        r_set = pd.DataFrame({"question": [tm.segment(sentence)]})
        x_raw = r_set['question']
        # Map data into vocabulary
        vocab_path = os.path.join(checkpoint_dir, "..", "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        x_test = np.array(list(vocab_processor.transform(x_raw)))
        batch_predictions = tm.sess.run(tm.predictions, {tm.input_x: x_test})

        self.write(label_dic[batch_predictions[0]])

        # respon = {'answer': label_dic[batch_predictions[0]]}
        # respon_json = tornado.escape.json_encode(respon)
        # self.write(respon_json)


if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r"/", MainHandler)])
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
