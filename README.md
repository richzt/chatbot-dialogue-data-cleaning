# chatbot-dialogue-data-cleaning
Dialogue data cleaning

the data is coming from xiaohuangji50w_fenciA.conv. I filter it with condition that sentence length >= 2. So the surplus data set is 13W+ lines.
 
At the first, you need divide the train set and validation set with:
python /data/get_qa.py
 
Then, you can run as:
python train.py

when the model is trained, you can evaluate all the origin data with:
python eval.py

In the end, setting a threshold to get high-level score dialogue part with:
python /data/filter_origin.py

You can use the remaining data from the last round to retrain the model.
After several iterations, the quality of data can be improved.

In this project, the environment as fellows:
python3.5
jieba
glove embedding
tensorflow 1.4

reference:
https://github.com/dennybritz/chatbot-retrieval
