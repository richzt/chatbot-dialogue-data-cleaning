# encoding:utf-8
import pandas as pd
import numpy as np
import jieba


def build_distractor(data, real_answer):
    for col in range(9):
        dis = "Distractor_"+str(col)
        distractor_n = list()
        for n1 in range(len(data)):
            n_i = np.random.randint(0, len(real_answer), 1)[0]
            distractor_n.append(real_answer[n_i])
        data[dis] = distractor_n


origin = pd.read_csv("origin.csv", encoding="utf-8")
question = list(origin["Context"])
answer = list(origin["Utterance"])
real_answer = list(origin["Utterance"])
label = [1] * len(question)   # true


for i in range(len(question)):
    q_i = np.random.randint(0, len(question), 1)[0]
    a_i = np.random.randint(0, len(question), 1)[0]
    question.append(question[q_i])
    real_answer.append(answer[q_i])
    answer.append(answer[a_i])
    label.append(0)    # false

shuffle_indices = np.random.permutation(np.arange(len(label)))  # 随机打乱索引
question = np.asarray(question)
question_shuffle = question[shuffle_indices]
answer = np.asarray(answer)
answer_shuffle = answer[shuffle_indices]
real_answer = np.asarray(real_answer)
real_answer_shuffle = real_answer[shuffle_indices]
label = np.asarray(label)
label_shuffle = label[shuffle_indices]

data = pd.DataFrame({"0Context": question_shuffle, "1Utterance": answer_shuffle, "3Label": label_shuffle,
                     "2Real_Utterance": real_answer_shuffle})
data.to_csv("data.csv", index=False, encoding="utf-8")

seg1 = int(len(label)*0.95)
print(seg1)

train_data = data.iloc[: seg1]
train_data = train_data[["0Context", "1Utterance", "3Label"]]
train_data.rename(columns={"0Context": "Context", "1Utterance": "Utterance", "3Label": "Label"}, inplace=True)
train_data.to_csv("train.csv", index=False, encoding="utf-8")

# valid_data = data.iloc[seg1:]
# valid_data = valid_data[["0Context", "2Real_Utterance"]]
# valid_data.rename(columns={"0Context": "Context", "2Real_Utterance": "Ground Truth Utterance"}, inplace=True)
# build_distractor(valid_data, real_answer)
# valid_data.to_csv("test.csv", index=False, encoding="utf-8")
valid_data = data.iloc[seg1:]
valid_data = valid_data[["0Context", "2Real_Utterance"]]
valid_data["3Label"] = [1] * len(valid_data)
valid_data.rename(columns={"0Context": "Context", "2Real_Utterance": "Utterance", "3Label": "Label"}, inplace=True)
valid_data.to_csv("test.csv", index=False, encoding="utf-8")
