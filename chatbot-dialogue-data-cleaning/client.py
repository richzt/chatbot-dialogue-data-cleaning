# encoding:utf-8
import json
import requests


# #POST
# url = "http://192.168.46.69:9000"
# param = {'k': 'v'}
# req_dict = {'param': param}
# req_json = json.dumps(req_dict)
# req_post = req_json.encode('utf-8')
# headers = {'Content-Type': 'application/json'}
# req = requests.post(url, headers=headers, data=req_post).content

# #Request
string = "赶不赶        小脚丫↘冰凉       ~~红狐狸~~      知会光         平明人       费话不说        奇迹no3       同学门          奇迹no1           奇迹no2        完美富翁          卸货          辉煌づ永恒        ゞ╪嘵楓o殘月の         小小宏        爱我的人吗           蝴蝶花&amp;        #79颖颖#79        ☆風流★浪子☆          ◇◆浪子情深↘          这些人我都举报过了。为啥他们又出来在2*开挂了？？"
url = "http://192.168.46.75:9000/?q="+string
req = requests.get(url).content
print(req)
