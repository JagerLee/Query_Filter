import json
import fasttext
import jieba
import re
import string
from zhon.hanzi import punctuation


class QueryFilter:
    def __init__(self, model_path):
        self.model = fasttext.load_model(model_path + '/model.bin')
        with open(model_path + '/label_dict.json') as f:
            self.label_dict = json.load(f)

    def _query2input(self, query):
        reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
        input = re.sub(reg, '', query)

        trantab = str.maketrans({key: None for key in string.punctuation})
        input = input.translate(trantab)
        for i in punctuation:
            input = input.replace(i, '')
        input = jieba.lcut(input)
        return ' '.join(input)

    def filter(self, query):
        result = self.model.predict(self._query2input(query))

        return query if self.label_dict[result[0][0]] == 'medical' else False


if __name__ == '__main__':
    query1 = '糖尿病患者可以吃水果吗？'
    query2 = '我叫什么名字？'
    Q = QueryFilter('save_model')
    result = Q.filter(query1)
    print(result) if result else print("not a medical query")
    result = Q.filter(query2)
    print(result) if result else print("not a medical query")