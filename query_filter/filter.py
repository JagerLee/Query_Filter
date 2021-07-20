import json
import fasttext
import jieba
import re
import string
import os
from zhon.hanzi import punctuation


class QueryFilter:
    def __init__(self, model_path):
        '''
        path:str,模型保存路径,path/model.bin为模型,path/label_dict为标签值
        '''

        self.model = fasttext.load_model(os.path.join(model_path, 'model.bin'))
        with open(os.path.join(model_path, 'label_dict.json')) as f:
            self.label_dict = json.load(f)

    def _query_split(self, query):
        reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
        input = re.sub(reg, '', query)

        trantab = str.maketrans({key: None for key in string.punctuation})
        input = input.translate(trantab)
        for i in punctuation:
            input = input.replace(i, '')
        input = jieba.lcut(input)
        return ' '.join(input)

    def _query2input(self, query):
        if type(query) == str:
            return self._query_split(query)
        return [self._query_split(q) for q in query]

    def filter(self, query):
        '''
        query:
            str,查询语句
            list,查询语句列表
        return:
            query:医疗问题
        	False:非医疗问题
        '''
        q_type = type(query)
        if q_type != str and q_type != list:
            return 'query type error:' + str(q_type) + ', expect str or list'
        result = self.model.predict(self._query2input(query))
        if q_type == str:
            result1 = query if self.label_dict[result[0][0]] == 'medical' else False
        else:
            result1 = []
            for i, q in enumerate(result[0]):
                result1.append(query[i]) if self.label_dict[q[0]] == 'medical' else result1.append(False)

        return result1


if __name__ == '__main__':
    query1 = '糖尿病患者可以吃水果吗？'
    query2 = ['我叫什么名字？', '糖尿病患者可以吃水果吗？']
    Q = QueryFilter('save_model')
    result = Q.filter(query1)
    print(result) if result else print("我只能回答医疗问题")
    result = Q.filter(query2)
    print([q if q else "我只能回答医疗问题" for q in result])
