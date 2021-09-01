# 数据预处理
## 原始数据

data/question_data.csv

```
question	label
结核性胸膜炎能吃桃子吗	medical
有没有脂肪肝怎么查	medical
高血压要终身吃药吗	medical
痛风能喝奶吗	medical
发烧寒颤怎么办	medical
肌肉收缩性头痛怎么办？	medical
二型糖尿病能好么	medical
肱骨肿瘤切除及重建手术做完已经四个月胳膊	medical
儿童急性咽喉炎地塞米松能不能用	medical
```

## 预处理
```
python data_preprocess.py
```

- 标签值 save_model/label_dict.json
```
{"__label__0":"medical", "__label__1":"nonmedical"}
```

- 训练集 data/seg.train
```
__label__1	赞比亚 首都 是
__label__1	我们 通常 用 哪道 菜名 形容 被 老板 辞退 解雇 开除
__label__1	钢丝 是 哪位 艺人 粉丝 最 常用 称呼
__label__0	癫痫 发作 早期 都 有 什么 表现
__label__1	乔 厂长 上任 记 作者
__label__1	诸葛亮 隐居 隆中 时常 自比 管仲 哪位 历史 人物
__label__0	法令 纹要 怎么 消除
__label__0	孕妇 甲减 可以 吃 黄豆 吗
__label__0	蒲公英 根 泡水 喝能治 慢性 咽喉炎 吗
__label__0	血压高 有点 头晕 像 没 睡 好 一样
...
```

- 测试集 data/seg.test

# 训练
```
python train.py
```

- 测试结果
```
time of train: 0:00:01
test: precision  0.975254730713246 , recall  0.975254730713246 , f-score  0.975254730713246
```

- 模型保存 save_model/model.bin

# 预测
```
python predict.py predict/input.txt predict/output.txt
```
- 输入 predict/input.txt
```
心脏不好可以做剧烈运动吗？
周杰伦的英文名是什么？
...
```

- 输出 predict/output.txt
```
medical
nonmedical
...
```

- 模型性能
```
number of questions: 100000
time of predict: 0:00:08
```
# 模型调用

fliter.py
```
class QueryFilter(path):
	'''
	path:str,模型保存路径,path/model.bin为模型,path/label_dict为标签值
	'''
	
	def filter(query):
		'''
        query:
            str,查询语句
            list,查询语句列表
        return:
        	query:医疗问题
        	False:非医疗问题
    	'''
```

example:
```
Q = QueryFilter('save_model')
query1 = '糖尿病患者可以吃水果吗？'
result = Q.filter(query1)
print(result) if result else print("我只能回答医疗问题")
# 糖尿病患者可以吃水果吗？
query2 = ['我叫什么名字？', '糖尿病患者可以吃水果吗？']
result = Q.filter(query2)
print([q if q else "我只能回答医疗问题" for q in result])
# ['我只能回答医疗问题', '糖尿病患者可以吃水果吗？']
```
