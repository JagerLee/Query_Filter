import pandas as pd
import re
import jieba
import json
import string
from zhon.hanzi import punctuation


def delete_punctuation(s):
    trantab = str.maketrans({key: None for key in string.punctuation})
    s = s.translate(trantab)
    for i in punctuation:
        s = s.replace(i, '')
    return s


def stop_words(path):
    with open(path, encoding='UTF-8') as f:
        return [l.strip() for l in f]


def save_label(df, path):
    label_set = set()
    for label in df['label']:
        label_set.add(label)
    label_dict = {}
    for i, label in enumerate(label_set):
        label_dict[label] = '__label__' + str(i)
    with open(path, 'w') as f:
        json_str = json.dumps(label_dict, ensure_ascii=False)
        f.write('%s\n' % json_str)


def train_test_split(df, test_size=.2, shuffle=True):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    train_num = df.shape[0] - int(test_size * df.shape[0])
    train_df = df[:train_num]
    test_df = df[train_num:]

    return train_df, test_df


def word_split(s, stop_words_list):
    reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
    tmp_token = re.sub(reg, '', s)
    tmp_token = delete_punctuation(tmp_token)
    tmp_token = jieba.lcut(tmp_token)
    content = [x for x in tmp_token if x not in stop_words_list]
    return content


def save_data(df, path):
    stop_words_list = stop_words('data/stopwords.txt')
    with open('data/label_dict.json', encoding='gb18030') as f:
        line = f.readline()
        label_dict = json.loads(line)
    with open(path, 'w', encoding='utf-8') as f:
        for i in df.index:
            label = label_dict[df['label'][i]]
            question = word_split(df['question'][i], stop_words_list)
            f.write(label + '\t')
            for i in range(len(question) - 1):
                f.write(question[i] + ' ')
            f.write(question[-1] + '\n')


if __name__ == '__main__':
    train_path = 'data/seg.train'
    test_path = 'data/seg.test'
    data_path = 'data/question_data.csv'
    label_path = 'data/label_dict.json'
    df = pd.read_csv(data_path, sep='\t', encoding='gbk', header=0)
    save_label(df, label_path)
    train_df, test_df = train_test_split(df)
    save_data(train_df, train_path)
    save_data(test_df, test_path)
