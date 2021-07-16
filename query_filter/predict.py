from datetime import timedelta
import fasttext
import json
import sys
import time
from data_preprocess import stop_words, word_split

if __name__ == '__main__':
    start_time = time.time()

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model = fasttext.load_model('save_model/trained_model.bin')
    stop_words_list = stop_words('data/stopwords.txt')
    with open(r'data/label_dict.json') as f:
        label_dict = json.load(f)
    with open(input_path, encoding='utf-8') as f:
        question_list = [l.strip() for l in f]
    with open(output_path, 'w') as f:
        for question in question_list:
            word_list = word_split(question, stop_words_list)
            label = model.predict(' '.join(word_list))
            for key in label_dict.items():
                if key[1] == label[0][0]:
                    f.write(key[0] + '\n')

    end_time = time.time()
    time_dif = end_time - start_time
    print('number of questions:', len(question_list))
    print('time of predict:', timedelta(seconds=int(round(time_dif))))
