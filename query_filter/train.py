import fasttext
import logging
import time
from datetime import timedelta


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    start_time = time.time()
    model = fasttext.train_supervised('data/seg.train', lr=0.1, epoch=50, wordNgrams=2)
    end_time = time.time()
    time_dif = end_time - start_time
    print('time of train:', timedelta(seconds=int(round(time_dif))))

    model.save_model("save_model/model.bin")

    num, precision, recall = model.test('data/seg.test')
    print('test:', 'precision ', precision, ', recall ', recall,
          ', f-score ', precision * recall * 2 / (recall + precision))