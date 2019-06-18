from torch.utils.data import Dataset
from prepro_lib import jieba_cut
from opencc import OpenCC
import numpy as np
import torch

cc = OpenCC('tw2s')

'''
x[0] = array([1436181, 1292903, 1292676, 1292658, 1292632, 1292617, 1294329,
         1295560, 1292617, 1292938, 1292671, 1436181, 1292735, 1292609,
               0,       0,       0,       0,       0,       0,       0,
               0,       0,       0,       0,       0,       0,       0,
               0,       0], dtype=int32)},

y[0] = {'極端化': 0,
  '災難性思考': 0,
  '讀心術': 0,
  '「應該」與「必須」的陳述': 0,
  '個人化': 0,
  '憂鬱情緒': 1,
  '焦慮情緒': 1,
  '正向情緒': 0,
  '治療性對話臨床關注：1非臨床關注：0': 1,}
'''


class SequenceDataset(Dataset):
    def __init__(
        self,
        label_names: list = None,
        max_len=0,
        w2id_dict: dict = None,
        data_info: dict = None,
        is_train: bool = True
    ):
        self.x = []
        self.y = []
        self.lens = []
        self.max_len = max_len

        for info in data_info:
            sentence = cc.convert(info['認知語句'])

            tokens = jieba_cut(sentence)

            sentence_idxs = torch.LongTensor(
                self.max_len).fill_(w2id_dict['PAD'])

            for i, w in enumerate(tokens):
                if i >= self.max_len:
                    break
                if w in w2id_dict.keys():
                    sentence_idxs[i] = w2id_dict[w]
                else:
                    sentence_idxs[i] = w2id_dict['OOV']

            s_len = len(tokens) if len(
                tokens) <= self.max_len else self.max_len

            self.x.append(sentence_idxs)
            label = [info[k] for k in label_names]  # label to binary [1,1,0]
            label_kind = int(''.join(str(l)
                                     for l in label), 2)  # binary to int
            self.y.append(label_kind)
            self.lens.append(s_len)

            if is_train:
                self._generate_random_data(tokens, label_kind, w2id_dict)

    def _generate_random_data(self, tokens, label_kind, w2id_dict):
        w2id_dict_len = len(w2id_dict)
        for i in range(50):
            sentence_idxs = np.random.randint(
                0, w2id_dict_len, size=self.max_len)

            for i, w in enumerate(tokens):
                if i >= self.max_len:
                    break
                if w in w2id_dict.keys():
                    sentence_idxs[i] = w2id_dict[w]
                else:
                    sentence_idxs[i] = w2id_dict['OOV']

            s_len = self.max_len
            self.x.append(torch.from_numpy(sentence_idxs))
            self.y.append(label_kind)
            self.lens.append(s_len)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.lens[index], self.y[index]
