import os
import sys
import torch
from prepro_lib import jieba_cut
from opencc import OpenCC
import pickle
import copy

cc = OpenCC('tw2s')

device = torch.device('cuda:0')


class System:
    def __init__(
            self,
            max_len=30,
            model_path=None,
            emb_matrix=None,
            w2id_dict=None
    ):
        if not os.path.exists(model_path):
            print("model path is not exist..")
            sys.exit(1)

        self.max_len = max_len

        self.emb_matrix = emb_matrix

        self.model_path = model_path

        self.w2id_dict = w2id_dict

    def test(self, in_):
        if not in_:
            raise ValueError("input is empty")

        x, x_len = self._prepro(in_)
        predicts = 0.0

        model = torch.load(self.model_path)
        with torch.no_grad():

            x = x.to(device)

            outs = model(x, x_len)

            _, predict = torch.max(outs.detach(), 1)

            answer = predict.item()

            return answer

    def _prepro(self, s):
        s = cc.convert(s)
        s_tokens = jieba_cut(s)
        s_len = torch.LongTensor([len(s_tokens)])

        sent_idxs = torch.LongTensor(
            self.max_len).fill_(self.w2id_dict['PAD'])

        for i, w in enumerate(s_tokens):
            if i >= self.max_len:
                break
            if w in self.w2id_dict.keys():
                sent_idxs[i] = self.w2id_dict[w]
            else:
                sent_idxs[i] = self.w2id_dict['OOV']

        return sent_idxs.view(1, self.max_len), s_len


if __name__ == "__main__":
    with open('./emb_matrix.pickle', 'rb') as f:
        emb_matrix = pickle.load(f)

    with open('./w2id_dict.pickle', 'rb') as f:
        w2id_dict = pickle.load(f)

    label = ['極端化', '災難性思考', '讀心術', '「應該」與「必須」的陳述', '個人化',
             '憂鬱情緒', '焦慮情緒', '正向情緒', '治療性對話臨床關注：1非臨床關注：0']

    system = System(
        model_path='./best.model',
        emb_matrix=emb_matrix,
        w2id_dict=w2id_dict
    )

    while True:
        input_ = input("in>")
        if input_ == '0':
            break
        predicts = system.test(input_)

        answer = list("{0:09b}".format(predicts))

        print(label)
        print(answer)

        table = dict(zip(label, answer))

        # rank = sorted(, key=lambda kv: kv[1], reverse=True)

        print("\n\nRank")
        for r in table.items():

            l, value = r
            print(f"{l} : {value}")
