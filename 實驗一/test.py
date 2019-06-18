import torch
from dataset import SequenceDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from model import Model
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn import metrics

device = torch.device('cuda:0')


def sort_batch_data(data):
    xs, lens, ys = data
    sorted_lens, sorted_id = lens.sort(dim=0, descending=True)
    sorted_xs = xs[sorted_id]
    sorted_ys = ys[sorted_id]

    return sorted_xs, sorted_lens, sorted_ys


def test(model, data_loader, label_names):
    model.eval()
    n = 0
    acc = 0.0

    results = []
    ys = []
    with torch.no_grad():
        for data in tqdm(data_loader, ascii=True, total=len(data_loader)):
            n += 1
            x, l, y = sort_batch_data(data)
            x, y = x.to(device), y.to(device)

            outs = model(x, l)

            _, predict = torch.max(outs.detach(), 1)

            correct_count = ((predict == y.detach()).sum()
                             ).double()

            acc += correct_count.double() / y.size(0)

            ys.extend(y.cpu().detach().numpy())
            results.extend(predict.cpu().detach().numpy())

        print(f"Test Accuracy: {acc/n:.4f}")
        print(f"F1: {metrics.f1_score(ys,results,average='micro')}")
        print(metrics.classification_report(
            y_true=ys, y_pred=results))


if __name__ == "__main__":
    data = pd.read_excel('./test.xlsx')
    label_names = list(data.columns[3:])
    infos = data.to_dict(orient='records')

    with open('./emb_matrix.pickle', 'rb') as f:
        emb_matrix = pickle.load(f)

    with open('./w2id_dict.pickle', 'rb') as f:
        w2id_dict = pickle.load(f)

    test_dataset = SequenceDataset(
        label_names=label_names,
        max_len=30,
        w2id_dict=w2id_dict,
        data_info=infos,
        is_train=False
    )

    test_data_loader = DataLoader(
        dataset=test_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=2,
    )

    model = torch.load("./best.model")
    model.to(device)

    print(model)

    test(model, test_data_loader, label_names)
