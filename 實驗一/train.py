import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import SequenceDataset
from model import Model
import pandas as pd
import pickle
from tqdm import tqdm
import argparse


torch.cuda.manual_seed(0)
torch.manual_seed(0)
device = torch.device('cuda:0')


def sort_batch_data(data):
    xs, lens, ys = data
    sorted_lens, sorted_id = lens.sort(dim=0, descending=True)
    sorted_xs = xs[sorted_id]
    sorted_ys = ys[sorted_id]

    return sorted_xs, sorted_lens, sorted_ys


def train(model, train_data_loader):
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    epochs = 20

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}/{epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{epochs}'))

        epoch_loss = 0.0
        acc = 0.0
        count = 0

        for step, data in tqdm(enumerate(train_data_loader),
                               total=len(train_data_loader),
                               ascii=True):

            optimizer.zero_grad()
            count += 1
            x, l, y = sort_batch_data(data)
            x, y = x.to(device), y.to(device)
            outs = model(x, l)

            _, predict = torch.max(outs.detach(), 1)

            batch_loss = criterion(outs, y)
            batch_loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()

            correct_count = torch.sum(predict == y.detach())

            acc += correct_count.double() / y.size(0)

            epoch_loss += batch_loss.item() / y.size(0)

        epoch_loss = epoch_loss / count
        acc = acc/count

        print(
            f'Training loss: {epoch_loss:.4f} Acc: {acc:.4f}\n')

    print("save model......")
    torch.save(model, 'best.model')


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--pretrained", type=str)

    args = parse.parse_args()

    data = pd.read_excel('./data.xlsx')
    label_names = list(data.columns[2:])

    data = data[(data[label_names] != 0).any(1)]  # 砍掉通0

    print("label names\n", label_names)
    infos = data.to_dict(orient='records')

    with open('./emb_matrix.pickle', 'rb') as f:
        emb_matrix = pickle.load(f)

    with open('./w2id_dict.pickle', 'rb') as f:
        w2id_dict = pickle.load(f)

    train_dataset = SequenceDataset(
        label_names=label_names,
        max_len=30,
        w2id_dict=w2id_dict,
        data_info=infos
    )

    train_data_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )

    print("Data loader size: ", len(train_data_loader))

    if args.pretrained == 'T':
        print("Pretrained...")
        model = torch.load("../pretrain/best.model")
        model.output_layer = nn.Linear(
            in_features=128 * 2,
            out_features=2**len(label_names)
        )
        model = model.to(device)
    else:
        model = Model(
            vocab_size=len(w2id_dict),
            emb_dim=300,
            num_classes=2**len(label_names),
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            weight_matrix=torch.FloatTensor(emb_matrix).to(device)
        ).to(device)

    print(model)

    train(model, train_data_loader)
