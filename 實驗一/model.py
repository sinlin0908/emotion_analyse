import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    def __init__(
        self,
        vocab_size=0,
        emb_dim=0,
        num_classes=1,
        hidden_size=1,
        num_layers=1,
        dropout=0.5,
        weight_matrix=None
    ):
        super(Model, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        self.embedding_layer = nn.Embedding(
            self.vocab_size, self.emb_dim).from_pretrained(weight_matrix)

        self.bigru = nn.GRU(
            input_size=self.emb_dim,
            hidden_size=self.hidden_size//2,
            num_layers=self.num_layers,
            bidirectional=True,
            dropout=dropout,
        )

        self.output_layer = nn.Linear(
            in_features=self.hidden_size * 2,
            out_features=self.num_classes
        )  # num_class output layer

        for module in self.modules():
            if isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if name in ['weight']:
                        nn.init.orthogonal_(param)
                    elif name in ['bias']:
                        nn.init.constant_(param, 0)

            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, sorted_l):
        batch_size, seq_len = x.size()

        # batch_size * seq_len * emb_dim
        embedded = self.embedding_layer(x)

        # max_seq_len(sorted_len[0]) * batch_size
        packed = pack_padded_sequence(embedded.permute([1, 0, 2]), sorted_l)

        # seq * batch_size * input_size
        gru_out, gru_hidden = self.bigru(packed, None)

        # seq * batch_size * input_size
        pad_seq, _ = pad_packed_sequence(gru_out)

        encoding = torch.cat([pad_seq[0], pad_seq[-1]], dim=1)
        encoding = torch.relu(encoding)

        outputs = self.output_layer(encoding)

        # output = torch.sigmoid(output)

        return outputs

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size//2)
