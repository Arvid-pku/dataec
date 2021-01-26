import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, data_size, emb_size, hidden_size, out_size):
        """初始化参数：
            vocab_size:字典的大小
            emb_size:词向量的维数
            hidden_size：隐向量的维数
            out_size:标注的种类
        """
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.embeddingD = nn.Embedding(data_size, emb_size)

        self.bilstm = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)
        self.bilstmD = nn.LSTM(emb_size, hidden_size,
                              batch_first=True,
                              bidirectional=True)

        self.lin = nn.Linear(4*hidden_size, out_size)

    def forward(self, sents_tensor, datas_tensor, lengths, lengthsD):
        emb = self.embedding(sents_tensor)  # [B, L, emb_size]
        embD = self.embedding(datas_tensor)  # [B, L, emb_size]
        try:
            packed = pack_padded_sequence(emb, lengths, batch_first=True)
        except RuntimeError:
            print(lengths)
            raise
        packedD = pack_padded_sequence(embD, lengthsD, batch_first=True, enforce_sorted=False)

        rnn_out, _ = self.bilstm(packed)
        _, hiddnD = self.bilstmD(packedD) # [n*d, b, h]
        hiddnD = hiddnD[0]
        # print(hiddnD.size())
        hiddnD = torch.cat((hiddnD[-2], hiddnD[-1]), 1).unsqueeze(1)
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        # print(hiddnD.size())
        # print(rnn_out.size())
        hiddnD = hiddnD.expand(hiddnD.size()[0], rnn_out.size()[1], hiddnD.size()[2])
        rnn_out = torch.cat((rnn_out, hiddnD), 2)
# todo
        scores = self.lin(rnn_out)  # [B, L, out_size]

        return scores

    def test(self, sents_tensor, data_tensor, lengths, lengthsD, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(sents_tensor, data_tensor, lengths, lengthsD)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids
