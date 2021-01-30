import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

def getentity(batch_datalabels, data_out, device):
    dataentitys = []
    h2=128
    for datalabel, batch in zip(batch_datalabels, data_out):
        entityvec = torch.zeros(h2).to(device)
        st = False
        entitylen = 0
        onebatchentity = []
        for wdlb, wdvec in zip(datalabel, batch):
            if wdlb == 1:
                if st:
                    entityvec = entityvec/entitylen
                    onebatchentity.append(entityvec)
                    entityvec = torch.zeros(h2).to(device)
                    st = False
                    entitylen = 0
                st = True
            elif wdlb == 0:
                if st:
                    entityvec = entityvec/entitylen
                    onebatchentity.append(entityvec)
                    entityvec = torch.zeros(h2).to(device)
                    st = False
                    entitylen = 0
                    
            if st:
                entityvec += wdvec
                entitylen += 1
        dataentitys.append(onebatchentity)
    maxlen = max([len(x) for x in dataentitys])
    dataentitys = [x+[torch.zeros(h2).to(device)]*(maxlen-len(x)) for x in dataentitys]
    dataentitys = torch.stack([torch.stack(x) for x in dataentitys])
    return dataentitys


class Attention(nn.Module):
  def __init__(self, hidden_size, units):
    super(Attention, self).__init__()
    self.W1 = nn.Linear(2*hidden_size, units, bias=False)
    self.W2 = nn.Linear(2*hidden_size, units, bias=False)
    self.V =  nn.Linear(units, 1, bias=False)

  def forward(self, 
              encoder_out: torch.Tensor, 
              decoder_hidden: torch.Tensor):
    # encoder_out: (BATCH, ARRAY_LEN, HIDDEN_SIZE)
    # decoder_hidden: (BATCH, HIDDEN_SIZE)

    # Add time axis to decoder hidden state
    # in order to make operations compatible with encoder_out
    # decoder_hidden_time: (BATCH, 1, HIDDEN_SIZE)
    # print(decoder_hidden.size())
    decoder_hidden_time = decoder_hidden.unsqueeze(1)
    # print(decoder_hidden_time.size())
    # print(encoder_out.size())
    # uj: (BATCH, ARRAY_LEN, ATTENTION_UNITS)
    # Note: we can add the both linear outputs thanks to broadcasting
    uj = self.W1(encoder_out) + self.W2(decoder_hidden_time)
    uj = torch.tanh(uj)

    # uj: (BATCH, ARRAY_LEN, 1)
    uj = self.V(uj)

    # Attention mask over inputs
    # aj: (BATCH, ARRAY_LEN, 1)
    aj = F.softmax(uj, dim=1)

    
    return aj


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, data_size, emb_size, hidden_size):
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
        self.attention = Attention(hidden_size, 64)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # self.lin = nn.Linear(4*hidden_size, out_size)

    def forward(self, tensorized_sents, tensorized_datas, batch_sentlabels, batch_datalabels, lengths, lengthsD):
        emb = self.embedding(tensorized_sents)  # [B, L, emb_size]
        embD = self.embedding(tensorized_datas)  # [B, L, emb_size]

        packed = pack_padded_sequence(emb, lengths, batch_first=True)
        packedD = pack_padded_sequence(embD, lengthsD, batch_first=True, enforce_sorted=False)

        rnn_out, _ = self.bilstm(packed)
        data_out, _ = self.bilstmD(packedD) 
        # rnn_out:[B, L, hidden_size*2]
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        data_out, _ = pad_packed_sequence(data_out, batch_first=True)
        # print(rnn_out.size())
        # print(data_out.size())
        # print([len(x) for x in batch_sentlabels])
        # print([len(x) for x in batch_datalabels])

        # print(batch_sentlabels)
        dataentitys = getentity(batch_datalabels, data_out, self.device)
        wordentitys = getentity(batch_sentlabels, rnn_out, self.device)
        # print('wordentity:', wordentitys.size())
        scores = []
        for i in range(wordentitys.size()[1]):
            scores.append(self.attention(dataentitys, wordentitys[:,i,:]))
        scores = torch.stack(scores)
        scores = scores.view(scores.size()[:3])
        scores = torch.stack([scores[:,i,:] for i in range(scores.size()[1])])
        # print(scores.size())

        return scores

    def test(self, tensorized_sents, tensorized_datas, batch_sentlabels, batch_datalabels, lengths, lengthsD, _):
        """第三个参数不会用到，加它是为了与BiLSTM_CRF保持同样的接口"""
        logits = self.forward(tensorized_sents, tensorized_datas, batch_sentlabels, batch_datalabels, lengths, lengthsD)  # [B, L, out_size]
        _, batch_tagids = torch.max(logits, dim=2)

        return batch_tagids
