from itertools import zip_longest
from copy import deepcopy
from math import ldexp

import torch
import torch.nn as nn
import torch.optim as optim

from .util import tensorized, sort_by_lengths, cal_loss, cal_lstm_crf_loss
from .config import TrainingConfig, LSTMConfig
from .bilstm import BiLSTM


class BILSTM_Model(object):
    def __init__(self, vocab_size, data_size, crf=True):
        """功能：对LSTM的模型进行训练与测试
           参数:
            vocab_size:词典大小
            out_size:标注种类
            crf选择是否添加CRF层"""
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型参数
        self.emb_size = LSTMConfig.emb_size
        self.hidden_size = LSTMConfig.hidden_size

        self.crf = crf
        # 根据是否添加crf初始化不同的模型 选择不一样的损失计算函数
        if not crf:
            self.model = BiLSTM(vocab_size, data_size, self.emb_size, self.hidden_size).to(self.device)
            self.cal_loss_func = cal_loss


        # 加载训练参数：
        self.epoches = TrainingConfig.epoches
        self.print_step = TrainingConfig.print_step
        self.lr = TrainingConfig.lr
        self.batch_size = TrainingConfig.batch_size

        # 初始化优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # 初始化其他指标
        self.step = 0
        self._best_val_loss = 1e18
        self.best_model = None
    def train(self, word_lists, data_lists, wordlabel_lists, datalabel_lists, dataptr_lists,
            dev_word_lists, dev_data_lists, dev_wordlabel_lists, dev_datalabel_lists, dev_dataptr_lists,
            word2id, data2id):
        word_lists, data_lists, wordlabel_lists, datalabel_lists, dataptr_lists, _ = sort_by_lengths(word_lists, data_lists, wordlabel_lists, datalabel_lists, dataptr_lists)
        dev_word_lists, dev_data_lists, dev_wordlabel_lists, dev_datalabel_lists, dev_dataptr_lists, _ = sort_by_lengths(
            dev_word_lists, dev_data_lists, dev_wordlabel_lists, dev_datalabel_lists, dev_dataptr_lists)

        B = self.batch_size
        for e in range(1, self.epoches+1):
            self.step = 0
            losses = 0.
            for ind in range(0, len(word_lists), B):
                batch_sents = word_lists[ind:ind+B]
                batch_datas = data_lists[ind:ind+B]
                batch_sentlabels = wordlabel_lists[ind:ind+B]
                batch_datalabels = datalabel_lists[ind:ind+B]
                batch_dataptrs = dataptr_lists[ind:ind+B]
                # print(B)
                # print([len(x) for x in batch_sents])
                # print([len(x) for x in batch_datas])
                # print([len(x) for x in batch_sentlabels])
                # print([len(x) for x in batch_datalabels])
                # print([len(x) for x in batch_dataptrs])


                losses += self.train_step(batch_sents, batch_datas,batch_sentlabels,batch_datalabels,
                                          batch_dataptrs, word2id, data2id)

                if self.step % TrainingConfig.print_step == 0:
                    total_step = (len(word_lists) // B + 1)
                    print("Epoch {}, step/total_step: {}/{} {:.2f}% Loss:{:.4f}".format(
                        e, self.step, total_step,
                        100. * self.step / total_step,
                        losses / self.print_step
                    ))
                    losses = 0.

            # 每轮结束测试在验证集上的性能，保存最好的一个
            # print([len(x) for x in dev_tag_lists])
            val_loss = self.validate(
                dev_word_lists, dev_data_lists, dev_wordlabel_lists, dev_datalabel_lists, dev_dataptr_lists, word2id, data2id)
            print("Epoch {}, Val Loss:{:.4f}".format(e, val_loss))

    def train_step(self, batch_sents, batch_datas, batch_sentlabels, batch_datalabels, batch_dataptrs, word2id, data2id):
        self.model.train()
        self.step += 1
        # 准备数据
        tensorized_sents, lengths = tensorized(batch_sents, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        tensorized_datas, lengthsD = tensorized(batch_datas, data2id)
        tensorized_datas = tensorized_datas.to(self.device)


        # forward

        scores = self.model(tensorized_sents, tensorized_datas, batch_sentlabels, batch_datalabels, lengths, lengthsD)
        # print(batch_sentlabels)
        # print(batch_datalabels)
        # print(batch_dataptrs)
        # print(scores.size())
        # 计算损失 更新参数
        self.optimizer.zero_grad()

        loss = self.cal_loss_func(scores, batch_dataptrs).to(self.device)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, dev_word_lists, dev_data_lists, dev_wordlabel_lists, dev_datalabel_lists, dev_dataptr_lists, word2id, data2id):
        self.model.eval()
        with torch.no_grad():
            val_losses = 0.
            val_step = 0
            for ind in range(0, len(dev_word_lists), self.batch_size):
                val_step += 1
                # 准备batch数据
                batch_sents = dev_word_lists[ind:ind+self.batch_size]
                batch_datas = dev_data_lists[ind:ind+self.batch_size]
                batch_sentlabels = dev_wordlabel_lists[ind:ind+self.batch_size]
                batch_datalabels = dev_datalabel_lists[ind:ind+self.batch_size]
                batch_dataptrs = dev_dataptr_lists[ind:ind+self.batch_size]

                tensorized_sents, lengths = tensorized(
                    batch_sents, word2id)
                # print(lengths)
                tensorized_sents = tensorized_sents.to(self.device)

                tensorized_datas, lengthsD = tensorized(
                    batch_datas, data2id)
                tensorized_datas = tensorized_datas.to(self.device)

                # print(lengths)
                # forward
                scores = self.model(tensorized_sents, tensorized_datas, batch_sentlabels, batch_datalabels, lengths, lengthsD)

                # 计算损失
                loss = self.cal_loss_func(
                    scores, batch_dataptrs).to(self.device)
                val_losses += loss.item()
            val_loss = val_losses / val_step

            if val_loss < self._best_val_loss:
                print("保存模型...")
                self.best_model = deepcopy(self.model)
                self._best_val_loss = val_loss

            return val_loss

    def test(self, word_lists, data_lists, wordlabel_lists, datalabel_lists, dataptr_lists, word2id, data2id):
        """返回最佳模型在测试集上的预测结果"""
        # 准备数据
        word_lists, data_lists, wordlabel_lists, datalabel_lists, dataptr_lists, indices = sort_by_lengths(word_lists, data_lists, wordlabel_lists, datalabel_lists, dataptr_lists)
        tensorized_sents, lengths = tensorized(word_lists, word2id)
        tensorized_sents = tensorized_sents.to(self.device)
        tensorized_datas, lengthsD = tensorized(data_lists, data2id)
        tensorized_datas = tensorized_datas.to(self.device)

        self.best_model.eval()
        with torch.no_grad():
            batch_tagids = self.best_model.test(
                tensorized_sents, tensorized_datas, wordlabel_lists, datalabel_lists, lengths, lengthsD, 0)

        # 将id转化为标注
        pred_tag_lists = batch_tagids
        # indices存有根据长度排序后的索引映射的信息
        # 比如若indices = [1, 2, 0] 则说明原先索引为1的元素映射到的新的索引是0，
        # 索引为2的元素映射到新的索引是1...
        # 下面根据indices将pred_tag_lists和tag_lists转化为原来的顺序
        ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
        indices, _ = list(zip(*ind_maps))
        pred_tag_lists = [pred_tag_lists[i] for i in indices]
        tag_lists = [dataptr_lists[i] for i in indices]

        return pred_tag_lists, tag_lists