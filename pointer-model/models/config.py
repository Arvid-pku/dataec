# 设置lstm训练参数
class TrainingConfig(object):
    batch_size = 2
    # 学习速率
    lr = 0.001
    epoches = 2
    print_step = 25


class LSTMConfig(object):
    emb_size = 128  # 词向量的维数
    hidden_size = 128  # lstm隐向量的维数
