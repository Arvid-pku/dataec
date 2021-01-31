from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from data import build_corpus
from evaluating import Metrics
from evaluate import ensemble_evaluate
import numpy

BiLSTM_MODEL_PATH = './ckpts/bilstm.pkl'

REMOVE_O = False  # 在评估的时候是否去除O标记


def main():
    print("读取数据...")
    train_word_lists, train_data_lists, train_wordlabel_lists, train_datalabel_lists, train_dataptr_lists, word2id, data2id = build_corpus("train")
    dev_word_lists, dev_data_lists, dev_wordlabel_lists, dev_datalabel_lists, dev_dataptr_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_data_lists, test_wordlabel_lists, test_datalabel_lists, test_dataptr_lists = build_corpus("test", make_vocab=False)

    # bilstm模型
    print("加载并评估bilstm模型...")
    bilstm_word2id, bilstm_data2id = extend_maps(word2id, data2id, for_crf=False)
    bilstm_model = load_model(BiLSTM_MODEL_PATH)
    bilstm_model.model.bilstm.flatten_parameters()  # remove warning
    lstm_pred, target_tag_list = bilstm_model.test(test_word_lists, test_data_lists, test_wordlabel_lists, test_datalabel_lists, test_dataptr_lists, bilstm_word2id, bilstm_data2id)
    allnum = 0
    correct = 0
    f = open('test.tgt.dataptr1', 'w')
    for pred, gold in zip(lstm_pred, target_tag_list):
        pred = pred.cpu().numpy().tolist()[:len(gold)]
        f.write(' '.join([str(x) for x in pred])+'\n')
        for x, y in zip(pred, gold):
            if x == y:
                correct += 1
            allnum += 1
    f.close()
    # TODO
    print(correct/allnum)

if __name__ == "__main__":
    main()
