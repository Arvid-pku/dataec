from os.path import join
from codecs import open


def build_corpus(split, make_vocab=True, data_dir="../data/pseudo"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    data_lists = []
    wordlabel_lists = []
    datalabel_lists = []
    dataptr_lists = []
    with open(join(data_dir, split+".src.text"), 'r', encoding='utf-8') as ftext:
        fdata = open(join(data_dir, split+".src.data"), 'r', encoding='utf-8')
        ftextlabel = open(join(data_dir, split+".tgt.textlabel"), 'r', encoding='utf-8')
        fdatalabel = open(join(data_dir, split+".tgt.datalabel"), 'r', encoding='utf-8')
        fdataptr = open(join(data_dir, split+".tgt.dataptr"), 'r', encoding='utf-8')

        for text, data, textlabel, datalabel, dataptr in zip(ftext, fdata, ftextlabel, fdatalabel, fdataptr):
            if len([int(x) for x in dataptr.strip().split(' ') if x]) == 0:
                continue
            dataptr_lists.append([int(x) for x in dataptr.strip().split(' ') if x])
            word_lists.append(text.strip().split(' '))
            data_lists.append(data.strip().split(' '))
            wordlabel_lists.append([int(x) for x in textlabel.strip().split(' ') if x])
            datalabel_lists.append([int(x) for x in datalabel.strip().split(' ') if x])

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        data2id = build_map(data_lists)
        return word_lists, data_lists, wordlabel_lists, datalabel_lists, dataptr_lists, word2id, data2id
    else:
        return word_lists, data_lists, wordlabel_lists, datalabel_lists, dataptr_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
