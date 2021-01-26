from os.path import join
from codecs import open


def build_corpus(split, make_vocab=True, data_dir="../data/pseudo"):
    """读取数据"""
    assert split in ['train', 'dev', 'test']

    word_lists = []
    tag_lists = []
    data_lists = []
    with open(join(data_dir, split+".src.text"), 'r', encoding='utf-8') as f:
        flabel = open(join(data_dir, split+".tgt.textlabel"), 'r', encoding='utf-8')
        fdata = open(join(data_dir, split+".src.data"), 'r', encoding='utf-8')

        for text, label, data in zip(f, flabel, fdata):
            word_lists.append(text.strip().split(' '))
            tag_lists.append(label.strip().split(' '))
            data_lists.append(data.strip().split(' '))

    # 如果make_vocab为True，还需要返回word2id和tag2id
    if make_vocab:
        word2id = build_map(word_lists)
        tag2id = build_map(tag_lists)
        data2id = build_map(data_lists)
        return word_lists, tag_lists, data_lists, word2id, tag2id, data2id
    else:
        return word_lists, tag_lists, data_lists


def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
