# to correct text

主要过程是：输入data和错误text，输出正确text。分成两步：识别错误实体、改正错误实体。
识别错误实体：首先识别实体，然后判断是否需要修改，是的话输出前后单词坐标。
    识别实体：使用现有实体识别模型+数字——得到实体位置。
    判断修改：平均化实体单词向量，与data总向量做attention
修改错误实体：每个错误实体的平均向量对data的实体平均做attention来直接copy实体

## 具体步骤

    制造伪数据：利用分词来随机加，分词前后注意标记位置对，得到：src.data\src.text\tgt.text\tgt.textpos\tgt.datapos
    尝试实体识别模型
    写向量的平均等操作
    尝试pointernetwork，学习阅读源码
