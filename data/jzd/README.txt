文件描述：
最原始文件是：webnlgxxx
从其中摘取data和text，得到 src.data, tgt.data(与前者没有区别，处理了一些空格), tgt.text
向tgt.text中根据data替换实体，得到src.text
记录这一过程：
    错text中的错位置：textpos 和 textlabel
    data中的实体标记：datalabel
    text中应该修改为data中的位置：datapos，按照实体来数：dataptr


最终的目的文件：
src.data：原data
src.text: 错text
tgt.text：对text
tgt.textlabel
tgt.datalabel:注意两个label不一样，一个是标记错误实体，另一个是标记实体
tgt.dataptr：实体计数的位置