# 语料库处理

## 语料库
国家语委现代汉语语料库，大概五十万条句子。

## 处理过程

1. 按照语料库中字出现频率编码，编码从0开始。用得到的字编码编码原始语料库。
   在处理过程中会对原始语料库进行清洗，删除空行以及无中文行，替换原始语料库。
   `python data_utils.py <corpus file path> <encode corpus path> <vocab list file path> <vacab pkl path>`

2. 在原始语料库中加入三种错误：漏字(0)，多字(1)，错字(2)。
   `python mixerror.py [corpus file path] [vocab pkl path]`

