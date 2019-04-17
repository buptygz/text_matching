import re
import jieba
import random
from tensorflow.contrib import learn


class Data_Prepare(object):

    def readfile(self, filename):
        texta = []
        textb = []
        tag = []
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                texta.append(self.pre_processing(line[0]))  # 所有的query
                textb.append(self.pre_processing(line[1]))  # 所有的doc
                tag.append(line[2])  # 所有的tag
        # shuffle
        index = [x for x in range(len(texta))]
        random.shuffle(index)  # 索引进行打乱
        texta_new = [texta[x] for x in index]
        textb_new = [textb[x] for x in index]
        tag_new = [tag[x] for x in index]

        type = list(set(tag_new))  # tag种类有几种
        dicts = {}  # 统计每个tag取值的个数
        tags_vec = []  # 将tag进行one-hot编码
        for x in tag_new:
            if x not in dicts.keys():
                dicts[x] = 1
            else:
                dicts[x] += 1
            temp = [0] * len(type)
            temp[int(x)] = 1
            tags_vec.append(temp)
        print(dicts)
        return texta_new[:5000], textb_new[:5000], tags_vec[:5000]

    def pre_processing(self, text):
        # 删除（）里的内容
        text = re.sub('（[^（.]*）', '', text)
        # 只保留中文部分
        text = ''.join([x for x in text if '\u4e00' <= x <= '\u9fa5'])
        # 利用jieba进行分词
        words = ' '.join(jieba.cut(text)).split(" ")
        # 不分词
        words = [x for x in ''.join(words)]
        return ' '.join(words)

    # 建立一个大的词典，将sentence都进行embeding,保存的是相应词在词典里的索引
    def build_vocab(self, sentences, path):
        lens = [len(sentence.split(" ")) for sentence in sentences]
        max_length = max(lens)
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length)
        vocab_processor.fit(sentences)
        vocab_processor.save(path)


if __name__ == '__main__':
    data_pre = Data_Prepare()
    data_pre.readfile('data/train.txt')
