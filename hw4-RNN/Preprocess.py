import torch
from torch import nn
from gensim.models import Word2Vec


class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path):
        self.w2v_path = w2v_path
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
        self.sentences = sentences

    def get_w2v_model(self):
        # 加载你训练好的word to vec 模型
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        # 将word 加进embedding，并赋予一个随机生成的representtaion vector
        # word 只会是"<PAD>" or  "<UNK>"  unk表示未知的向量，pad表示填充0的字符
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)  # 0表示按行拼接

    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 获取训练好的 Word2Vec 词嵌入模型
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 製作一個 word2idx 的 dictionary
        # 製作一個 idx2word 的 list
        # 製作一個 word2vector 的 list
        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i + 1), end='\r')
            # e.g. self.word2index['he'] = 1
            # e.g. self.index2word[1] = 'he'
            # e.g. self.vectors[1] = 'he' vector
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)  # 把嵌入矩阵变成tensor
        # 將 "<PAD>" 跟 "<UNK>" 加進 embedding 裡面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        # 将每一个句子变成一样长度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        # 把句子里面的单词转换成index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i + 1), end='\r') # '\r'可以实现屏幕数值滚动
            sentence_idx = []
            for word in sen:
                if word in self.word2idx.keys():
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 将每个句子长度变成一样
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)
