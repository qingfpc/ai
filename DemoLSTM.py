from torch import nn
#LSTM
class BiRNN(nn.Module):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers):
        '''
        @params:
            vocab: 在数据集上创建的词典，用于获取词典大小
            embed_size: 嵌入维度大小
            num_hiddens: 隐藏状态维度大小
            num_layers: 隐藏层个数
        '''
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embed_size)  # 映射长度,这里是降维度的作用

        # encoder-decoder framework
        # bidirectional设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=embed_size,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               bidirectional=True)  # 双向循环网络
        self.decoder = nn.Linear(4 * num_hiddens, 2)  # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        # 循环神经网络最后的隐藏状态可以用来表示一句话