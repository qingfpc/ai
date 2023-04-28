import torch

import DemoLSTM

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

    def forward(self, inputs):
        '''
        @params:
            inputs: 词语下标序列，形状为 (batch_size, seq_len) 的整数张量
        @return:
            outs: 对文本情感的预测，形状为 (batch_size, 2) 的张量
        '''
        # 因为LSTM需要将序列长度(seq_len)作为第一维，所以需要将输入转置,注意这里转置了!!!!
        embeddings = self.embedding(inputs.permute(1, 0))  # (seq_len, batch_size, d)500*64*100
        # print(embeddings.shape)
        # rnn.LSTM 返回输出、隐藏状态和记忆单元，格式如 outputs, (h, c)
        outputs, _ = self.encoder(embeddings)  # (seq_len, batch_size, 2*h)每一个输出,然后将第一次输出和最后一次输出拼接
        # print(outputs.shape)# 如果是双向LSTM，每个time step的输出h = [h正向, h逆向] (同一个time step的正向和逆向的h连接起来)
        encoding = torch.cat((outputs[0], outputs[-1]), -1)  # (batch_size, 4*h)
        outs = self.decoder(encoding)  # (batch_size, 2)
        return outs

# device = torch.device("cuda")
LoadModel = torch.load('./model.pth')
# loadModel = model.load_state_dict(torch.load('./params.pth'))
vocab = torch.load('./vocab_obj.pth')

#评价模型
def predict_sentiment(net, vocab, sentence):
    '''
    @params：
        net: 训练好的模型
        vocab: 在该数据集上创建的词典，用于将给定的单词序转换为单词下标的序列，从而输入模型
        sentence: 需要分析情感的文本，以单词序列的形式给出
    @return: 预测的结果，positive 为正面情绪文本，negative 为负面情绪文本
    '''
    device = list(net.parameters())[0].device # 读取模型所在的环境
    sentence = torch.tensor([vocab.stoi[word] for word in sentence], device=device)
    label = torch.argmax(net(sentence.view((1, -1))), dim=1)# 这里输入之后,进入embedding,进入lstm,进入全连接层,输出结果
    return 'positive' if label.item() == 1 else 'negative'


sentence=input("请输入：")
tokenized = sentence.lower().split()
print(predict_sentiment(LoadModel, vocab, tokenized))