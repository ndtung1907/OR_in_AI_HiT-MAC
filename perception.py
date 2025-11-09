import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

"""
Thay vì dùng deterministic nn.Linear, thêm gaussian noise parametrized (sigma) 
lên weight và bias để khuyến khích exploration.
"""
class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_init = sigma_init
        self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))
        self.sigma_bias = Parameter(torch.Tensor(out_features))
        self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
        self.register_buffer('epsilon_bias', torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'sigma_weight'):
            init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
            init.constant_(self.sigma_weight, self.sigma_init)
            init.constant_(self.sigma_bias, self.sigma_init)

    def forward(self, input):
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight,
                        self.bias + self.sigma_bias * self.epsilon_bias)

# self.epsilon_weight và self.epsilon_bias là buffer đăng ký trước. 
# Ở đây code gán thuộc tính mới torch.randn(...), torch.zeros(...) cho self.epsilon_weight và self.epsilon_bias, 
# tức thay thế buffer bằng tensor mới không phải buffer
# -> Tensor mới sẽ không bên trong state_dict() dưới tên 'epsilon_weight', 'epsilon_bias' (mất khi save/load).

    def sample_noise(self):
        # self.epsilon_weight = torch.randn(self.out_features, self.in_features)
        # self.epsilon_bias = torch.randn(self.out_features)
        self.epsilon_weight.normal_()
        self.epsilon_bias.normal_()

    def remove_noise(self):
        # self.epsilon_weight = torch.zeros(self.out_features, self.in_features)
        # self.epsilon_bias = torch.zeros(self.out_features)
        self.epsilon_weight.zero_()
        self.epsilon_bias.zero_()


class BiRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, head_name):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if 'lstm' in head_name:
            self.lstm = True
        else:
            self.lstm = False
        if self.lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True).to(device)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True).to(device)
        self.feature_dim = hidden_size * 2
        self.device = device

    def forward(self, x, state=None):
        # Set initial states

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        # LSTM self.rnn(x, (h0, c0)) trả về (out, (h_n, c_n)).
        if self.lstm:
            # out, (_, hn) = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
            out, (hn, cn) = self.rnn(x, ((h0, c0)))
        else:
            out, hn = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # return out, hn
        return out, cn

class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device, head_name):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if 'lstm' in head_name:
            self.lstm = True
        else:
            self.lstm = False
        if self.lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True).to(device)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True).to(device)
        self.feature_dim = hidden_size
        self.device = device

    def forward(self, x, state=None):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        # Forward propagate LSTM
        if self.lstm:
            # out, (_, hn) = self.rnn(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
            out, (hn, cn) = self.rnn(x, ((h0, c0)))
        else:
            out, hn = self.rnn(x, h0)  # out: tensor of shape (batch_size, seq_length, hidden_size)

        return out, hn

'''
Mục đích: khởi tạo trọng số sao cho phương sai của đầu vào/đầu ra của mỗi layer cân bằng 
—> giúp gradient lan truyền ổn định trong các mạng sâu.
'''
def xavier_init(layer):
    torch.nn.init.xavier_uniform_(layer.weight)
    # torch.nn.init.constant_(layer.bias, 0)
    # Kiểm tra layer có bias trước khi khởi tạo
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, 0)
    return layer


class AttentionLayer(torch.nn.Module):
    def __init__(self, feature_dim, weight_dim, device):
        super(AttentionLayer, self).__init__()
        self.in_dim = feature_dim
        self.device = device

        self.Q = xavier_init(nn.Linear(self.in_dim, weight_dim))
        self.K = xavier_init(nn.Linear(self.in_dim, weight_dim))
        self.V = xavier_init(nn.Linear(self.in_dim, weight_dim))

        self.feature_dim = weight_dim

    def forward(self, x):
        '''
        inference
        :param x: [num_agent, num_target, feature_dim]
        :return z: [num_agent, num_target, weight_dim]
        '''
        # z = softmax(Q,K)*V
        q = torch.tanh(self.Q(x))  # [batch_size, sequence_len, weight_dim]
        k = torch.tanh(self.K(x))  # [batch_size, sequence_len, weight_dim]
        v = torch.tanh(self.V(x))  # [batch_size, sequence_len, weight_dim]

        # z = torch.bmm(F.softmax(torch.bmm(q, k.permute(0, 2, 1)), dim=2), v)  # [batch_size, sequence_len, weight_dim]
        # Thiếu scaling: không chia score cho sqrt(d_k)
        # torch.bmm: batch matrix multiplication (nhân ma trận theo từng batch)
        z = torch.bmm(F.softmax((torch.bmm(q, k.permute(0, 2, 1))/math.sqrt(self.feature_dim)), dim=-1), v)
        global_feature = z.sum(dim=1)
        return z, global_feature