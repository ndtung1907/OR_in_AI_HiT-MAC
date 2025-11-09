from __future__ import division
import torch
import numpy as np
import torch.nn as nn
from gym import spaces
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical

from utils import norm_col_init, weights_init
from perception import NoisyLinear, BiRNN, AttentionLayer


def build_model(obs_space, action_space, args, device):
    name = args.model

    if 'single' in name:
        model = A3C_Single(obs_space, action_space, args, device)
    elif 'multi' in name:
        model = A3C_Multi(obs_space, action_space, args, device)

    model.train()
    return model


def wrap_action(self, action):
    action = np.squeeze(action)
    out = action * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2.0
    return out


def sample_action(z, device, test=False):
    # Discrete actions (Categorical) actor trả logits z. Ta dùng softmax trên logits để lấy phân phối. sigma không có ý nghĩa ở đây
    logits = z
    prob = F.softmax(logits, dim=-1)
    log_prob = F.log_softmax(logits, dim=-1)
    entropy = -(log_prob * prob).sum(-1, keepdim=True)
    if test:
        action = prob.max(-1)[1].data
        action_env = action.cpu().numpy()  # np.squeeze(action.cpu().numpy(), axis=0)
    else:
        action = prob.multinomial(1).data
        log_prob = log_prob.gather(1, Variable(action))  # [num_agent, 1] # comment for sl slave
        action_env = action.squeeze(0)

    return action_env, entropy, log_prob


class ValueNet(nn.Module):
    def __init__(self, input_dim, head_name, num=1):
        super(ValueNet, self).__init__()
        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, num, sigma_init=0.017)
        else:
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)

    def forward(self, x):
        value = self.critic_linear(x)
        return value

    def sample_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            # self.critic_linear.sample_noise()
            self.critic_linear.remove_noise()


class AMCValueNet(nn.Module):
    def __init__(self, input_dim, head_name, num=1, device=torch.device('cpu')):
        super(AMCValueNet, self).__init__()
        self.head_name = head_name
        self.device = device

        if 'ns' in head_name:
            self.noise = True
            self.critic_linear = NoisyLinear(input_dim, num, sigma_init=0.017)
        if 'onlyJ' in head_name:
            self.noise = False
            self.critic_linear = nn.Linear(input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)
        else:
            self.noise = False
            self.critic_linear = nn.Linear(2 * input_dim, num)
            self.critic_linear.weight.data = norm_col_init(self.critic_linear.weight.data, 0.1)
            self.critic_linear.bias.data.fill_(0)
            self.attention = AttentionLayer(input_dim, input_dim, device)

        self.feature_dim = input_dim

    def forward(self, x, goal):
        _, feature_dim = x.shape
        value = []

        coalition = x.view(-1, feature_dim)
        n = coalition.shape[0]

        feature = torch.zeros([self.feature_dim]).to(self.device)
        value.append(self.critic_linear(torch.cat([feature, coalition[0]])))
        for j in range(1, n):
            _, feature = self.attention(coalition[:j].unsqueeze(0))
            value.append(self.critic_linear(torch.cat([feature.squeeze(), coalition[j]])))  # delta f = f[:j]-f[:j-1]

        # mean and sum
        value = torch.cat(value).sum()

        return value.unsqueeze(0)

    def sample_noise(self):
        if self.noise:
            self.critic_linear.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.critic_linear.remove_noise()
            

class PolicyNet(nn.Module):
    def __init__(self, input_dim, action_space, head_name, device):
        super(PolicyNet, self).__init__()
        self.head_name = head_name
        self.device = device
        num_outputs = action_space.n

        if 'ns' in head_name:
            self.noise = True
            self.actor_linear = NoisyLinear(input_dim, num_outputs, sigma_init=0.017)
        else:
            self.noise = False
            self.actor_linear = nn.Linear(input_dim, num_outputs)

            # init layers
            self.actor_linear.weight.data = norm_col_init(self.actor_linear.weight.data, 0.1)
            self.actor_linear.bias.data.fill_(0)

    def forward(self, x, test=False):
        # logits = F.relu(self.actor_linear(x)) # không nên để Relu ở logits, logits cần là real-valued
        logits = self.actor_linear(x)
        # sigma = torch.ones_like(logits)
        dist = Categorical(logits=logits)
        if test:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        # action, entropy, log_prob = sample_action(logits, sigma, self.device, test)
        log_prob = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        return action, entropy, log_prob

    def sample_noise(self):
        if self.noise:
            self.actor_linear.sample_noise()
            self.actor_linear2.sample_noise()

    def remove_noise(self):
        if self.noise:
            self.actor_linear.remove_noise()
            self.actor_linear2.remove_noise()


class EncodeBiRNN(torch.nn.Module):
    def __init__(self, dim_in, lstm_out=128, head_name='birnn_lstm', device=None):
        super(EncodeBiRNN, self).__init__()
        self.head_name = head_name

        self.encoder = BiRNN(dim_in, int(lstm_out / 2), 1, device, 'gru')

        self.feature_dim = self.encoder.feature_dim
        self.global_feature_dim = self.encoder.feature_dim
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        x = inputs
        hn, cn = self.encoder(x)

        feature = hn  # shape: [bs, num_camera, lstm_dim]

        global_feature = cn.permute(1, 0, 2).reshape(-1)

        return feature, global_feature


class EncodeLinear(torch.nn.Module):
    def __init__(self, dim_in, dim_out=32, head_name='lstm', device=None):
        super(EncodeLinear, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, dim_out),
            nn.ReLU(inplace=True)
        )

        self.head_name = head_name
        self.feature_dim = dim_out
        self.train()

    def forward(self, inputs):
        x = inputs
        feature = self.features(x)
        return feature


class A3C_Single(torch.nn.Module):  # single vision Tracking
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(A3C_Single, self).__init__()
        self.n = len(obs_space)
        obs_dim = obs_space[0].shape[1]

        lstm_out = args.lstm_out
        head_name = args.model

        self.head_name = head_name

        self.encoder = AttentionLayer(obs_dim, lstm_out, device)
        self.critic = ValueNet(lstm_out, head_name, 1)
        self.actor = PolicyNet(lstm_out, action_spaces[0], head_name, device)

        self.train()
        self.device = device

    def forward(self, inputs, test=False):
        data = inputs
        _, feature = self.encoder(data)

        actions, entropies, log_probs = self.actor(feature, test)
        values = self.critic(feature)

        return values, actions, entropies, log_probs

    def sample_noise(self):
        self.actor.sample_noise()

    def remove_noise(self):
        self.actor.remove_noise()


class A3C_Multi(torch.nn.Module):
    def __init__(self, obs_space, action_spaces, args, device=torch.device('cpu')):
        super(A3C_Multi, self).__init__()
        self.num_agents, self.num_targets, self.pose_dim = obs_space.shape

        lstm_out = args.lstm_out
        head_name = args.model
        self.head_name = head_name

        self.encoder = EncodeLinear(self.pose_dim, lstm_out, head_name, device)
        feature_dim = self.encoder.feature_dim

        self.attention = AttentionLayer(feature_dim, lstm_out, device)
        feature_dim = self.attention.feature_dim

        # create actor & critic
        self.actor = PolicyNet(feature_dim, spaces.Discrete(2), head_name, device)

        if 'shap' in head_name:
            self.ShapleyVcritic = AMCValueNet(feature_dim, head_name, 1, device)
        else:
            self.critic = ValueNet(feature_dim, head_name, 1)

        self.train()
        self.device = device

    def forward(self, inputs, test=False):
        pos_obs = inputs

        feature_target = Variable(pos_obs, requires_grad=True)
        feature_target = self.encoder(feature_target)  # num_agent, num_target, feature_dim

        feature_target = feature_target.reshape(-1, self.encoder.feature_dim).unsqueeze(0)  # [1, agent*target, feature_dim]
        feature, global_feature = self.attention(feature_target)  # num_agents, feature_dim
        feature = feature.squeeze()

        actions, entropies, log_probs = self.actor(feature, test)
        actions = actions.reshape(self.num_agents, self.num_targets, -1)

        if 'shap' not in self.head_name:
            values = self.critic(global_feature)
        else:
            values = self.ShapleyVcritic(feature, actions)  # shape [1,1]

        return values, actions, entropies, log_probs

    def sample_noise(self):
        self.actor.sample_noise()
        
    def remove_noise(self):
        self.actor.remove_noise()
