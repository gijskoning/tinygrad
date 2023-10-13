# Small neural nets with PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from extra.optimization.helpers import load_worlds


class State:
    '''Board implementation of Tic-Tac-Toe'''
    X, Y = 'ABC',  '123'
    C = {0: '_', BLACK: 'O', WHITE: 'X'}

    def __init__(self):
        self.board = np.zeros((3, 3)) # (x, y)
        self.color = 1
        self.win_color = 0
        self.record = []

    def action2str(self, a):
        return self.X[a // 3] + self.Y[a % 3]

    def str2action(self, s):
        return self.X.find(s[0]) * 3 + self.Y.find(s[1])

    def record_string(self):
        return ' '.join([self.action2str(a) for a in self.record])

    def __str__(self):
        # output board.
        s = '   ' + ' '.join(self.Y) + '\n'
        for i in range(3):
            s += self.X[i] + ' ' + ' '.join([self.C[self.board[i, j]] for j in range(3)]) + '\n'
        s += 'record = ' + self.record_string()
        return s

    def play(self, action):
        # state transition function
        # action is position inerger (0~8) or string representation of action sequence
        if isinstance(action, str):
            for astr in action.split():
                self.play(self.str2action(astr))
            return self

        x, y = action // 3, action % 3
        self.board[x, y] = self.color

        # check whether 3 stones are on the line
        if self.board[x, :].sum() == 3 * self.color \
          or self.board[:, y].sum() == 3 * self.color \
          or (x == y and np.diag(self.board, k=0).sum() == 3 * self.color) \
          or (x == 2 - y and np.diag(self.board[::-1,:], k=0).sum() == 3 * self.color):
            self.win_color = self.color

        self.color = -self.color
        self.record.append(action)
        return self

    def terminal(self):
        # terminal state check
        return self.win_color != 0 or len(self.record) == 3 * 3

    def terminal_reward(self):
        # terminal reward
        return self.win_color

    def action_length(self):
        return 3 * 3

    def legal_actions(self):
        # list of legal actions on each state
        return [a for a in range(3 * 3) if self.board[a // 3, a % 3] == 0]

    def feature(self):
        # input tensor for neural net (state)
        return np.stack([self.board == self.color, self.board == -self.color]).astype(np.float32)

    def action_feature(self, action):
        # input tensor for neural net (action)
        a = np.zeros((1, 3, 3), dtype=np.float32)
        a[0, action // 3, action % 3] = 1
        return a

class Conv(nn.Module):
    def __init__(self, filters0, filters1, kernel_size, bn=False):
        super().__init__()
        self.conv = nn.Conv2d(filters0, filters1, kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(filters1)

    def forward(self, x):
        h = self.conv(x)
        if self.bn is not None:
            h = self.bn(h)
        return h

class ResidualBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.conv = Conv(filters, filters, 3, True)

    def forward(self, x):
        return F.relu(x + (self.conv(x)))


num_filters = 16
num_blocks = 4


class Representation(nn.Module):
  ''' Conversion from observation to inner abstract state '''

  def __init__(self, input_shape):
    super().__init__()
    self.input_shape = input_shape
    self.board_size = self.input_shape[1] * self.input_shape[2]

    self.layer0 = Conv(self.input_shape[0], num_filters, 3, bn=True)
    self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

  def forward(self, x):
    h = F.relu(self.layer0(x))
    for block in self.blocks:
      h = block(h)
    return h

  def inference(self, x):
    self.eval()
    with torch.no_grad():
      rp = self(torch.from_numpy(x).unsqueeze(0))
    return rp.cpu().numpy()[0]


class Prediction(nn.Module):
  ''' Policy and value prediction from inner abstract state '''

  def __init__(self, action_shape):
    super().__init__()
    self.board_size = np.prod(action_shape[1:])
    self.action_size = action_shape[0] * self.board_size

    self.conv_p1 = Conv(num_filters, 4, 1, bn=True)
    self.conv_p2 = Conv(4, 1, 1)

    self.conv_v = Conv(num_filters, 4, 1, bn=True)
    self.fc_v = nn.Linear(self.board_size * 4, 1, bias=False)

  def forward(self, rp):
    h_p = F.relu(self.conv_p1(rp))
    h_p = self.conv_p2(h_p).view(-1, self.action_size)

    h_v = F.relu(self.conv_v(rp))
    h_v = self.fc_v(h_v.view(-1, self.board_size * 4))

    # range of value is -1 ~ 1
    return F.softmax(h_p, dim=-1), torch.tanh(h_v)

  def inference(self, rp):
    self.eval()
    with torch.no_grad():
      p, v = self(torch.from_numpy(rp).unsqueeze(0))
    return p.cpu().numpy()[0], v.cpu().numpy()[0][0]


class Dynamics(nn.Module):
  '''Abstract state transition'''

  def __init__(self, rp_shape, act_shape):
    super().__init__()
    self.rp_shape = rp_shape
    self.layer0 = Conv(rp_shape[0] + act_shape[0], num_filters, 3, bn=True)
    self.blocks = nn.ModuleList([ResidualBlock(num_filters) for _ in range(num_blocks)])

  def forward(self, rp, a):
    h = torch.cat([rp, a], dim=1)
    h = self.layer0(h)
    for block in self.blocks:
      h = block(h)
    return h

  def inference(self, rp, a):
    self.eval()
    with torch.no_grad():
      rp = self(torch.from_numpy(rp).unsqueeze(0), torch.from_numpy(a).unsqueeze(0))
    return rp.cpu().numpy()[0]


class Net(nn.Module):
  '''Whole net'''

  def __init__(self):
    super().__init__()
    state = State()
    input_shape = state.feature().shape
    action_shape = state.action_feature(0).shape
    rp_shape = (num_filters, *input_shape[1:])

    self.representation = Representation(input_shape)
    self.prediction = Prediction(action_shape)
    self.dynamics = Dynamics(rp_shape, action_shape)

  def predict(self, state0, path):
    '''Predict p and v from original state and path'''
    outputs = []
    x = state0.feature()
    rp = self.representation.inference(x)
    outputs.append(self.prediction.inference(rp))
    for action in path:
      a = state0.action_feature(action)
      rp = self.dynamics.inference(rp, a)
      outputs.append(self.prediction.inference(rp))
    return outputs


# Training of neural net

import torch.optim as optim

batch_size = 32
num_steps = 100
K = 1


def gen_target(state, ep):
  '''Generate inputs and targets for training'''
  # path, reward, observation, action, policy
  ep_length = len(ep['feature'])
  turn_idx = np.random.randint(ep_length)

  x = ep['feature'][turn_idx]
  ps, rs, acts, axs = [], [], [], []
  sas, seas, szs = [], [], []
  for t in range(turn_idx, turn_idx + K + 1):
    if t < ep_length:
      p = ep['policy'][t]
      a = ep['action'][t]
      ax = ep['action_feature'][t]
      sa = ep['sampled_info'][t]['a']
      sea = ep['sampled_info'][t]['exadv']
      sz = ep['sampled_info'][t]['z']
    else:  # state after finishing game
      p = np.zeros_like(ep['policy'][-1])
      # random action selection
      a = np.random.randint(state.action_length())
      ax = state.action_feature(a)
      sa = np.random.randint(state.action_length(), size=len(sa))
      sea = np.ones_like(sea)
      sz = np.ones_like(sz)

    rs.append([ep['reward'] if t % 2 == 0 else -ep['reward']])
    acts.append([a])
    axs.append(ax)
    ps.append(p)
    sas.append(sa)
    seas.append(sea)
    szs.append(sz)

  return x, rs, acts, axs, ps, sas, seas, szs


def train(episodes, net, opt):
  '''Train neural net'''
  pg_loss_sum, cmpo_loss_sum, v_loss_sum = 0, 0, 0
  net.train()
  state = State()

  for _ in range(num_steps):
    targets = [gen_target(state, episodes[np.random.randint(len(episodes))]) for j in range(batch_size)]
    x, r, a, ax, p_prior, sa, sea, sz = zip(*targets)
    x = torch.from_numpy(np.array(x))
    r = torch.from_numpy(np.array(r))
    a = torch.from_numpy(np.array(a))
    ax = torch.from_numpy(np.array(ax))
    p_prior = torch.from_numpy(np.array(p_prior))
    sa = torch.from_numpy(np.array(sa))
    sea = torch.from_numpy(np.array(sea))
    sz = torch.from_numpy(np.array(sz))

    # Compute losses for k (+ current) steps
    ps, vs = [], []
    rp = net.representation(x)
    for t in range(K + 1):
      p, v = net.prediction(rp)
      ps.append(p)
      vs.append(v)
      rp = net.dynamics(rp, ax[:, t])

    cmpo_loss, v_loss = 0, 0
    for t in range(K, -1, -1):
      cmpo_loss += -torch.mean(sea[:, t] / sz[:, t] * torch.log(ps[t].gather(1, sa[:, t])), dim=1).sum()
      v_loss += torch.sum(((vs[t] - r[:, t]) ** 2) / 2)

    p_selected = ps[0].gather(1, a[:, 0])
    p_selected_prior = p_prior[:, 0].gather(1, a[:, 0])
    clipped_rho = torch.clamp(p_selected.detach() / p_selected_prior, 0, 1)
    pg_loss = torch.sum(-clipped_rho * torch.log(p_selected) * (r[:, 0] - vs[0]))

    pg_loss_sum += pg_loss.item()
    cmpo_loss_sum += cmpo_loss.item() / (K + 1)
    v_loss_sum += v_loss.item() / (K + 1)

    optimizer.zero_grad()
    (pg_loss + cmpo_loss + v_loss).backward()
    optimizer.step()

  data_count = num_steps * batch_size
  print('pg_loss %f cmpo_loss %f v_loss %f' % (pg_loss_sum / data_count, cmpo_loss_sum / data_count, v_loss_sum / data_count))
  return net


# Main algorithm of Muesli

num_games = 5000
num_games_one_epoch = 40
num_sampled_actions = 10
simulation_depth = 1

C = 1

net = Net()
optimizer = optim.SGD(net.parameters(), lr=3e-4, weight_decay=3e-5, momentum=0.8)

# Display battle results

episodes = []
result_distribution = {1: 0, 0: 0, -1: 0}

for g in range(num_games):
  # Generate one episode
  # state = State()

  features, policies, selected_actions, selected_action_features = [], [], [], []
  sampled_infos = []
  while not state.terminal():
    feature = state.feature()
    rp_root = net.representation.inference(feature)
    p_root, v_root = net.prediction.inference(rp_root)
    p_mask = np.zeros_like(p_root)
    p_mask[state.legal_actions()] = 1
    p_root *= p_mask
    p_root /= p_root.sum()

    features.append(feature)
    policies.append(p_root)

    actions, exadvs = [], []
    for i in range(num_sampled_actions):
      action = np.random.choice(np.arange(len(p_root)), p=p_root)
      actions.append(action)

      rp = rp_root
      qs = []
      for t in range(simulation_depth):
        action_feature = state.action_feature(action)
        rp = net.dynamics.inference(rp, action_feature)
        p, v = net.prediction.inference(rp)
        qs.append(-v if t % 2 == 0 else v)
        action = np.random.choice(np.arange(len(p)), p=p)

      q = np.mean(qs)
      exadvs.append(np.exp(np.clip(q - v_root, -C, C)))

    exadv_sum = np.sum(exadvs)
    zs = []
    for exadv in exadvs:
      z = (1 + exadv_sum - exadv) / num_sampled_actions
      zs.append(z)
    sampled_infos.append({'a': actions, 'q': qs, 'exadv': exadvs, 'z': zs})

    # Select action with generated distribution, and then make a transition by that action
    selected_action = np.random.choice(np.arange(len(p_root)), p=p_root)
    selected_actions.append(selected_action)
    selected_action_features.append(state.action_feature(selected_action))
    state.play(selected_action)

  # reward seen from the first turn player
  reward = state.terminal_reward()
  result_distribution[reward] += 1
  episodes.append({
    'feature': features, 'action': selected_actions,
    'action_feature': selected_action_features, 'policy': policies,
    'reward': reward,
    'sampled_info': sampled_infos})

  if g % num_games_one_epoch == 0:
    print('game ', end='')
  print(g, ' ', end='')

  # Training of neural net
  if (g + 1) % num_games_one_epoch == 0:
    # Show the result distributiuon of generated episodes
    print('generated = ', sorted(result_distribution.items()))
    net = train(episodes, net, optimizer)
    vs_random_once = vs_random(net)
    print('vs_random   win: %d  draw: %d  lose: %d' %
          (vs_random_once.get(1, 0), vs_random_once.get(0, 0), vs_random_once.get(-1, 0)))
    for r, n in vs_random_once.items():
      vs_random_sum[r] += n
    print('(total)           win: %d  draw: %d  lose: %d ' %
          (vs_random_sum.get(1, 0), vs_random_sum.get(0, 0), vs_random_sum.get(-1, 0)))
    # show_net(net, State())
    # show_net(net, State().play('A1 C1 A2 C2'))
    # show_net(net, State().play('A1 B2 C3 B3 C1'))
    # show_net(net, State().play('B2 A2 A3 C1 B3'))
    # show_net(net, State().play('B2 A2 A3 C1'))
print('finished')