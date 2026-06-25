import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoisyLinear(nn.Module):
    """학습되는 노이즈를 가중치에 더해 탐험을 만드는 Linear 층 (NoisyNet, Factorized Gaussian).

    epsilon-greedy 대신 네트워크가 탐험량을 스스로 학습한다.
    학습 모드에서는 노이즈를 섞고, 평가 모드에서는 평균 가중치(mu)만 사용한다.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # 학습 파라미터: 평균(mu)과 노이즈 크기(sigma)를 가중치/편향 각각에 둠
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # 매 스텝 새로 뽑는 노이즈 (학습 대상이 아니므로 buffer로 등록)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """논문 기준 초기화: mu는 균등분포, sigma는 차원으로 스케일한 상수."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        """Factorized 노이즈용 스케일 함수 f(x) = sign(x) * sqrt(|x|)."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """새 노이즈를 뽑아 epsilon 버퍼를 갱신 (외적으로 가중치 모양 노이즈 생성)."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            # 학습: 가중치/편향에 노이즈를 섞어 탐험
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # 평가: 노이즈를 빼고 평균 가중치만 사용
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class DQN_CNN(nn.Module):
    """Atari용 CNN Q-네트워크. dueling / noisy 스위치로 완전연결부 구조를 바꾼다."""
    def __init__(self, n_action: int, in_channels: int = 4, dueling: bool = False, noisy: bool = False):
        super(DQN_CNN, self).__init__()
        self.dueling = dueling   # Dueling 구조 on/off

        # --- 합성곱 특징 추출부 (입력: (N, 4, 84, 84)) ---
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=8, stride=4)  # -> (32, 20, 20)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)                         # -> (64, 9, 9)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)                         # -> (64, 7, 7)

        feature_dim = 64 * 7 * 7   # 펼친 특징 차원: 64채널 * 7 * 7 = 3136

        # --- 완전연결부 (noisy면 NoisyLinear, 아니면 nn.Linear) ---
        if dueling:
            if noisy:
                # Dueling + Noisy: 공유층 / 가치(V) / 이점(A) 모두 NoisyLinear
                self.fc1 = NoisyLinear(feature_dim, 512)
                self.value_fc = NoisyLinear(512, 1)
                self.adv_fc = NoisyLinear(512, n_action)
            else:
                # Dueling: 상태가치 V(s)와 행동이점 A(s,a)를 분리해 추정
                self.fc1 = nn.Linear(feature_dim, 512)
                self.value_fc = nn.Linear(512, 1)
                self.adv_fc = nn.Linear(512, n_action)
        else:
            if noisy:
                # Noisy: 기본 구조에 NoisyLinear만 적용
                self.fc1 = NoisyLinear(feature_dim, 512)
                self.fc2 = NoisyLinear(512, n_action)
            else:
                # Vanilla: 기본 Linear 두 층
                self.fc1 = nn.Linear(feature_dim, 512)
                self.fc2 = nn.Linear(512, n_action)

    def forward(self, x):
        x = x.float()/255.0   # 0~255 픽셀을 0~1로 정규화
        # 합성곱 + ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # (N, C, H, W) -> (N, C*H*W) 평탄화
        x = x.view(x.size(0), -1)

        if self.dueling:
            # Dueling 결합: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
            x=F.relu(self.fc1(x))
            tmp = self.adv_fc(x)
            x=self.value_fc(x)+(tmp-tmp.mean(dim=1, keepdim=True))
            return x
        else:
            # 일반: fc1 -> ReLU -> fc2
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def reset_noise(self):
        """noisy=True일 때 모든 NoisyLinear 층의 노이즈를 새로 샘플링한다."""
        if self.dueling:
            self.fc1.reset_noise()
            self.value_fc.reset_noise()
            self.adv_fc.reset_noise()
        else:
            self.fc1.reset_noise()
            self.fc2.reset_noise()
