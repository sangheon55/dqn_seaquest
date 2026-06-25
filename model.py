import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NoisyLinear(nn.Module):
    """평범한 Linear + 학습 가능한 크기의 랜덤 perturbation (NoisyNet, Factorized Gaussian).

    출력을 분해하면:
        y = (μ_W x + μ_b)            ← 그냥 평범한 linear layer
          + (σ_W ⊙ ε_W) x + (σ_b ⊙ ε_b)  ← NoisyNet의 본질

    뒤 항(σ ⊙ ε)의 의미:
        σ : 학습되는 파라미터. "이 weight를 얼마나 흔들어도 되는가"를 gradient descent로 직접 조절
        ε : 매 forward마다 새로 뽑는 random noise — 학습 안 됨, 그냥 랜덤 방향 소스
        σ⊙ε : "학습된 불확실성 크기 × 랜덤 방향" = 매번 다른 perturbation

    σ 가 0으로 수렴하면 perturbation 항이 사라져 평범한 linear로 퇴화한다.
    즉 탐험이 필요 없는 차원은 네트워크 스스로 노이즈를 꺼버린다.
    μ 는 본질이 아니라 "원래 있어야 했던 평범한 부분"이고,
    σ 가 이 레이어를 epsilon-greedy와 다르게 만드는 진짜 메커니즘이다.
    """
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # μ: 평범한 linear의 W, b와 동일한 역할
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        # σ: perturbation 크기를 학습 — 이것이 NoisyLinear의 정체성
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # ε: 매 forward마다 reset_noise()로 교체되는 랜덤 방향 (학습 대상 아님)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """μ는 균등분포, σ는 논문 기준 상수로 초기화."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        """Factorized noise의 스케일 함수 f(x) = sign(x)·√|x|.
        in/out 각각 1D noise를 뽑아 외적하면 행렬 크기 noise를 O(p+q)로 생성할 수 있다.
        """
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """ε 버퍼를 새로 샘플링 — 매 forward 전에 호출해 perturbation 방향을 교체한다."""
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        # 외적: (out,) ⊗ (in,) → (out, in) 크기의 weight noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            # y = (μ_W x + μ_b) + (σ_W ⊙ ε_W) x + (σ_b ⊙ ε_b)
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            # 평가 시: perturbation 항을 제거하고 μ만 사용 (deterministic)
            weight = self.weight_mu
            bias   = self.bias_mu

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
