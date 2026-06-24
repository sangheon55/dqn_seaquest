import torch
import torch.nn.functional as F
import random
import numpy as np

def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    """epsilon을 eps_start에서 eps_end까지 decay_steps에 걸쳐 선형으로 감소시킨다."""
    # step/decay_steps를 1로 캡 -> decay_steps 이후로는 eps_end에 고정
    frac = min(step/decay_steps,1)
    # frac=0이면 eps_start, frac=1이면 eps_end가 되도록 선형보간
    return (1 - frac) * eps_start + frac * eps_end

def select_action(obs, q_net, epsilon, n_actions, device):
    """epsilon 확률로 무작위 행동, 아니면 Q값이 가장 큰 행동을 고른다 (epsilon-greedy)."""
    if random.random() < epsilon:
        return random.randrange(0,n_actions)   # 0 ~ n_actions-1 중 무작위 정수
    else:
        with torch.no_grad():   # 행동 선택엔 backward가 없으므로 그래디언트 계산 끔 (메모리/속도 절약)
            # obs(numpy, (4,84,84))를 tensor로 바꾸고 배치 차원 추가 -> (1,4,84,84)
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
            q_values = q_net(obs_t)   # (1, n_actions)
            return torch.argmax(q_values).item()   # 최댓값 인덱스를 python int로 변환

def compute_loss(q_net, target_net, obs, actions, rewards, next_obs, dones, gamma, use_double):
    """TD 타깃과의 차이로 요소별 Huber loss와 TD-error를 계산한다."""
    # 현재 상태에서 실제로 취한 행동의 Q값만 뽑기: (batch, n_actions) -> (batch,)
    current_q = q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

    # TD 타깃 계산 (학습 대상이 아니므로 no_grad)
    with torch.no_grad():
        if use_double:
            # Double DQN: 행동 선택은 q_net, 그 행동의 평가는 target_net
            best_action = q_net(next_obs).argmax(dim=1)
            next_q = target_net(next_obs).gather(1,best_action.unsqueeze(1)).squeeze(1)
        else:
            # 기본 DQN: target_net의 다음 상태 Q값 중 최댓값
            next_q = target_net(next_obs).max(dim=1)[0]
        # done이면 부트스트랩 항을 0으로 (종료 이후 보상 없음)
        target = rewards + gamma * next_q * (1 - dones)

    # TD-error: PER 우선순위 갱신에 사용
    td_errors = current_q - target
    # Huber(smooth L1): 이상치에 덜 민감. PER 가중치를 곱하려고 reduction='none'
    elementwise_loss = F.smooth_l1_loss(current_q, target, reduction='none')
    return elementwise_loss, td_errors

def train_step(q_net, target_net, optimizer, buffer, batch_size, gamma, use_double, use_noisy ,beta):
    """버퍼에서 한 배치를 뽑아 q_net을 한 번 업데이트한다."""
    obs, actions, rewards, next_obs, dones, weights, indices = buffer.sample(batch_size, beta)

    elementwise_loss, td_errors = compute_loss(
        q_net, target_net, obs, actions, rewards, next_obs, dones, gamma, use_double
    )
    # PER 중요도 가중치 적용 (균등 버퍼는 weights=1이라 단순 평균과 동일)
    loss = (elementwise_loss * weights).mean()

    optimizer.zero_grad()   # 이전 그래디언트 초기화
    # NoisyNet: 매 업데이트마다 두 네트워크의 노이즈를 새로 샘플링
    if use_noisy:
        q_net.reset_noise()
        target_net.reset_noise()
    loss.backward()          # 그래디언트 계산 (q_net에만 생김)

    # DQN 계열 표준: grad norm 10으로 클리핑 (PER에서 큰 TD-error가 들어와도 안정)
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()         # q_net 가중치 업데이트

    # PER: TD-error로 우선순위 갱신 (균등 버퍼는 no-op)
    buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
    return loss.item()

def train(env, q_net, target_net, optimizer, buffer, n_actions, device,
          total_steps, learning_starts, train_freq, target_update_freq,
          batch_size, gamma_n, beta_start, use_double, use_noisy):
    """환경과 상호작용하며 버퍼를 채우고 주기적으로 q_net을 학습시키는 메인 루프."""
    obs, _ = env.reset(seed=0)

    for step in range(1, total_steps + 1):
        # NoisyNet을 쓰면 노이즈가 탐험을 담당하므로 epsilon=0
        epsilon = 0.0 if use_noisy else linear_epsilon(step, 1.0, 0.01, 250_000)
        action = select_action(obs, q_net, epsilon, n_actions, device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated       # 두 종료 신호를 하나로 합침

        # 보상 클리핑: {-1, 0, +1}로 부호만 남김 (Atari DQN 표준, 학습 안정화)
        clipped_reward = float(np.sign(reward))

        buffer.push(obs, action, clipped_reward, next_obs, done)
        obs = next_obs

        if done:
            obs, _ = env.reset()   # 에피소드 끝났으면 환경 리셋

        # 버퍼가 learning_starts 이상 쌓였고 train_freq 배수일 때만 학습
        if step >= learning_starts and step % train_freq == 0:
            beta = min(1.0, beta_start + (1.0 - beta_start) * step / total_steps)   # PER beta anneal (0.4 -> 1.0)
            train_step(q_net, target_net, optimizer, buffer, batch_size, gamma_n, use_double, use_noisy, beta)

        if step % target_update_freq == 0:   # 주기마다 target_net을 q_net으로 동기화
            target_net.load_state_dict(q_net.state_dict())

    env.close()
