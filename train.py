import torch
import torch.nn.functional as F
import random
import numpy as np
import os
from collections import deque

def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    frac = min(step/decay_steps, 1)
    return (1 - frac) * eps_start + frac * eps_end

def select_action(obs, q_net, epsilon, n_actions, device):
    """epsilon-greedy 행동 선택.
    NoisyNet 사용 시 epsilon=0.0 으로 호출되며, 탐험은 네트워크의 stochastic weight가 담당한다.
    이 함수를 호출하기 전에 q_net.reset_noise()로 noise를 재샘플링해야 한다.
    """
    if random.random() < epsilon:
        return random.randrange(0, n_actions)
    else:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
            q_values = q_net(obs_t)
            return torch.argmax(q_values).item()

def compute_loss(q_net, target_net, obs, actions, rewards, next_obs, dones, gamma, use_double):
    """TD 타깃과의 차이로 요소별 Huber loss와 TD-error를 계산한다."""
    current_q = q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        if use_double:
            best_action = q_net(next_obs).argmax(dim=1)
            next_q = target_net(next_obs).gather(1, best_action.unsqueeze(1)).squeeze(1)
        else:
            next_q = target_net(next_obs).max(dim=1)[0]
        target = rewards + gamma * next_q * (1 - dones)

    td_errors = current_q - target
    elementwise_loss = F.smooth_l1_loss(current_q, target, reduction='none')
    return elementwise_loss, td_errors

def train_step(q_net, target_net, optimizer, buffer, batch_size, gamma, use_double, use_noisy, beta):
    """버퍼에서 한 배치를 뽑아 q_net을 한 번 업데이트한다."""
    obs, actions, rewards, next_obs, dones, weights, indices = buffer.sample(batch_size, beta)

    elementwise_loss, td_errors = compute_loss(
        q_net, target_net, obs, actions, rewards, next_obs, dones, gamma, use_double
    )
    loss = (elementwise_loss * weights).mean()

    optimizer.zero_grad()
    if use_noisy:
        q_net.reset_noise()
        target_net.reset_noise()

    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
    optimizer.step()

    buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
    return loss.item()

def train(env, q_net, target_net, optimizer, buffer, n_actions, device,
          total_steps, learning_starts, train_freq, target_update_freq,
          batch_size, gamma_n, beta_start, use_double, use_noisy,
          writer, run_name, save_freq=500_000):
    """환경과 상호작용하며 버퍼를 채우고 주기적으로 q_net을 학습시키는 메인 루프."""
    os.makedirs("checkpoints", exist_ok=True)
    recent_returns = deque(maxlen=100)
    last_loss = 0.0

    frame, _ = env.reset(seed=0)
    last_idx = buffer.start_episode(frame)

    for step in range(1, total_steps + 1):
        epsilon = 0.0 if use_noisy else linear_epsilon(step, 1.0, 0.1, 1_000_000)
        if use_noisy:
            q_net.reset_noise()
        obs = buffer.get_obs(last_idx)
        action = select_action(obs, q_net, epsilon, n_actions, device)

        next_frame, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        clipped_reward = float(np.sign(reward))

        next_idx = buffer.add_frame(next_frame)
        buffer.push(last_idx, action, clipped_reward, next_idx, done)
        last_idx = next_idx

        if done:
            frame, _ = env.reset()
            last_idx = buffer.start_episode(frame)
            if "episode" in info:
                ep_return = float(info["episode"]["r"])
                ep_length = int(info["episode"]["l"])
                recent_returns.append(ep_return)
                writer.add_scalar("charts/episode_return", ep_return, step)
                writer.add_scalar("charts/episode_length", ep_length, step)

        if step >= learning_starts and step % train_freq == 0:
            beta = min(1.0, beta_start + (1.0 - beta_start) * step / total_steps)
            last_loss = train_step(q_net, target_net, optimizer, buffer, batch_size, gamma_n, use_double, use_noisy, beta)

        if step % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        if step % 1000 == 0:
            writer.add_scalar("charts/epsilon", epsilon, step)
            writer.add_scalar("losses/td_loss", last_loss, step)

        if step % 10_000 == 0:
            avg = np.mean(recent_returns) if recent_returns else 0.0
            print(f"step={step:,}  eps={epsilon:.3f}  loss={last_loss:.4f}  avg_return(100ep)={avg:.1f}")

        if step % save_freq == 0:
            ckpt_path = f"checkpoints/{run_name}_{step}.pt"
            torch.save(q_net.state_dict(), ckpt_path)
            torch.save(q_net.state_dict(), f"checkpoints/{run_name}_latest.pt")
            print(f"  [checkpoint] {ckpt_path}")

    torch.save(q_net.state_dict(), f"checkpoints/{run_name}_latest.pt")
    writer.close()
    env.close()
    print("학습 완료.")
