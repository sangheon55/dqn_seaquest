from model import DQN_CNN
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from env_utils import make_env
from train import train
import torch

def main():
    # ===== Rainbow 스위치 (켜고 끄기) =====
    USE_DOUBLE  = False   # Double DQN
    USE_DUELING = False   # Dueling network
    USE_NSTEP   = False   # n-step return
    N_STEP      = 3       # n-step에서 쓸 n (USE_NSTEP=True일 때만 의미)
    USE_PER     = False   # Prioritized Experience Replay
    USE_NOISY   = True    # Noisy Linear (켜면 epsilon-greedy 대신 노이즈로 탐험)
    # =====================================

    env = make_env("ALE/Seaquest-v5", seed=0)
    n_actions = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, n_actions={n_actions}")
    print(f"switches: double={USE_DOUBLE} dueling={USE_DUELING} nstep={USE_NSTEP}(n={N_STEP}) per={USE_PER} noisy={USE_NOISY}")

    # q_net은 학습용, target_net은 타깃 고정용 (주기적으로 q_net을 복사)
    q_net = DQN_CNN(n_action=n_actions, dueling=USE_DUELING, noisy=USE_NOISY).to(device)
    target_net = DQN_CNN(n_action=n_actions, dueling=USE_DUELING, noisy=USE_NOISY).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4,eps=1.5e-4)

    # ===== 하이퍼파라미터 =====
    total_steps = 1_000_000
    learning_starts = 10_000      # 버퍼에 이만큼 쌓이기 전까진 학습 안 함
    train_freq = 4                # 4 env step마다 1번씩만 학습
    target_update_freq = 10_000   # 이 주기마다 target_net 동기화
    batch_size = 32
    gamma = 0.99
    beta_start = 0.4              # PER 중요도 가중치 보정 시작값 (학습하며 1.0까지 anneal)

    # 버퍼 선택 (PER on/off) + n-step 길이 반영
    n_step = N_STEP if USE_NSTEP else 1
    gamma_n = gamma ** n_step     # n-step이면 target 할인율이 γ^n (1-step이면 그대로 γ)
    buffer_kwargs = dict(capacity=100_000, obs_shape=env.observation_space.shape,
                         device=device, n_step=n_step, gamma=gamma)
    if USE_PER:
        buffer = PrioritizedReplayBuffer(**buffer_kwargs)
    else:
        buffer = ReplayBuffer(**buffer_kwargs)

    # 설정/객체 준비 끝 -> 실제 학습 루프 실행
    train(
        env, q_net, target_net, optimizer, buffer, n_actions, device,
        total_steps, learning_starts, train_freq, target_update_freq,
        batch_size, gamma_n, beta_start, USE_DOUBLE, USE_NOISY,
    )


if __name__ == "__main__":
    main()
