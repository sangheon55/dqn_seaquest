from model import DQN_CNN
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from env_utils import make_env
from train import train
import torch
import time
from torch.utils.tensorboard import SummaryWriter

# ===== 실험 순서 정의 =====
CONFIGS = [
    dict(USE_DOUBLE=False, USE_DUELING=False, USE_NSTEP=False, N_STEP=3, USE_PER=False, USE_NOISY=False),  # 0. Vanilla
    dict(USE_DOUBLE=True,  USE_DUELING=False, USE_NSTEP=False, N_STEP=3, USE_PER=False, USE_NOISY=False),  # 1. +Double
    dict(USE_DOUBLE=True,  USE_DUELING=True,  USE_NSTEP=False, N_STEP=3, USE_PER=False, USE_NOISY=False),  # 2. +Dueling
    dict(USE_DOUBLE=True,  USE_DUELING=True,  USE_NSTEP=True,  N_STEP=3, USE_PER=False, USE_NOISY=False),  # 3. +N-step
    dict(USE_DOUBLE=True,  USE_DUELING=True,  USE_NSTEP=True,  N_STEP=3, USE_PER=True,  USE_NOISY=False),  # 4. +PER
    dict(USE_DOUBLE=True,  USE_DUELING=True,  USE_NSTEP=True,  N_STEP=3, USE_PER=True,  USE_NOISY=True),   # 5. +Noisy (Rainbow)
]
# ==========================

def run_experiment(cfg, exp_idx, total_exps):
    USE_DOUBLE  = cfg["USE_DOUBLE"]
    USE_DUELING = cfg["USE_DUELING"]
    USE_NSTEP   = cfg["USE_NSTEP"]
    N_STEP      = cfg["N_STEP"]
    USE_PER     = cfg["USE_PER"]
    USE_NOISY   = cfg["USE_NOISY"]

    STACK_SIZE = 4

    env = make_env("ALE/Seaquest-v5", seed=0)
    n_actions = env.action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    flags = []
    if USE_DOUBLE:  flags.append("Double")
    if USE_DUELING: flags.append("Dueling")
    if USE_NSTEP:   flags.append(f"Nstep{N_STEP}")
    if USE_PER:     flags.append("PER")
    if USE_NOISY:   flags.append("Noisy")
    run_name = f"seaquest_{'_'.join(flags) if flags else 'vanilla'}_{int(time.time())}"

    print(f"\n{'='*60}")
    print(f"[{exp_idx}/{total_exps}] {run_name}")
    print(f"double={USE_DOUBLE} dueling={USE_DUELING} nstep={USE_NSTEP}(n={N_STEP}) per={USE_PER} noisy={USE_NOISY}")
    print(f"{'='*60}")

    q_net = DQN_CNN(n_action=n_actions, in_channels=STACK_SIZE, dueling=USE_DUELING, noisy=USE_NOISY).to(device)
    target_net = DQN_CNN(n_action=n_actions, in_channels=STACK_SIZE, dueling=USE_DUELING, noisy=USE_NOISY).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-4, eps=1.5e-4)

    # ===== 하이퍼파라미터 (A100 / 5M steps 기준) =====
    total_steps        = 5_000_000
    learning_starts    = 50_000
    train_freq         = 4
    target_update_freq = 10_000
    batch_size         = 256
    gamma              = 0.99
    beta_start         = 0.4
    # =================================================

    n_step = N_STEP if USE_NSTEP else 1
    gamma_n = gamma ** n_step
    buffer_kwargs = dict(capacity=500_000, frame_shape=env.observation_space.shape,
                         stack_size=STACK_SIZE, device=device, n_step=n_step, gamma=gamma)
    if USE_PER:
        buffer = PrioritizedReplayBuffer(**buffer_kwargs)
    else:
        buffer = ReplayBuffer(**buffer_kwargs)

    writer = SummaryWriter(f"runs/{run_name}")

    train(
        env, q_net, target_net, optimizer, buffer, n_actions, device,
        total_steps, learning_starts, train_freq, target_update_freq,
        batch_size, gamma_n, beta_start, USE_DOUBLE, USE_NOISY,
        writer, run_name,
    )
    # train() 안에서 env.close(), writer.close() 호출됨


def main():
    total = len(CONFIGS)
    for i, cfg in enumerate(CONFIGS):
        run_experiment(cfg, exp_idx=i, total_exps=total - 1)
    print("\n모든 실험 완료!")


if __name__ == "__main__":
    main()
