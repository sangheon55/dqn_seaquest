import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    """균등 샘플링 경험 리플레이 버퍼. n-step return을 지원한다."""
    def __init__(self, capacity: int, obs_shape, device, n_step: int = 1, gamma: float = 0.99):
        self.capacity = capacity
        self.device = device

        # transition을 항목별 numpy 배열로 보관 (obs는 메모리 절약 위해 uint8)
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

        self.idx = 0    # 다음에 덮어쓸 위치 (원형 버퍼)
        self.size = 0   # 현재 쌓인 transition 개수

        # --- n-step return용 ---
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)   # 최근 n개 transition 임시 보관
        # γ^0, γ^1, ..., γ^(n-1) 미리 계산 (push마다 gamma**i 재계산 방지)
        self.gamma_powers = [gamma ** i for i in range(n_step)]

    def push(self, obs, action, reward, next_obs, done):
        """transition 하나를 받아 저장한다 (n_step>1이면 n개 누적 후 저장)."""
        if self.n_step == 1:
            # 1-step: 받은 그대로 저장
            self._store_transition(obs, action, reward, next_obs, done)
            return

        self.n_step_buffer.append((obs, action, reward, next_obs, done))
        # 아직 n개가 안 모였고 에피소드도 안 끝났으면 저장 보류
        if len(self.n_step_buffer) < self.n_step and not done:
            return

        while self.n_step_buffer:
            n_step_reward=0
            # 누적 할인 보상 R = r0 + γ r1 + ... + γ^(n-1) r_{n-1}
            for i in range(len(self.n_step_buffer)):
                n_step_reward += self.gamma_powers[i] * self.n_step_buffer[i][2]
                # 윈도 안에서 done을 만나면 누적/부트스트랩을 거기서 중단
                if self.n_step_buffer[i][4]: # done이 True라면
                    break

            first_obs = self.n_step_buffer[0][0]
            first_action = self.n_step_buffer[0][1]
            last_next_obs = self.n_step_buffer[-1][3]
            is_done = self.n_step_buffer[-1][4]

            # (s_t, a_t, R, s_{t+n}, done) 형태로 저장
            self._store_transition(first_obs, first_action, n_step_reward, last_next_obs, is_done)

            self.n_step_buffer.popleft()

            # 진행 중이면 하나만 저장하고 종료, 에피소드 끝이면 남은 윈도를 모두 비움
            if not done:
                break


    def _store_transition(self, obs, action, reward, next_obs, done):
        """실제 배열 슬롯에 transition을 기록하고 인덱스를 전진시킨다."""
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity   # 가득 차면 앞에서부터 덮어씀
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        """무작위로 batch_size개를 뽑는다 (beta는 PER 호환용 인자, 여기선 미사용)."""
        indices = np.random.randint(0, self.size, size=batch_size)
        obs, actions, rewards, next_obs, dones = self._gather(indices)
        # 균등 샘플링은 중요도 보정이 필요 없음 -> weight = 1
        weights = torch.ones(batch_size, device=self.device)
        return obs, actions, rewards, next_obs, dones, weights, indices

    def update_priorities(self, indices, td_errors):
        """균등 버퍼는 우선순위 개념이 없어 no-op (PER에서 override)."""
        pass

    def _gather(self, indices):
        """주어진 인덱스의 transition을 tensor로 묶어 device로 올린다."""
        obs = torch.as_tensor(self.obs[indices], device=self.device)
        next_obs = torch.as_tensor(self.next_obs[indices], device=self.device)
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        dones = torch.as_tensor(self.dones[indices], device=self.device)
        return obs, actions, rewards, next_obs, dones

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """TD-error에 비례해 샘플링하는 PER 버퍼 (SumTree 기반)."""
    def __init__(self, capacity: int, obs_shape, device, n_step: int = 1, gamma: float = 0.99, alpha: float = 0.5, eps: float = 1e-6):
        super().__init__(capacity, obs_shape, device, n_step, gamma)
        self.alpha = alpha          # 우선순위 반영 강도 (0이면 균등)
        self.eps = eps              # 우선순위 0 방지용 작은 값
        self.max_priority = 1.0     # 새 transition에 줄 초기 우선순위
        self.tree = np.zeros(2 * capacity - 1)  # SumTree (내부 합 노드 + 리프)

    def _set_priority(self, tree_idx, priority):
        """리프 우선순위를 바꾸고 그 변화량을 루트까지 부모 합에 전파한다."""
        diff = priority - self.tree[tree_idx]
        curr = tree_idx
        while True:
            self.tree[curr] += diff
            if curr == 0:
                break
            curr = (curr - 1) // 2   # 부모 노드로 거슬러 올라감

    def _store_transition(self, obs, action, reward, next_obs, done):
        # 데이터 배열에 저장 (부모 클래스와 동일)
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_obs[self.idx] = next_obs
        self.dones[self.idx] = done

        # 새 transition은 최대 우선순위로 등록 (적어도 한 번은 뽑히도록)
        tree_idx = self.idx + self.capacity - 1   # 데이터 인덱스 -> SumTree 리프 인덱스
        self._set_priority(tree_idx, self.max_priority)

        # 인덱스/크기 갱신 (부모 클래스와 동일)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        """우선순위 비례 샘플링 + 중요도 가중치(IS weight)를 함께 반환한다."""
        # 전체 합을 batch_size개 구간으로 나눠 구간마다 하나씩 뽑음 (stratified)
        segment_size = self.tree[0]/batch_size
        tree_indices = []
        weights = []

        for i in range(batch_size):
            v = np.random.uniform(i*segment_size, (i+1)*segment_size)

            # 루트에서 시작해 누적합을 따라 리프까지 내려감
            curr=0
            while 1:
                left = 2 * curr + 1     # 왼쪽 자식
                right = 2 * curr + 2    # 오른쪽 자식

                if left >= len(self.tree):  # 리프 도달
                    break

                left_val = self.tree[left]
                if v <= left_val:
                    curr = left
                else:
                    v -= left_val
                    curr = right

            tree_indices.append(curr)
            prob = self.tree[curr] / self.tree[0]   # 이 샘플이 뽑힐 확률

            # IS weight = (N * P(i))^(-beta), 우선순위 샘플링이 만든 편향 보정
            weight = (self.size * prob) ** (-beta)
            weights.append(weight)

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()   # 최댓값으로 정규화 -> 0~1 범위
        weights = torch.as_tensor(weights, device=self.device)
        data_indices = [ti - (self.capacity - 1) for ti in tree_indices]   # 리프 -> 데이터 인덱스

        obs, actions, rewards, next_obs, dones = self._gather(data_indices)
        return obs, actions, rewards, next_obs, dones, weights, tree_indices

    def update_priorities(self, tree_indices, td_errors):
        """p = (|td_error| + eps)^alpha 로 우선순위를 갱신하고 max_priority도 추적한다."""
        for tree_idx, td_error in zip(tree_indices,td_errors):
            p = (abs(float(td_error)) + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, p)
            self._set_priority(tree_idx, p)
