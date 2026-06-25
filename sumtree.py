import numpy as np

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # sum tree 구조를 위한 배열
    
    def _set_priority(self, idx, priority):
        """리프 우선순위를 바꾸고 그 변화량을 루트까지 부모 합에 전파한다."""
        tree_idx = idx + self.capacity - 1  # 데이터 인덱스를 sum tree 인덱스로 변환
        diff = priority - self.tree[tree_idx]
        curr = tree_idx
        while True:
            self.tree[curr] += diff
            if curr == 0:
                break
            curr = (curr - 1) // 2   # 부모 노드로 거슬러 올라감
    
    def _get_sample_indices_weights(self, batch_size: int, beta: float, size: int):
        """우선순위 비례 샘플링 + 중요도 가중치(IS weight)를 함께 반환한다."""
        segment_size = self.tree[0] / batch_size
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
            weight = (size * prob) ** (-beta)
            weights.append(weight)

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()   # 최댓값으로 정규화 -> 0~1 범위
        data_indices = [ti - (self.capacity - 1) for ti in tree_indices]   # 리프 -> 데이터 인덱스

        return data_indices, weights