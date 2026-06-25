import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

def make_env(env_id: str, seed: int, render_mode=None):
    """Atari 환경을 만들고 DQN 표준 전처리 래퍼를 순서대로 씌워 반환한다."""
    # 원본 환경: frameskip=1로 둬야 AtariPreprocessing의 frame_skip과 겹치지 않음
    # render_mode="human"이면 게임 창이 뜸 (play.py 용, 학습 때는 None)
    env = gym.make(env_id, frameskip=1, render_mode=render_mode)

    # 에피소드 점수/길이를 info['episode']에 기록 (로깅용)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # Atari 표준 전처리: noop reset + frame skip & maxpool + 흑백 + 리사이즈
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,                  # reset 직후 0~30프레임 무작위 NOOP -> 매번 다른 시작 상태
        frame_skip=4,                 # 4프레임마다 한 번만 행동 결정 (나머지는 같은 행동 반복)
        screen_size=84,               # 84x84로 리사이즈
        terminal_on_life_loss=False,  # 목숨 하나 잃어도 종료 안 함, 진짜 게임오버에만 에피소드 종료
        grayscale_obs=True,           # 흑백(채널 1개), 색 정보는 거의 불필요
    )

    # 최근 4프레임을 쌓아 (4, 84, 84) 관측 생성 (움직임/방향 정보 확보)
    #env = gym.wrappers.FrameStackObservation(env, stack_size=1) 
    # => frame_stack는 버퍼에서 직접 구현하므로 여기선 쓰지 않음
    
    env.action_space.seed(seed)   # 행동 샘플링 재현성 확보
    return env
