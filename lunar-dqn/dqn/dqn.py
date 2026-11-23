import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ENV_ID = "LunarLander-v3"
SEED = 0
TOTAL_STEPS = 200_000
GAMMA = 0.99
LR = 1e-3
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 50_000
PLOT_WINDOW = 20


def make_q_net(obs_dim: int, n_actions: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(obs_dim, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, n_actions),
    )


def epsilon(step: int) -> float:
    frac = min(step / EPS_DECAY_STEPS, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)


def greedy_action(q_net: nn.Module, obs: np.ndarray) -> int:
    with torch.no_grad():
        return int(q_net(torch.as_tensor(obs, dtype=torch.float32)).argmax())


def draw_returns(steps: list[int], returns: list[float]) -> None:
    plt.clf()
    plt.plot(steps, returns, alpha=0.3, label="episode return")
    if len(returns) >= PLOT_WINDOW:
        avg = np.convolve(returns, np.ones(PLOT_WINDOW) / PLOT_WINDOW, mode="valid")
        plt.plot(steps[PLOT_WINDOW - 1:], avg, label=f"mean of last {PLOT_WINDOW}")
    plt.axhline(200, linestyle="--", color="black", label="solved = 200")
    plt.xlabel("environment step")
    plt.ylabel("return")
    plt.title(f"{ENV_ID} | Mark I | seed {SEED}")
    plt.legend(loc="lower right")
    plt.pause(0.001)


def train() -> nn.Module:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    env = gym.make(ENV_ID)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    q_net = make_q_net(obs_dim, n_actions)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=LR)

    plt.ion()
    plt.figure(figsize=(9, 4.5))

    obs, _ = env.reset(seed=SEED)
    ep_return, returns, steps_at_end = 0.0, [], []

    for step in range(TOTAL_STEPS):
        if rng.random() < epsilon(step):
            action = int(rng.integers(n_actions))
        else:
            action = greedy_action(q_net, obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        ep_return += float(reward)

        obs_t = torch.as_tensor(obs, dtype=torch.float32)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32)
        with torch.no_grad():
            target = reward + GAMMA * (1.0 - float(terminated)) * q_net(next_obs_t).max()
        q_pred = q_net(obs_t)[action]
        loss = (q_pred - target) ** 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        obs = next_obs
        if terminated or truncated:
            returns.append(ep_return)
            steps_at_end.append(step)
            if len(returns) % PLOT_WINDOW == 0:
                print(f"step {step + 1:7d} | ep {len(returns):4d} | eps {epsilon(step):.2f} "
                      f"| mean return (last {PLOT_WINDOW}) {np.mean(returns[-PLOT_WINDOW:]):8.1f}")
                draw_returns(steps_at_end, returns)
            obs, _ = env.reset()
            ep_return = 0.0

    draw_returns(steps_at_end, returns)
    plt.savefig(f"returns_mark1_seed{SEED}.png", dpi=120)
    env.close()
    return q_net


def watch(q_net: nn.Module, n_episodes: int = 3) -> None:
    env = gym.make(ENV_ID, render_mode="human")
    for i in range(n_episodes):
        obs, _ = env.reset(seed=1000 + i)
        ep_return, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(greedy_action(q_net, obs))
            ep_return += float(reward)
            done = terminated or truncated
        print(f"watch episode {i}: return {ep_return:.1f}")
    env.close()


if __name__ == "__main__":
    q_net = train()
    watch(q_net)
    plt.ioff()
    plt.show()
