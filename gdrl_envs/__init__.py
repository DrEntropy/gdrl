"""Environments used by the chapter 6 notebook.

SlipperyWalkSeven and Russell and Norvig's gridworld are not in Gymnasium.
FrozenLake is, so the notebook makes that one directly.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry

LEFT, DOWN, RIGHT, UP = range(4)
WEST, EAST = 0, 1


class WalkEnv(gym.Env):
    """A line of states with terminals at both ends. Right terminal pays +1."""

    metadata = {"render_modes": []}

    def __init__(self, n_states=7, p_stay=0.0, p_backward=0.5):
        super().__init__()
        self.shape = (1, n_states + 2)
        self.start_state_index = self.shape[1] // 2
        self.nS = nS = int(np.prod(self.shape))
        self.nA = nA = 2
        p_forward = 1.0 - p_stay - p_backward

        P = {}
        for s in range(nS):
            P[s] = {}
            for a in range(nA):
                if s in (0, nS - 1):
                    s_forward = s_backward = s
                else:
                    s_forward = int(np.clip(s - 1 if a == WEST else s + 1, 0, nS - 1))
                    s_backward = int(np.clip(s + 1 if a == WEST else s - 1, 0, nS - 1))

                r_forward = 1.0 if s == nS - 2 and s_forward == nS - 1 else 0.0
                r_backward = 1.0 if s == nS - 2 and s_backward == nS - 1 else 0.0
                d_forward = (s >= nS - 2 and s_forward == nS - 1) or (s <= 1 and s_forward == 0)
                d_backward = (s >= nS - 2 and s_backward == nS - 1) or (s <= 1 and s_backward == 0)
                terminal = s in (0, nS - 1)
                P[s][a] = [
                    (p_forward, s_forward, r_forward, bool(d_forward or terminal)),
                    (p_stay, s, 0.0, terminal),
                    (p_backward, s_backward, r_backward, bool(d_backward or terminal)),
                ]

        self.P = P
        self.isd = np.zeros(nS)
        self.isd[self.start_state_index] = 1.0
        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = int(self.np_random.choice(self.nS, p=self.isd))
        return self.s, {}

    def step(self, action):
        action = int(action)
        transitions = self.P[self.s][action]
        probs = [t[0] for t in transitions]
        i = int(self.np_random.choice(len(transitions), p=probs))
        _p, s, r, terminated = transitions[i]
        self.s = int(s)
        return self.s, float(r), bool(terminated), False, {}


class RussellNorvigGridworldEnv(gym.Env):
    """3x4 gridworld from AIMA. Goal is state 3. Living reward is -0.04."""

    metadata = {"render_modes": []}

    def __init__(self, noise=0.2, living_rew=-0.04):
        super().__init__()
        desc = np.asarray(["FFFG", "FWFH", "SFFF"], dtype="c")
        self.desc = desc
        nrow, ncol = desc.shape
        nS = nrow * ncol
        nA = 4

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return row, col

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(nA):
                    letter = desc[row, col]
                    if letter in b"GHW":
                        P[s][a] = [(1.0, s, 0.0, True)]
                        continue
                    for b in ((a - 1) % 4, a, (a + 1) % 4):
                        newrow, newcol = inc(row, col, b)
                        newstate = to_s(newrow, newcol)
                        newletter = desc[newrow, newcol]
                        if newletter in b"W":
                            newstate = s
                        done = bytes(newletter) in b"GH"
                        rew = living_rew
                        if newletter == b"G":
                            rew += 1.0
                        elif newletter == b"H":
                            rew -= 1.0
                        prob = round(1.0 - noise if a == b else noise / 2.0, 2)
                        P[s][a].append((prob, int(newstate), float(rew), bool(done)))

        self.P = P
        self.nS = nS
        self.nA = nA
        self.isd = (desc.ravel() == b"S").astype(np.float64)
        self.isd /= self.isd.sum()
        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.s = int(self.np_random.choice(self.nS, p=self.isd))
        return self.s, {}

    def step(self, action):
        action = int(action)
        transitions = self.P[self.s][action]
        probs = [t[0] for t in transitions]
        i = int(self.np_random.choice(len(transitions), p=probs))
        _p, s, r, terminated = transitions[i]
        self.s = int(s)
        return self.s, float(r), bool(terminated), False, {}


def _register(env_id, **kwargs):
    if env_id not in registry:
        register(id=env_id, **kwargs)


_register(
    "SlipperyWalkSeven-v0",
    entry_point="gdrl_envs:WalkEnv",
    kwargs={"n_states": 7, "p_stay": 0.5 * 2 / 3, "p_backward": 0.5 / 3},
    max_episode_steps=100,
)
_register(
    "RussellNorvigGridworld-v0",
    entry_point="gdrl_envs:RussellNorvigGridworldEnv",
    max_episode_steps=100,
)
