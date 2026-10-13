"""
maze_rl.py  -  Core reinforcement learning logic for Maze Escape AI
Contains: maze generator, environment, tabular agent (Q-learning / SARSA)
"""
import random
import numpy as np

# Actions: (row change, col change)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Up", "Down", "Left", "Right"]


# ----------------------------------------------------------------------
# 1. MAZE GENERATION  (randomised depth-first search)
# ----------------------------------------------------------------------
def generate_maze(n_cells=8, seed=42, loops=0.10):
    """
    Returns a 2D numpy array: 1 = wall, 0 = open path.
    n_cells : maze is (2*n_cells+1) x (2*n_cells+1) grid
    loops   : fraction of extra walls to knock down, so the maze has
              multiple routes (0 = perfect maze with one single path)
    """
    rng = random.Random(seed)
    size = 2 * n_cells + 1
    grid = np.ones((size, size), dtype=int)

    grid[1, 1] = 0
    stack = [(1, 1)]
    while stack:
        r, c = stack[-1]
        neighbours = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 0 < nr < size - 1 and 0 < nc < size - 1 and grid[nr, nc] == 1:
                neighbours.append((nr, nc, dr, dc))
        if neighbours:
            nr, nc, dr, dc = rng.choice(neighbours)
            grid[r + dr // 2, c + dc // 2] = 0   # carve wall between
            grid[nr, nc] = 0                     # carve new cell
            stack.append((nr, nc))
        else:
            stack.pop()

    # Knock down some walls that separate two open cells -> creates loops
    candidates = []
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if grid[r, c] == 1:
                vertical = grid[r - 1, c] == 0 and grid[r + 1, c] == 0
                horizontal = grid[r, c - 1] == 0 and grid[r, c + 1] == 0
                if vertical or horizontal:
                    candidates.append((r, c))
    rng.shuffle(candidates)
    for r, c in candidates[: int(len(candidates) * loops)]:
        grid[r, c] = 0
    return grid


# ----------------------------------------------------------------------
# 2. ENVIRONMENT
# ----------------------------------------------------------------------
class MazeEnv:
    """
    State  : (row, col) of the agent
    Actions: 0=Up 1=Down 2=Left 3=Right
    Reward : +100 reaching goal, -5 hitting a wall, -1 every normal step
    """
    GOAL_REWARD = 100.0
    WALL_PENALTY = -5.0
    STEP_PENALTY = -1.0

    def __init__(self, grid):
        self.grid = grid
        self.h, self.w = grid.shape
        self.start = (1, 1)
        self.goal = (self.h - 2, self.w - 2)
        self.max_steps = self.h * self.w * 2
        self.reset()

    def reset(self):
        self.pos = self.start
        self.steps = 0
        return self.pos

    def step(self, action):
        dr, dc = ACTIONS[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        self.steps += 1

        if self.grid[nr, nc] == 1:              # hit wall -> stay in place
            reward = self.WALL_PENALTY
        else:
            self.pos = (nr, nc)
            reward = self.STEP_PENALTY

        done = False
        if self.pos == self.goal:
            reward = self.GOAL_REWARD
            done = True
        elif self.steps >= self.max_steps:
            done = True
        return self.pos, reward, done


# ----------------------------------------------------------------------
# 3. AGENT  (tabular Q-learning or SARSA)
# ----------------------------------------------------------------------
class TabularAgent:
    def __init__(self, env, algorithm="Q-learning", alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, seed=0):
        self.env = env
        self.algorithm = algorithm
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q = np.zeros((env.h, env.w, len(ACTIONS)))
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(ACTIONS)))
        values = self.q[state]
        best = np.flatnonzero(values == values.max())   # random tie-break
        return int(self.rng.choice(best))

    def train(self, episodes=2000, progress_callback=None):
        """Trains the agent. Returns dict of per-episode statistics."""
        rewards, steps_list, successes = [], [], []

        for ep in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)      # needed for SARSA
            total_reward = 0.0
            done = False

            while not done:
                next_state, reward, done = self.env.step(action)
                next_action = self.choose_action(next_state)

                if self.algorithm == "Q-learning":
                    # off-policy: look at the BEST next action
                    target_next = self.q[next_state].max()
                else:
                    # SARSA, on-policy: use the action we will REALLY take
                    target_next = self.q[next_state][next_action]

                if done and next_state == self.env.goal:
                    target_next = 0.0               # terminal state

                td_target = reward + self.gamma * target_next
                td_error = td_target - self.q[state][action]
                self.q[state][action] += self.alpha * td_error

                state, action = next_state, next_action
                total_reward += reward

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            rewards.append(total_reward)
            steps_list.append(self.env.steps)
            successes.append(1 if state == self.env.goal else 0)

            if progress_callback and (ep % 20 == 0 or ep == episodes - 1):
                progress_callback(ep + 1, episodes, self.epsilon)

        return {"rewards": rewards, "steps": steps_list, "success": successes}

    def greedy_path(self):
        """Follow the learned policy (no exploration). Returns (path, reached_goal)."""
        state = self.env.reset()
        path = [state]
        visited = {state}
        for _ in range(self.env.max_steps):
            action = self.choose_action(state, greedy=True)
            state, _, done = self.env.step(action)
            path.append(state)
            if state == self.env.goal:
                return path, True
            if state in visited or done:      # stuck in a loop
                return path, False
            visited.add(state)
        return path, False


def shortest_path_length(grid, start, goal):
    """BFS baseline - the true optimal number of moves, for comparison."""
    from collections import deque
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (r, c), d = q.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if grid[nr, nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append(((nr, nc), d + 1))
    return None
