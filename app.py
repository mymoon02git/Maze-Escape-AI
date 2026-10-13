
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from maze_rl import MazeEnv, TabularAgent, generate_maze, shortest_path_length

st.set_page_config(page_title="Maze Escape AI", page_icon="🧩", layout="wide")


# ----------------------------------------------------------------------
# Drawing helper
# ----------------------------------------------------------------------
def draw_maze(env, path=None, agent_pos=None, q_table=None, show_heatmap=False):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(env.grid, cmap=ListedColormap(["#f8fafc", "#1e293b"]))

    if show_heatmap and q_table is not None:
        vmax = q_table.max(axis=2).astype(float)
        vmax = np.ma.masked_where(env.grid == 1, vmax)
        ax.imshow(vmax, cmap="viridis", alpha=0.75)

    if path:
        rows, cols = zip(*path)
        ax.plot(cols, rows, color="#f97316", linewidth=2.5, zorder=3)

    ax.scatter(env.start[1], env.start[0], c="#22c55e", s=140, marker="o", zorder=4, label="Start")
    ax.scatter(env.goal[1], env.goal[0], c="#ef4444", s=180, marker="*", zorder=4, label="Goal")
    if agent_pos is not None:
        ax.scatter(agent_pos[1], agent_pos[0], c="#3b82f6", s=140, marker="s", zorder=5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False)
    plt.tight_layout()
    return fig


def moving_average(x, window):
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


# ----------------------------------------------------------------------
# Sidebar - settings
# ----------------------------------------------------------------------
st.title("🧩 Maze Escape AI")
st.caption("A reinforcement learning agent learns to escape a maze from scratch, using trial and error.")

with st.sidebar:
    st.header("Maze")
    n_cells = st.slider("Maze size (cells)", 4, 14, 8)
    seed = st.number_input("Maze seed", 0, 9999, 42)
    loops = st.slider("Extra loops (multiple routes)", 0.0, 0.4, 0.10, 0.05)

    st.header("Algorithm")
    algorithm = st.radio("Learning algorithm", ["Q-learning", "SARSA"])
    episodes = st.slider("Training episodes", 200, 5000, 1500, 100)
    alpha = st.slider("Learning rate (α)", 0.01, 1.0, 0.10, 0.01)
    gamma = st.slider("Discount factor (γ)", 0.50, 0.999, 0.99, 0.001)
    eps_decay = st.slider("Epsilon decay", 0.980, 0.9995, 0.995, 0.0005, format="%.4f")

    generate_clicked = st.button(" Generate new maze", use_container_width=True)

# ----------------------------------------------------------------------
# Session state - keep maze and trained agent between button clicks
# ----------------------------------------------------------------------
maze_key = (n_cells, seed, loops)
if generate_clicked or st.session_state.get("maze_key") != maze_key:
    grid = generate_maze(n_cells, int(seed), loops)
    st.session_state.maze_key = maze_key
    st.session_state.env = MazeEnv(grid)
    st.session_state.agent = None
    st.session_state.stats = None

env = st.session_state.env

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Maze")
    maze_slot = st.empty()
    maze_slot.pyplot(draw_maze(env, agent_pos=env.start))
    plt.close("all")

with right:
    st.subheader("Training")
    train_clicked = st.button(" Train agent", type="primary", use_container_width=True)
    progress_bar = st.progress(0.0)
    status = st.empty()

    if train_clicked:
        agent = TabularAgent(env, algorithm=algorithm, alpha=alpha, gamma=gamma,
                             epsilon_decay=eps_decay)

        def on_progress(done, total, eps):
            progress_bar.progress(done / total)
            status.write(f"Episode {done}/{total}  |  ε = {eps:.3f}")

        t0 = time.time()
        stats = agent.train(episodes, progress_callback=on_progress)
        status.success(f"Training finished in {time.time() - t0:.1f}s")
        st.session_state.agent = agent
        st.session_state.stats = stats

    agent = st.session_state.get("agent")
    stats = st.session_state.get("stats")

    if stats:
        window = max(10, episodes // 50)
        chart_df = pd.DataFrame({
            "Reward (moving avg)": moving_average(stats["rewards"], window),
        })
        st.markdown("**Learning curve - reward per episode**")
        st.line_chart(chart_df)

        steps_df = pd.DataFrame({
            "Steps (moving avg)": moving_average(stats["steps"], window),
        })
        st.markdown("**Steps taken per episode** (lower = more efficient)")
        st.line_chart(steps_df)

        recent = stats["success"][-100:]
        st.metric("Success rate (last 100 episodes)", f"{100 * np.mean(recent):.0f}%")

# ----------------------------------------------------------------------
# Results: run the trained agent
# ----------------------------------------------------------------------
if agent is not None:
    st.divider()
    st.subheader("Watch the trained agent escape")

    path, reached = agent.greedy_path()
    optimal = shortest_path_length(env.grid, env.start, env.goal)

    c1, c2, c3 = st.columns(3)
    c1.metric("Reached goal?", "Yes " if reached else "No ❌")
    c2.metric("Agent's path length", len(path) - 1)
    c3.metric("Optimal (BFS) length", optimal)

    opt_col, view_col = st.columns([1, 1])
    with opt_col:
        show_heat = st.checkbox("Show learned value heatmap (max Q per cell)")
        speed = st.slider("Animation speed (seconds per step)", 0.0, 0.3, 0.05, 0.01)
        animate = st.button(" Animate escape")

    with view_col:
        final_slot = st.empty()
        final_slot.pyplot(draw_maze(env, path=path, agent_pos=path[-1],
                                    q_table=agent.q, show_heatmap=show_heat))
        plt.close("all")

    if animate:
        for i in range(1, len(path) + 1):
            maze_slot.pyplot(draw_maze(env, path=path[:i], agent_pos=path[i - 1]))
            plt.close("all")
            time.sleep(speed)

    if not reached:
        st.warning("The agent didn't find the exit. Try more episodes, a smaller maze, "
                   "or a slower epsilon decay (closer to 1).")

    with st.expander("How does this work?"):
        st.markdown("""
**State:** the agent's (row, col) position.  **Actions:** up, down, left, right.
**Rewards:** +100 for the goal, -1 per step, -5 for bumping a wall.

The agent keeps a **Q-table**, one value per (cell, action), and updates it after every move:

`Q(s,a) ← Q(s,a) + α [ r + γ · next_value − Q(s,a) ]`

- **Q-learning** uses `max Q(s',·)`, the best possible next move (off-policy).
- **SARSA** uses `Q(s',a')`, the move it will really take, exploration included (on-policy).

**ε-greedy:** early on the agent explores randomly (ε = 1). As ε decays, it exploits what it learned.
        """)
