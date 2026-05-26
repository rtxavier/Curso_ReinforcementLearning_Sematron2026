# Curso_ReinforcementLearning_Sematron2026
# Prof. Dr. Ricardo Taoni Xavier

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import deque

# =========================================================
# CONFIGURAÇÕES DO AMBIENTE
# =========================================================

grid_size = 8

start = (0, 0)
goal = (7, 7)

obstacles = [
    (1, 1), (1, 2), (1, 5),
    (2, 5), (3, 1), (3, 2),
    (3, 3), (4, 5), (5, 5),
    (6, 2), (6, 3)
]

actions = {
    0: (-1, 0),   # cima
    1: (1, 0),    # baixo
    2: (0, -1),   # esquerda
    3: (0, 1)     # direita
}

action_symbols = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→"
}

# =========================================================
# Q-TABLE
# =========================================================

q_table = np.zeros((grid_size, grid_size, 4))

# =========================================================
# HIPERPARÂMETROS
# =========================================================

alpha = 0.15
gamma = 0.95

epsilon = 1.0
epsilon_decay = 0.97
epsilon_min = 0.03

episodes = 100
max_steps = 120

# =========================================================
# VARIÁVEIS GLOBAIS
# =========================================================

episode = 0
state = start
step_counter = 0
total_reward = 0

rewards_history = []

success_history = deque(maxlen=100)

current_path = [start]

training_finished = False

success_rate = 0

# =========================================================
# FUNÇÕES DO AMBIENTE
# =========================================================

def is_valid_position(pos):

    x, y = pos

    if x < 0 or x >= grid_size:
        return False

    if y < 0 or y >= grid_size:
        return False

    if pos in obstacles:
        return False

    return True


def environment_step(state, action):

    dx, dy = actions[action]

    next_state = (
        state[0] + dx,
        state[1] + dy
    )

    # colisão
    if not is_valid_position(next_state):
        return state, -10, False

    # objetivo
    if next_state == goal:
        return next_state, 100, True

    # movimento normal
    return next_state, -1, False


def choose_action(state):

    global epsilon

    # exploração
    if random.random() < epsilon:
        return random.choice(list(actions.keys()))

    # exploitation
    x, y = state
    return np.argmax(q_table[x, y])


def get_best_action(pos):

    x, y = pos
    return np.argmax(q_table[x, y])


def get_policy_strength(pos):

    x, y = pos

    values = q_table[x, y]

    return np.max(values) - np.min(values)

# =========================================================
# CONFIGURAÇÃO VISUAL
# =========================================================

plt.style.use("dark_background")

fig = plt.figure(figsize=(18, 8))

gs = fig.add_gridspec(2, 4)

ax_grid = fig.add_subplot(gs[:, 0])

ax_reward = fig.add_subplot(gs[0, 1:3])

ax_reward_bar = fig.add_subplot(gs[0, 3])

ax_info = fig.add_subplot(gs[1, 1:])

fig.suptitle(
    "Q-Learning AO VIVO — Robô aprendendo por tentativa e erro",
    fontsize=18,
    fontweight="bold"
)

# =========================================================
# DESENHO DO GRID
# =========================================================

def draw_grid():

    ax_grid.clear()

    ax_grid.set_title(
        "Política aprendida em tempo real",
        fontsize=14
    )

    ax_grid.set_xlim(-0.5, grid_size - 0.5)
    ax_grid.set_ylim(-0.5, grid_size - 0.5)

    ax_grid.set_xticks(range(grid_size))
    ax_grid.set_yticks(range(grid_size))

    ax_grid.grid(True, alpha=0.3)

    ax_grid.invert_yaxis()

    for i in range(grid_size):

        for j in range(grid_size):

            pos = (i, j)

            # obstáculos
            if pos in obstacles:

                ax_grid.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        color="gray"
                    )
                )

                ax_grid.text(
                    j,
                    i,
                    "X",
                    ha="center",
                    va="center",
                    fontsize=14
                )

                continue

            # objetivo
            if pos == goal:

                ax_grid.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        color="green",
                        alpha=0.6
                    )
                )

                ax_grid.text(
                    j,
                    i,
                    "G",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold"
                )

                continue

            # início
            if pos == start:

                ax_grid.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        color="blue",
                        alpha=0.35
                    )
                )

            # =================================================
            # POLÍTICA VISUAL
            # =================================================

            strength = get_policy_strength(pos)

            alpha_strength = min(0.7, strength / 120)

            final_alpha = max(
                0.15,
                min(1.0, 0.2 + alpha_strength)
            )

            if np.max(q_table[i, j]) != 0:

                best_action = get_best_action(pos)

                color_intensity = min(
                    1.0,
                    strength / 60
                )

                arrow_color = (
                    min(1.0, 0.2 + color_intensity),
                    0.8,
                    1.0
                )

                ax_grid.text(
                    j,
                    i,
                    action_symbols[best_action],
                    ha="center",
                    va="center",
                    fontsize=18,
                    alpha=final_alpha,
                    color=arrow_color,
                    fontweight="bold"
                )

    # caminho atual
    if len(current_path) > 1:

        ys = [p[0] for p in current_path]
        xs = [p[1] for p in current_path]

        ax_grid.plot(
            xs,
            ys,
            linewidth=2,
            alpha=0.7,
            color="cyan"
        )

    # agente
    ax_grid.plot(
        state[1],
        state[0],
        "o",
        markersize=20,
        color="red"
    )

    ax_grid.text(
        state[1],
        state[0],
        "A",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

# =========================================================
# GRÁFICO DE EVOLUÇÃO
# =========================================================

def draw_rewards():

    ax_reward.clear()

    ax_reward.set_title(
        "Evolução do aprendizado",
        fontsize=14
    )

    ax_reward.set_xlabel("Episódio")
    ax_reward.set_ylabel("Reward")

    ax_reward.grid(True, alpha=0.3)

    if len(rewards_history) > 0:

        ax_reward.plot(
            rewards_history,
            linewidth=2,
            color="cyan",
            alpha=0.6
        )

        # média móvel
        if len(rewards_history) >= 10:

            moving_avg = np.convolve(
                rewards_history,
                np.ones(10) / 10,
                mode="valid"
            )

            ax_reward.plot(
                range(9, len(rewards_history)),
                moving_avg,
                linewidth=4,
                color="yellow"
            )

# =========================================================
# DASHBOARD INTUITIVA
# =========================================================

def draw_reward_dashboard():

    ax_reward_bar.clear()

    ax_reward_bar.set_title(
        "Estado do Aprendizado",
        fontsize=14,
        fontweight="bold"
    )

    ax_reward_bar.set_xlim(0, 100)

    ax_reward_bar.set_ylim(0, 1)

    ax_reward_bar.set_xticks([])

    ax_reward_bar.set_yticks([])

    # =====================================================
    # SCORE DE APRENDIZADO
    # =====================================================

    learning_score = max(
        0,
        min(
            100,
            success_rate * 1.2
        )
    )

    # =====================================================
    # DEFINE ESTADO VISUAL
    # =====================================================

    if learning_score < 20:

        state_text = "😵 Perdido"
        color = "red"

    elif learning_score < 40:

        state_text = "🤔 Explorando"
        color = "orange"

    elif learning_score < 70:

        state_text = "🧠 Aprendendo"
        color = "yellow"

    else:

        state_text = "🚀 Dominando"
        color = "lime"

    # fundo da barra
    ax_reward_bar.barh(
        0.5,
        100,
        height=0.3,
        color="white",
        alpha=0.08
    )

    # barra principal
    ax_reward_bar.barh(
        0.5,
        learning_score,
        height=0.3,
        color=color,
        alpha=0.85
    )

    # texto principal
    ax_reward_bar.text(
        50,
        0.82,
        state_text,
        ha="center",
        fontsize=18,
        fontweight="bold"
    )

    ax_reward_bar.text(
        50,
        0.18,
        f"{learning_score:.0f}% da política aprendida",
        ha="center",
        fontsize=11
    )

    # marcadores
    for x in [20, 40, 70]:

        ax_reward_bar.axvline(
            x,
            color="white",
            alpha=0.2,
            linestyle="--"
        )

# =========================================================
# PAINEL DE TEXTO
# =========================================================

def draw_info():

    global success_rate

    ax_info.clear()

    ax_info.axis("off")

    success_rate = 0

    if len(success_history) > 0:

        success_rate = (
            sum(success_history)
            / len(success_history)
        ) * 100

    info_text = f"""
EPISÓDIO: {episode}/{episodes}

EPSILON (exploração): {epsilon:.3f}

REWARD DO EPISÓDIO: {total_reward}

PASSOS: {step_counter}/{max_steps}

SUCESSO:
{success_rate:.1f}%

INTERPRETAÇÃO:

• O agente explora o ambiente
• Erros geram punições
• Acertos geram recompensa
• A política emerge gradualmente
• As setas mostram a decisão aprendida
"""

    ax_info.text(
        0.02,
        0.95,
        info_text,
        va="top",
        fontsize=13,
        family="monospace"
    )

# =========================================================
# LOOP DE TREINAMENTO
# =========================================================

def update(frame):

    global episode
    global state
    global step_counter
    global total_reward
    global epsilon
    global training_finished
    global current_path

    if training_finished:

        draw_grid()
        draw_rewards()
        draw_reward_dashboard()
        draw_info()

        return

    steps_per_frame = 25

    for _ in range(steps_per_frame):

        if episode >= episodes:

            training_finished = True
            break

        # escolhe ação
        action = choose_action(state)

        # executa ação
        next_state, reward, done = environment_step(
            state,
            action
        )

        x, y = state
        nx, ny = next_state

        # atualização Q-Learning
        old_q = q_table[x, y, action]

        next_max_q = np.max(q_table[nx, ny])

        q_table[x, y, action] = old_q + alpha * (
            reward +
            gamma * next_max_q -
            old_q
        )

        # atualiza estado
        state = next_state

        current_path.append(state)

        total_reward += reward

        step_counter += 1

        # fim do episódio
        if done or step_counter >= max_steps:

            rewards_history.append(total_reward)

            success_history.append(
                1 if done else 0
            )

            episode += 1

            state = start

            step_counter = 0

            total_reward = 0

            current_path = [start]

            epsilon = max(
                epsilon_min,
                epsilon * epsilon_decay
            )

    # desenha dashboard
    draw_grid()
    draw_rewards()
    draw_reward_dashboard()
    draw_info()

# =========================================================
# ANIMAÇÃO
# =========================================================

ani = animation.FuncAnimation(
    fig,
    update,
    interval=50,
    cache_frame_data=False
)

plt.tight_layout()

plt.show()
