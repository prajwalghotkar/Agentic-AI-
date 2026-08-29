# Agentic AI 🧐

---

# Simple Reflex Agent:

### Agentic Vacuum-World Simulation
 
A simple Python simulation of a **vacuum-cleaning agent** moving through a 2x2 grid of rooms, cleaning any room that is dirty. It's built on the classic AI concept of a **Simple Reflex Agent**, then extended to show two smarter agent types for comparison.
 
### What it does
 
- The environment is a 2x2 grid of rooms (Room1–Room4), each either **Clean** or **Dirty**.
- An agent moves between rooms and cleans the ones that are dirty.
- The simulation is animated live using `matplotlib`, and also saved as a GIF.
- Rooms can randomly become dirty again over time, so the agent keeps working instead of stopping after one pass.
### The three agent types
 
| Agent | How it decides |
|---|---|
| `simple_reflex` | Only looks at the room it's currently in. Dirty → clean it. Clean → move to the next room. No memory. |
| `model_based_reflex` | Remembers what it has seen in every room before, and uses that memory to skip rooms it already knows are clean. |
| `utility_based` | Looks at everything it knows and goes straight to the **nearest known dirty room** — goal-directed instead of just reacting. |
 
You can switch between them by changing `AGENT_TYPE` near the top of `agentic_reflex_agent.py`.
 
### What you'll see on screen
 
- The grid of rooms, color-coded red (dirty) / green (clean), with the agent shown as a blue dot.
- A side panel showing the agent's memory of each room and simple performance stats (cleans, moves, efficiency).
### How to run
 
```bash
python3 agentic_reflex_agent.py
```
 
This will play the animation and save the result to `simulation.gif`.

https://github.com/prajwalghotkar/Agentic-AI-/blob/main/simulation.gif

---


# Learning Agents using Q-Learning
---
### Snake Agent with Q-Learning

A reinforcement learning project where a Snake-playing agent learns to survive and grow entirely through trial and error, using tabular Q-learning. No supervised labels, no hand-coded rules for movement decisions - the agent starts out acting randomly and gradually learns a policy purely from reward signals.

This document explains the theory behind Q-learning in simple terms, and then walks through how this specific project is built, why certain design decisions were made, and what the results look like.

---

## Table of Contents

1. [Reinforcement Learning in One Page](#reinforcement-learning-in-one-page)
2. [What is Q-Learning](#what-is-q-learning)
3. [The Value Function Explained](#the-value-function-explained)
4. [Q-Values Explained](#q-values-explained)
5. [The Bellman Equation](#the-bellman-equation)
6. [Exploration vs Exploitation](#exploration-vs-exploitation)
7. [Why Tabular Q-Learning (and its Limits)](#why-tabular-q-learning-and-its-limits)
8. [Project Description](#project-description)
9. [State Representation](#state-representation)
10. [Reward Design](#reward-design)
11. [Training Process](#training-process)
12. [Results](#results)
13. [How to Run](#how-to-run)
14. [Future Work](#future-work)

---

## Reinforcement Learning in One Page

Reinforcement learning is a way of teaching a program to make decisions by letting it interact with an environment and learn from the consequences of its own actions, rather than being told the correct answer directly.

Four things define this setup:

- **Agent** - the decision maker. In this project, the snake.
- **Environment** - everything the agent interacts with. Here, the grid, the food, and the walls.
- **State** - a snapshot of the situation the agent currently finds itself in.
- **Action** - a choice the agent can make. Here, move up, down, left, or right.
- **Reward** - a number the environment gives back after each action, telling the agent how good or bad that action turned out to be.

The agent's only goal is to choose actions that maximize the total reward it collects over time, not just the reward from the very next step. This is what separates reinforcement learning from simple greedy decision-making - the agent has to think about long-term consequences, not just immediate payoff.

---

## What is Q-Learning

Q-learning is a specific reinforcement learning algorithm, originally introduced by Christopher Watkins in 1989. It belongs to a family of methods called **model-free** and **off-policy** learning.

**Model-free** means the agent does not need to know the rules of the environment in advance. It does not need a map of the grid, a formula for how the snake moves, or any prior knowledge of where the food will appear. It learns purely by trying actions and observing what happens.

**Off-policy** means the agent can learn the optimal way to behave even while it is still behaving somewhat randomly or exploring. It separates "the policy I am currently using to act" from "the policy I am learning about," which is a subtle but important distinction explained further below.

At its core, Q-learning tries to answer one question, over and over, for every situation the agent finds itself in:

> "If I am in this exact situation, and I take this specific action, how good is that decision really, considering not just what happens right now but everything that follows afterward?"

The answer to that question, for every state and every action, is stored in something called a **Q-value**, and the collection of all of them is called a **Q-table**.

---

## The Value Function Explained

Before getting to Q-values specifically, it helps to understand the more general idea of a **value function**.

A value function answers a simpler question first: "If I am in this state, and I follow my current strategy from here onward, how much total reward can I expect to collect?"

This is written as:

```
V(s) = expected total future reward, starting from state s, following policy pi
```

Here, "policy" (written as pi) just means the strategy the agent is using to pick actions - for example, "always move toward the food if it is safe to do so."

The important part is that V(s) is not just about the reward from the very next step. It is about the entire future, added up. A state that looks slightly worse right now but leads to much better outcomes later should have a *higher* value than a state that feels good immediately but leads to disaster soon after.

In this project, a state near an open part of the grid with a clear path to food has high value. A state where the snake is boxed into a corner, even if it happens to be one step away from food, would have lower value if taking that food traps the snake afterward.

---

## Q-Values Explained

The value function V(s) tells you how good a *state* is. But it does not directly tell you *which action* to take to achieve that value. This is exactly the gap that Q-values fill.

A Q-value goes one level more specific:

```
Q(s, a) = expected total future reward, starting from state s,
          taking action a right now,
          and then following policy pi afterward
```

The difference between V(s) and Q(s, a) is small but critical. V(s) tells you the value of a state assuming you already have a strategy. Q(s, a) tells you the value of a state **and a specific choice**, which is exactly what an agent needs in order to actually decide what to do next.

Once you have Q(s, a) for every possible action in a given state, deciding what to do becomes simple: pick the action with the highest Q-value.

```
best_action = argmax over all actions a of Q(s, a)
```

In this project, for a given snake state (explained in detail later), there are four possible actions: up, down, left, right. The Q-table stores four numbers for every state the agent has encountered, one for each direction, and the agent simply picks whichever number is largest when it wants to act optimally.

---

## The Bellman Equation

The obvious next question is: where do these Q-values actually come from? They are not given in advance. They start at zero (or some initial value) and are updated over time using an update rule based on the **Bellman equation**, named after Richard Bellman.

The Bellman equation used in Q-learning looks like this:

```
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
```

This looks dense at first, so here is what every part means, broken down piece by piece.

- `Q(s, a)` - the current estimate of how good it is to take action a in state s, before this update.
- `r` - the immediate reward received right after taking action a. In this project, that might be +10 for eating food, or -100 for dying.
- `s'` - the next state the agent lands in after taking the action.
- `max_a' Q(s', a')` - the best possible Q-value available from that next state, assuming the agent acts optimally from then on.
- `gamma` (discount factor, between 0 and 1) - controls how much the agent cares about future rewards versus immediate ones. A gamma close to 1 means the agent thinks far ahead. A gamma close to 0 means the agent only cares about the very next reward.
- `alpha` (learning rate, between 0 and 1) - controls how much each new experience is allowed to change the existing estimate. A high alpha means the agent updates its beliefs aggressively based on the latest experience. A low alpha means it updates cautiously, averaging in new information slowly.

The term inside the brackets, `r + gamma * max_a' Q(s', a') - Q(s, a)`, is called the **TD error** (temporal difference error). It measures the gap between what the agent expected to happen and what it now realizes is a better estimate, given the reward it just received and the value of where it ended up.

In plain language, the whole update rule says:

> "My old estimate of how good this move was is being nudged closer to a better estimate - the reward I actually got, plus a discounted guess of how good things look from here onward."

Every time the agent takes an action anywhere in the environment, this update happens, and over thousands of these small corrections, the Q-values slowly settle into accurate estimates of true long-term value. This is why Q-learning typically needs many episodes of experience before the agent behaves sensibly - each individual update is only a small nudge, not a final answer.

---

## Exploration vs Exploitation

There is a fundamental tension in every reinforcement learning problem. If the agent always chooses the action with the highest known Q-value, it may get stuck repeating a mediocre strategy forever, because it never tries anything else long enough to discover something better. This is called pure **exploitation**.

On the other hand, if the agent always chooses actions randomly, it never actually uses anything it has learned, and its behavior stays poor forever. This is called pure **exploration**.

The standard solution is **epsilon-greedy** action selection:

```
with probability epsilon: choose a random action (explore)
with probability 1 - epsilon: choose the action with the highest Q-value (exploit)
```

Early in training, epsilon is kept high, so the agent explores heavily and gathers a wide variety of experiences. As training progresses, epsilon is gradually decreased, so the agent leans more and more on what it has actually learned. This project uses this exact schedule, starting at full exploration and decaying down to a small residual exploration rate that is never fully removed, since a small amount of exploration helps the agent avoid getting permanently stuck in a suboptimal habit.

---

## Why Tabular Q-Learning (and its Limits)

The version of Q-learning used here is called **tabular** Q-learning, because the Q-values are stored explicitly in a table (a dictionary, in this implementation) with one row per state and one column per action.

This approach works well when the number of distinct states is small enough to enumerate and store in memory. It has a major advantage: it is simple, transparent, and easy to inspect - you can literally open the Q-table and see exactly what the agent has learned about any given situation.

Its limitation is that it does not scale to very large or continuous state spaces. If the state were, for example, the raw pixel image of the entire grid, the number of possible states would be astronomically large, and a table could never store a value for every one of them. This is the exact motivation behind Deep Q-Networks (DQN), which replace the table with a neural network that can generalize across similar states it has never exactly seen before. That extension is listed under Future Work below.

---

## Project Description

I built this project to understand reinforcement learning from first principles by implementing Q-learning on a problem simple enough to reason about by hand, but rich enough to expose the real challenges of the algorithm - designing a good state representation, shaping rewards correctly, and balancing exploration against exploitation.

The environment is a classic Snake game on a fixed-size grid. The snake starts as a single segment in the middle of the board, food spawns at a random empty cell, and the snake must navigate toward it, growing by one segment each time it eats, while avoiding collisions with the walls or its own body.

The interesting part of this project was not the game itself, but getting the Q-learning setup right. My first version of the state representation used only the raw coordinates of the snake's head on the grid. It became clear fairly quickly that this does not work, because the reward the agent receives depends heavily on where the food currently is, and the food's position was not part of the state at all. Two visits to the exact same grid cell could require completely different correct actions depending on where the food happened to be that episode, which breaks the core assumption Q-learning relies on - that the future depends only on the current state and action, not on hidden information the agent cannot see.

I redesigned the state to include the food's position relative to the snake's head, along with local danger indicators showing whether moving straight, left, or right relative to the snake's current heading would result in a collision, plus the current heading itself. This state representation is small enough to keep fully tabular, while still giving the agent everything it actually needs to make a locally and strategically sound decision.

Beyond the core algorithm, I added a set of practical engineering pieces that turn this from a script into something closer to a small research pipeline: a headless training loop so training does not slow down with rendering, reward shaping based on distance to food to speed up early learning, a small experience replay buffer to stabilize updates by reusing past transitions, saving and loading trained agents so training and evaluation are separate steps, a metrics dashboard tracking reward, food eaten, survival time, and exploration rate across training, and a hyperparameter sweep comparing different learning rates, discount factors, and exploration decay schedules against each other rather than tuning by guesswork.

---

## State Representation

Each state is represented as a tuple of six values:

```
(food_dx, food_dy, danger_straight, danger_left, danger_right, current_direction)
```

- `food_dx`, `food_dy` - the sign of the difference between the food's position and the snake's head position along each axis, taking values of -1, 0, or 1. This tells the agent roughly which direction the food is in, without needing exact distances.
- `danger_straight`, `danger_left`, `danger_right` - binary flags indicating whether moving in that direction relative to the snake's current heading would immediately result in a collision with a wall or the snake's own body.
- `current_direction` - the direction the snake is currently moving in, since "left" and "right" are relative to heading, not absolute grid directions.

This keeps the total number of distinct states small enough that a table-based approach remains practical, while making sure the state is Markovian - the reward and outcome of any action depend only on this state, not on anything outside it.

---

## Reward Design

The reward signal combines a sparse component and a small shaping component:

- Eating food: a large positive reward, since this is the actual objective.
- Colliding with a wall or itself: a large negative reward, since this ends the episode and represents total failure.
- A small step penalty on every move that neither eats food nor causes a collision, to discourage the agent from wandering aimlessly forever.
- A small shaping bonus or penalty based on whether the move brought the snake closer to or farther from the food, compared to the previous step.

The shaping term is kept deliberately small relative to the main food and collision rewards. Its only purpose is to give the agent a denser signal early in training, when it would otherwise wander randomly for a long time before ever stumbling onto food by chance. Without shaping, learning is still possible, but noticeably slower, especially on larger grids.

---

## Training Process

Training runs for a fixed number of episodes. In each episode:

1. The environment resets - the snake returns to its starting position, and a new food location is chosen.
2. The agent observes the current state and chooses an action, using epsilon-greedy selection.
3. The environment applies the action, returns a reward, the next state, and whether the episode has ended.
4. The Q-table is updated using the Bellman equation described above, both from this immediate transition and from a batch of past transitions sampled from the replay buffer.
5. This repeats until the snake collides with something or the episode times out.
6. Epsilon is decayed slightly after every episode.

Throughout training, reward, food eaten, steps survived, and epsilon are logged per episode, so progress can be inspected afterward rather than just trusted blindly.

---

## Results

Across training, the agent's average reward over rolling windows of episodes moved from strongly negative (frequent, early collisions with very little food eaten) to consistently positive, with the average amount of food eaten per episode increasing by roughly an order of magnitude compared to early training. This is the expected signature of a working Q-learning setup - a noisy but clearly upward trend in reward and food collected, alongside a smooth decay in exploration rate, rather than an agent that never improves or that improves and then collapses.

Actual numbers, plots, and the hyperparameter sweep comparison are generated directly by running the accompanying notebook, since they depend on the random seed and number of training episodes used in a given run.

---

## How to Run

1. Install dependencies: `numpy`, `matplotlib`, `tqdm`.
2. Open the notebook and run all cells in order.
3. Training will print progress periodically and produce plots for reward, food eaten, steps survived, and epsilon decay once complete.
4. A trained agent is saved to disk automatically, and can be reloaded later without retraining.
5. The final cells produce an animated replay of the trained agent playing a full episode using its learned policy.

---

