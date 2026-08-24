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
