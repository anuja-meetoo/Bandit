This repository provides source code for the following bandit-style algorithms for wireless network selection.

* Smart EXP3 is a novel bandit-style algorithm that (a) retains the good theoretical properties of EXP3, i.e. minimizing regret and converging to (weak) Nash equilibrium, (b) bounds the number of switches, and (c) yields significantly better performance in practice. It stabilizes at the optimal state, achieves fairness among devices and gracefully deals with transient behaviors. In real world experiments, it can achieve 18% faster download over alternate strategies.

* Co-Bandit is a novel collaborative bandit-style algorithm that allows devices to occasionally share their observations. It stabilizes to the optimal state relatively quickly with only a very small amount of information, even if the latter is received with a delay — it is adequate to nudge other devices to select the right network and yields significantly faster stabilization at the optimal state.

* Smart Periodic EXP4 is a novel bandit-style algorithm that can effectively learn periodic patterns in network data rates.

## Note: The code is not optimized and this repository will not be maintained.
