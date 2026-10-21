import numpy as np

states = ["In-form", "Out-of-form"]
obs_names = ["Low (0-19)", "Medium (20-49)", "High (50+)"]
start = np.array([0.5, 0.5])
trans = np.array([[0.8, 0.2],        # In-form  -> In, Out
                  [0.3, 0.7]])       # Out-form -> In, Out
emit  = np.array([[0.2, 0.4, 0.4],   # In-form:  Low, Medium, High
                  [0.6, 0.3, 0.1]])  # Out-form: Low, Medium, High

runs = [72, 34, 58, 8, 12, 25, 3, 15]
obs = [0 if r < 20 else 1 if r < 50 else 2 for r in runs]

# Forward algorithm with scaling: log P(observations)
def forward(obs):
    alpha = start * emit[:, obs[0]]
    log_p = np.log(alpha.sum()); alpha /= alpha.sum()
    for o in obs[1:]:
        alpha = (alpha @ trans) * emit[:, o]
        c = alpha.sum(); log_p += np.log(c); alpha /= c
    return log_p

# Viterbi algorithm: most likely hidden form sequence
def viterbi(obs):
    T, N = len(obs), len(states)
    V = np.zeros((T, N)); back = np.zeros((T, N), dtype=int)
    V[0] = start * emit[:, obs[0]]
    for t in range(1, T):
        for j in range(N):
            p = V[t-1] * trans[:, j]
            back[t, j] = p.argmax()
            V[t, j] = p.max() * emit[j, obs[t]]
    path = [V[-1].argmax()]
    for t in range(T-1, 0, -1):
        path.insert(0, back[t, path[0]])
    return [states[s] for s in path]

print("Runs:", runs)
print("log P(observations) =", round(forward(obs), 4))
for r, f in zip(runs, viterbi(obs)):
    print(f"{r:>4} runs -> {f}")
