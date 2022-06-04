import matplotlib.pyplot as plt
import numpy as np


# Plot the progress/result
# Inputs: input values, best input, rewards, current iteration no.
def plot_result(in_vals, best_in, rewards, cur_it, N_EPOCHS, N_ITER):
    plt.cla()
    plt.plot(in_vals, label='NN')
    # Best result: best_in
    y = np.empty(len(in_vals))
    y.fill(best_in)
    plt.plot(y, label='Best Value Found')
    plt.plot(rewards, label='Reward')
    # plt.scatter(range(len(in_start)), in_start, color='k', s=8, label='Epoch Starting Value')
    if cur_it < len(in_vals):
        plt.axvline(x=cur_it, color='r', linestyle='--')
    # Titles etc.
    plt.title('Rewards', fontsize=22)
    plt.xlabel('Iteration No.', fontsize=18)
    plt.ylabel('Input Value', fontsize=18)
    plt.legend()
    # plt.xticks(np.arange(0, N_EPOCHS * N_ITER + 1, N_ITER))
    step = 1
    if len(in_vals) > 10:
        step = 10
    if len(in_vals) > N_ITER:
        step = N_ITER
    plt.xticks(np.arange(0, len(in_vals), step))
    plt.grid()

    plt.pause(1e-9)
    # plt.show()