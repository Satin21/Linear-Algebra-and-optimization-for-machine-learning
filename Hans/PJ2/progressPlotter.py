import matplotlib.pyplot as plt
import numpy as np


# Plot the progress/results
# Inputs: input values, best input, rewards, current iteration no.
def plot_result(rewards, cur_it, n_iter, n_epochs=1, in_vals=None, best_in=None):
    plt.cla()
    if in_vals is not None:
        plt.plot(in_vals, label='Input')

    # Best result: best_in
    if best_in is not None:
        y = np.empty(len(in_vals))
        y.fill(best_in)
        plt.plot(y, label='Best Value Found')

    plt.plot(rewards, label='Loss')  # Reward or loss
    # plt.scatter(range(len(in_start)), in_start, color='k', s=8, label='Epoch Starting Value')
    # if cur_it < len(in_vals):
    if cur_it < n_iter:
        plt.axvline(x=cur_it, color='r', linestyle='--')

    # Titles etc.
    plt.title('Losses', fontsize=22)  # Rewards or losses
    plt.xlabel('Iteration No.', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.legend()
    plt.grid()
    # plt.xticks(np.arange(0, n_epochs * N_ITER + 1, n_iter))
    step = round(n_iter / 100) * 10
    if cur_it > n_iter:
        step = n_iter
    plt.xticks(np.arange(0, n_iter + 1, step))

    if cur_it < n_iter:
        plt.pause(1e-9)
    else:
        plt.show()
