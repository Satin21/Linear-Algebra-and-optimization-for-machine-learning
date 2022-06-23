import matplotlib.pyplot as plt
import numpy as np


# Plot the progress/results
# Inputs: input values, best input, rewards, current iteration no.
def plot_result(losses: list, accuracy: list, cur_it: int, n_iter: int, fname: str = None):
    plt.clf()
    ax1 = plt.gca()

    # Plot the accuracy
    color = 'tab:red'
    l1 = ax1.plot(accuracy, '--', color=color, label='Accuracy')
    ax1.set_xlabel('Iteration No.', fontsize=18)
    ax1.set_ylabel('Accuracy', fontsize=18, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot the losses
    ax2 = ax1.twinx()
    color = 'tab:blue'
    l2 = ax2.plot(losses, color=color, label='Loss')
    ax2.set_ylabel('Loss', fontsize=18, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    if cur_it < n_iter:
        plt.axvline(x=cur_it, color='r', linestyle='--')

    # Titles etc.
    plt.title('Losses & Accuracy', fontsize=22)  # Rewards or losses
    lns = l1 + l2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs)
    plt.grid()

    # x ticks
    step = round(n_iter / 100) * 10
    if cur_it > n_iter:
        step = n_iter
    plt.xticks(np.arange(0, n_iter + 1, step))

    if cur_it < n_iter:
        plt.pause(1e-9)
    else:
        plt.show()
        if fname is not None:
            fig.savefig(fname, format='png', dpi=100, bbox_inches='tight')
