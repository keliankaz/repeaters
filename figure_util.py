import matplotlib.pyplot as plt
import numpy as np

def plot_phase_distribution(phases, stacked=True, ax=None):

    if ax is None:
        plt.figure(figsize=(3, 3), dpi=300)
        ax = plt.subplot(polar=True)

    number_of_divisions = 100
    bins = np.linspace(0, 2 * np.pi, number_of_divisions)

    counts = []
    for phase in phases:
        counts.append(np.histogram(phase, bins)[0])

    cummulative_counts = np.cumsum(counts, axis=0)

    if stacked:
        for i in range(len(phases)):
            ax.bar(
                0.5 * (bins[:-1] + bins[1:]),
                cummulative_counts[-i, :],
                width=2 * np.pi / number_of_divisions,
                color="grey",
                linewidth=0.2,
                edgecolor="white",
            )
    else:
        ax.bar(
            bins[:-1],
            cummulative_counts[-1, :],
            width=2 * np.pi / number_of_divisions,
            color="mediumpurple",
        )

    ax.plot(
        np.linspace(0, 2 * np.pi, 100),
        [np.mean(cummulative_counts[-1, :])] * 100,
        c="k",
        lw=0.5,
        ls="--",
    )

    ax.axvline(c="k", lw=0.5)
    ax.scatter(0, 0, c="k", s=1)
    ax.set_rticks([])
    ax.set_xticks([])
    plt.grid(b=None)

    return ax
