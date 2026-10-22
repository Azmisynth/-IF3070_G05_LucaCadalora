from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def _finish(fig, ax, save_path):
    ax.grid(alpha=.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if ax.lines:
        ax.legend()
    fig.tight_layout()
    if save_path is not None:
        destination = Path(save_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=150, bbox_inches="tight")
    return fig


def plot_objective(history, save_path=None, title="Objective vs iterasi"):
    if not history:
        raise ValueError("History kosong")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = [h["iteration"] for h in history]
    y = [h["objective"] for h in history]
    ax.plot(x, y, label="Objective saat ini", color="#24748e", marker="." if len(x) < 30 else None)
    if "best_objective" in history[0]:
        ax.plot(x, [h["best_objective"] for h in history], label="Terbaik sejauh ini", ls="--")
    elif "restart" in history[0]:
        best = []
        for value in y:
            best.append(max(value, best[-1]) if best else value)
        ax.plot(x, best, label="Terbaik seluruh restart", ls="--")
        for prev, cur in zip(history, history[1:]):
            if prev["restart"] != cur["restart"]:
                ax.axvline(cur["iteration"], color="grey", alpha=.3, ls=":")
    ax.set(xlabel="Indeks history (termasuk awal tiap restart)" if "restart" in history[0] else "Iterasi",
           ylabel="Objective", title=title)
    return _finish(fig, ax, save_path)


def plot_sa_probability(history, save_path=None):

    samples = [h for h in history if h["iteration"] > 0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot([h["iteration"] for h in samples], [h["acceptance_probability"] for h in samples],
            label="Probabilitas penerimaan kandidat", color="#9a4f96")
    rejected = [h for h in samples if not h["accepted"]]
    if rejected:
        ax.scatter([h["iteration"] for h in rejected], [h["acceptance_probability"] for h in rejected],
                   s=10, color="#b63737", label="Ditolak")
    ax.set(xlabel="Iterasi", ylabel="Probabilitas", ylim=(-.03, 1.03),
           title="Simulated Annealing - probabilitas penerimaan")
    return _finish(fig, ax, save_path)


def plot_ga_fitness(history, save_path=None):
    if not history:
        raise ValueError("History kosong")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = [h["generation"] for h in history]
    ax.plot(x, [h["max_fitness"] for h in history], label="Fitness maksimum", color="#24748e")
    ax.plot(x, [h["average_fitness"] for h in history], label="Fitness rata-rata", color="#cd8239")
    ax.set(xlabel="Generasi", ylabel="Fitness", title="Genetic Algorithm - fitness populasi")
    return _finish(fig, ax, save_path)
