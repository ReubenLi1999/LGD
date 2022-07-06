import numpy as np
import matplotlib.pyplot as plt


def main():
    x = np.asarray([184, 211])
    y = np.asarray([38.049, 73.5433])

    fig, ax = plt.subplots(figsize=(16, 8))  # 2020-07-02
    plt.scatter(x, y, marker="*", s=100)
    ax.tick_params(labelsize=25, width=2.9)
    ax.set_xlabel('Day in year [day]', fontsize=20)
    ax.set_xlim([1, 366])
    ax.set_ylim([-10, 150])
    ax.yaxis.get_offset_text().set_fontsize(24)
    ax.set_ylabel(r'Water storage [Gt]', fontsize=20)
    ax.grid(True, which='both', ls='dashed', color='0.5', linewidth=0.6)
    plt.setp(ax.spines.values(), linewidth=3)
    plt.tight_layout()
    plt.show()


if __name__  == "__main__":
    main()
