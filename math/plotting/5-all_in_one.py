#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def all_in_one():

    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    fig, ax = plt.subplots(3, 2, figsize=(10, 8))

    ax[0, 0].plot(range(0, 11), y0, color='red', linestyle='-')
    ax[0, 0].set_yticks(np.arange(0, 1001, 500))
    ax[0, 0].set_ylim(0, 1000)
    ax[0, 0].set_xlim(0, 10)

    ax[0, 1].scatter(x1, y1, color="purple")
    ax[0, 1].set_yticks(np.arange(170, 191, 10))
    ax[0, 1].set_xticks(np.arange(60, 81, 10))
    ax[0, 1].set_title("Men's Height vs Weight")
    ax[0, 1].set_xlabel("Height (in)", fontsize='x-small')
    ax[0, 1].set_ylabel("Weight (lbs)", fontsize='x-small')

    ax[1,0].plot(x2, y2)
    ax[1,0].set_title("Exponential Decay of C-14", fontsize='x-small')
    ax[1,0].set_xlabel("Time (years)", fontsize='x-small')
    ax[1,0].set_ylabel("Fraction Remaining", fontsize='x-small')
    ax[1,0].set_xticks(np.arange(0, 28651, 10000))
    ax[1, 0].set_xlim(0, 28651)
    ax[1,0].set_yscale('log')

    ax[1, 1].plot(x3, y31, label="C-14", color="red", linestyle="--")
    ax[1, 1].plot(x3, y32, label="Ra-226", color="green", linestyle="-")
    ax[1, 1].legend()
    ax[1, 1].set_ylim(0, 1)
    ax[1, 1].set_yticks(np.arange(0, 1.1, 0.5))
    ax[1, 1].set_xlim(0, 20000)
    ax[1, 1].set_xticks(np.arange(0, 20001, 5000))
    ax[1, 1].set_title("Exponential Decay of Radioactive Elements", fontsize='x-small')
    ax[1, 1].set_xlabel("Time (years)", fontsize='x-small')
    ax[1, 1].set_ylabel("Fraction Remaining", fontsize='x-small')

    gs = ax[2, 0].get_gridspec()
    for a in ax[2, :]:
        a.remove()
    new_ax = fig.add_subplot(gs[2, :])
    bins = np.arange(0, 101, 10)
    new_ax.hist(student_grades, bins=bins, color='cyan', edgecolor='black')
    new_ax.set_title("Project A", fontsize='x-small')
    new_ax.set_xlabel("Grades", fontsize='x-small')
    new_ax.set_ylabel("Number of Students", fontsize='x-small')
    new_ax.set_xlim(0, 100)
    new_ax.set_ylim(0, 30)
    new_ax.set_xticks(np.arange(0, 101, 10))
    new_ax.set_yticks(np.arange(0, 31, 10))

    plt.suptitle("All in One")
    plt.tight_layout()
    plt.show()