import matplotlib.pyplot as plt
import numpy as np

def Plotvec2(a, b):
    # this function is used in code
    ax = plt.axes()
    ax.arrow(0, 0, *a, head_width=0.05, color='r', head_length=0.1)
    plt.text(*(a + 0.1), 'a')
    ax.arrow(0, 0, *b, head_width=0.05, color='b', head_length=0.1)
    plt.text(*(b + 0.1), 'b')

    plt.ylim(-10, 10)
    plt.xlim(-10, 10)


def Test(ar1, ar2):
    a = np.array(ar1)
    b = np.array(ar2)
    Plotvec2(a, b)
    return np.dot(a, b)


Test([1, 2], [2, 3])

print("end of program")
