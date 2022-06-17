import numpy as np
import warnings
from matplotlib import pyplot as plt

# def func(x, y):
#     return np.sin(x ** 2) * np.exp(x) + 3 * x * y ** 2
#
# def derivative_x(x, y):
#     return 2 * x * np.exp(x) * np.cos(x ** 2) + 3 * y ** 2 + np.exp(x) * np.sin(x ** 2)
#
#
# def derivative_y(x, y):
#     return 6 * y * x


# sin(x^2)*exp(x) + 3 * x * y^2x

def func(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.sin(x ** 2) * np.exp(x) + 3 * x * (y ** 2) * x


def derivative_x(x, y):
    return 6 * x * y ** 2 + 2 * x * np.exp(x) * np.cos(x ** 2) + np.exp(x) * np.sin(x ** 2)


def derivative_y(x, y):
    return 6 * y * x ** 2


def derivative_x_x(x, y):
    return - 4 * x ** 2 * np.exp(x) * np.sin(x ** 2) + 4 * x * np.exp(x) * np.cos(x ** 2) + 6 * y ** 2 + np.exp(x) * np.sin(x ** 2) + 2 * np.exp(x) * np.cos(x ** 2)


def derivative_x_y(x, y):
    return 12 * x * y


def derivative_y_y(x):
    return 6 * x ** 2

# def func(x, y):
#     with warnings.catch_warnings():
#         warnings.simplefilter('ignore')
#         return (x ** 2) + (y ** 2)
#
#
# def derivative_x(x, y):
#     return 2 * x
#
#
# def derivative_y(x, y):
#     return 2 * y
#
#
# def derivative_x_x(x, y):
#     return 2
#
#
# def derivative_x_y(x, y):
#     return 0
#
#
# def derivative_y_y(x):
#     return 2

# def func(x, y):
#     return x ** 3 + 2 * y ** 2 - 3 * x - 4 * y
#
# def derivative_x(x, y):
#     return 3 * x**2 - 3
#
#
# def derivative_y(x, y):
#     return 4 * y - 4

def compute_gradient(x, y):
    return derivative_x(x, y), derivative_y(x, y)


def gradient_descent(x0, y0, lr=1e-2, epsilon=1e-2, arr_shape=1000):
    count = 0
    x, y = x0, y0
    z_list = []
    z_list.append(np.linalg.norm(compute_gradient(x, y)))
    while np.linalg.norm(compute_gradient(x, y)) > epsilon and count < arr_shape:
        dx, dy = compute_gradient(x, y)
        x -= lr * dx
        y -= lr * dy
        z_list.append(np.linalg.norm(compute_gradient(x, y)))
        count += 1
    return x, y, count, z_list


def gradient_descent_momentum(x0, y0, rh0=0.9, lr=1e-2, epsilon=1e-2, arr_shape=1000):  # rh0 -> 1, подбирается для каждой функции свой
    count = 0
    x, y = x0, y0
    vx, vy = 0, 0
    z_list = []
    z_list.append(np.linalg.norm(compute_gradient(x, y)))
    while np.linalg.norm(compute_gradient(x, y)) > epsilon and count < arr_shape:
        dx, dy = compute_gradient(x, y)
        vx = rh0 * vx + dx
        vy = rh0 * vy + dy
        x -= lr * vx
        y -= lr * vy
        z_list.append(np.linalg.norm(compute_gradient(x, y)))
        count += 1
    return x, y, count, z_list


def ada_grad(x0, y0, lr=1e-1, epsilon=1e-3, arr_shape=1000):
    count = 0
    x, y = x0, y0
    z_list = []
    z_list.append(np.linalg.norm(compute_gradient(x, y)))
    grad_squared_x, grad_squared_y = 0, 0
    while np.linalg.norm(compute_gradient(x, y)) > epsilon and count < arr_shape:
        dx, dy = compute_gradient(x, y)
        grad_squared_x += dx * dx
        grad_squared_y += dy * dy
        x -= lr * dx / np.sqrt( grad_squared_x + 1e-7)  #выравнивается по направлению, уменьшая разницу между шагами
        y -= lr * dy / np.sqrt(grad_squared_y + 1e-7)
        z_list.append(np.linalg.norm(compute_gradient(x, y)))
        count += 1
    return x, y, count, z_list


def rmsp_rop(x0, y0, decay_rate=0.95, lr=1e-2, epsilon=1e-3, arr_shape=1000):  # decay_rate -> 1
    count = 0
    x, y = x0, y0
    z_list = []
    z_list.append(np.linalg.norm(compute_gradient(x, y)))
    grad_squared_x, grad_squared_y = 0, 0
    while np.linalg.norm(compute_gradient(x, y)) > epsilon and count < arr_shape:
        dx, dy = compute_gradient(x, y)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            grad_squared_x = (decay_rate * grad_squared_x + (1 - decay_rate) * dx * dx)
            grad_squared_y = (decay_rate * grad_squared_y + (1 - decay_rate) * dy * dy)
        x -= lr * dx / np.sqrt(grad_squared_x + 1e-7)
        y -= lr * dy / np.sqrt(grad_squared_y + 1e-7)
        z_list.append(np.linalg.norm(compute_gradient(x, y)))
        count += 1
    return x, y, count, z_list


def adam(x0, y0, lr=1e-3, beta1=0.95, beta2=0.95, beta=0.999, epsilon=1e-3, arr_shape=1000):
    first_momentim_x, second_momentim_x = 0, 0
    first_momentim_y, second_momentim_y = 0, 0
    count = 0
    x, y = x0, y0
    z_list = []
    z_list.append(np.linalg.norm(compute_gradient(x, y)))
    t = 1
    while np.linalg.norm(compute_gradient(x, y)) > epsilon and count < arr_shape:
        dx, dy = compute_gradient(x, y)
        first_momentim_x = beta1 * first_momentim_x + (1 - beta1) * dx
        second_momentim_x = beta2 * second_momentim_x + (1 - beta2) * dx * dx
        first_unbias_x = first_momentim_x / (1 - beta ** t)
        second_unbias_x = second_momentim_x / (1 - beta ** t)

        first_momentim_y = beta1 * first_momentim_y + (1 - beta1) * dy
        second_momentim_y = beta2 * second_momentim_y + (1 - beta2) * dy * dy
        first_unbias_y = first_momentim_y / (1 - beta ** t)
        second_unbias_y = second_momentim_y / (1 - beta ** t)

        x -= lr * first_unbias_x / (np.sqrt(second_unbias_x) + 1e-7)
        y -= lr * first_unbias_y / (np.sqrt(second_unbias_y) + 1e-7)
        y -= lr * dy
        z_list.append(np.linalg.norm(compute_gradient(x, y)))
        t += 1
        count += 1
    return x, y, count, z_list

def is_pos_def(x):
    if np.all(np.linalg.eigvals(x) > 0):
        return True
    else:
        print("The matrix is not positive definite")
        return False

def newton(x_init, y_init, epsilon=1e-2, arr_shape=100):
    count = 0
    x, y = x_init, y_init

    z_list = []
    z_list.append(np.linalg.norm(compute_gradient(x, y)))
    while np.linalg.norm(compute_gradient(x, y)) > epsilon and count < arr_shape:
        matrix = [[derivative_x_x(x, y), derivative_x_y(x, y)],
                  [derivative_x_y(x, y), derivative_y_y(x)]]

        # if not is_pos_def(matrix):
        #     raise Exception("not positive definite")
        if is_pos_def(matrix):
            res = np.linalg.solve(np.array(matrix), -np.array(compute_gradient(x, y)))
            z_list.append(np.linalg.norm(compute_gradient(x, y)))
            x += res[0]
            y += res[1]
        else:
            dx, dy = compute_gradient(x, y)
            x -= lr * dx
            y -= lr * dy
            z_list.append(np.linalg.norm(compute_gradient(x, y)))

        count += 1
    return x, y, count, z_list


if __name__ == '__main__':
    x = 0.4
    y = 0.2
    lr = 1e-2
    eps = 1e-4

    arr_shape = 100

    rh0 = 0.99

    x1, y1, count1, z_list1 = gradient_descent(x, y)
    print("gradient descent: {}, {}, {}, {}".format(x1, y1, func(x1, y1), count1))


    x2, y2, count2, z_list2 = gradient_descent_momentum(x, y)
    print("gradient descent momentum: {}, {}, {}, {}".format(x2, y2, func(x2, y2), count2))


    x3, y3, count3, z_list3 = ada_grad(x, y)
    print("ada_grad: {}, {}, {}, {}".format(x3, y3, func(x3, y3), count3))


    x4, y4, count4, z_list4 = rmsp_rop(x, y)
    print("rmsp_rop: {}, {}, {}, {}".format(x4, y4, func(x4, y4), count4))


    x5, y5, count5, z_list5 = adam(x, y)
    print("adam: {}, {}, {}, {}".format(x5, y5, func(x5, y5), count5))

    x6, y6, count6, z_list6 = newton(x, y)
    print("newton: {}, {}, {}, {}".format(x6, y6, func(x6, y6), count6))

    # x = np.arange(-1, 1, 1e-1)
    # y = np.arange(-1, 1, 1e-1)
    # z = [[func(xi, yi) for xi in x] for yi in y]
    # z = np.array(z).reshape(len(x), len(y))
    # plt.plot(z_list6[0], z_list6[1], c='r')
    # plt.contour(x, y, z, 10)
    # plt.scatter(z_list6[0], z_list6[1], c='black')

    plt.plot([i for i in range(len(z_list1))], z_list1, color='green', label='gradient descent')
    plt.plot([i for i in range(len(z_list2))], z_list2, color='orange', label='gradient descent momentum)')
    plt.plot([i for i in range(len(z_list3))], z_list3, linewidth=3, color='blue', label='adagrad')
    plt.plot([i for i in range(len(z_list4))], z_list4, linewidth=3, color='red', label='rmsprop')
    plt.plot([i for i in range(len(z_list5))], z_list5, color='yellow', label='adam')
    plt.plot([i for i in range(len(z_list6))], z_list6, color='black', label='newton')
    # plt.yscale('log')
    plt.xlabel("iterations")
    plt.ylabel('value')
    plt.legend()
    plt.show()
