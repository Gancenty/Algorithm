import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# k->次数 u->knot取值 i->控制点索引


def bspline_basis(i, k, u, knots):
    if k == 0:
        return 1 if knots[i] <= u and u < knots[i + 1] else 0
    else:
        coefficient1 = 0.0
        coefficient2 = 0.0

        if knots[i + k] - knots[i] != 0:
            coefficient1 = (
                (u - knots[i]) / (knots[i + k] - knots[i])
            ) * bspline_basis(i, k - 1, u, knots)
        if knots[i + k + 1] - knots[i + 1] != 0:
            coefficient2 = (
                (knots[i + k + 1] - u) / (knots[i + k + 1] - knots[i + 1])
            ) * bspline_basis(i + 1, k - 1, u, knots)

        return coefficient1 + coefficient2


def bspline_curve(degree, control_points, points_num):
    N = len(control_points)
    K = degree
    knots = np.concatenate(([0] * K, np.linspace(0, 1, N - K + 1), [1] * K))
    point_curve = []
    for u in np.linspace(0, 1, points_num):
        point = np.zeros(2)
        for i in range(len(control_points)):
            point += control_points[i] * bspline_basis(i, K, u, knots)
        point_curve.append(point)
    return np.array(point_curve)


control_points = np.array([(1, 2), (3, 6), (4, 2), (7, 10)])
degree = 3

curve = bspline_curve(degree, control_points, 100)

fig, axes = plt.subplots(1, 1)
axes.grid()
axes.plot(curve[:-1, 0], curve[:-1, 1],'--',color='blue')
axes.scatter(control_points[:, 0], control_points[:, 1], color='red', label='Control Points')
plt.show()
