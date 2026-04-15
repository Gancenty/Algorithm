import math
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
            coefficient1 = ((u - knots[i]) / (knots[i + k] - knots[i])) * bspline_basis(
                i, k - 1, u, knots
            )
        if knots[i + k + 1] - knots[i + 1] != 0:
            coefficient2 = (
                (knots[i + k + 1] - u) / (knots[i + k + 1] - knots[i + 1])
            ) * bspline_basis(i + 1, k - 1, u, knots)

        return coefficient1 + coefficient2


def bspline_curve(degree, control_points, points_num):
    N = len(control_points)
    K = degree
    point_dim = control_points.shape[1]
    knots = np.concatenate(([0] * K, np.linspace(0, 1, N - K + 1), [1] * K))
    # knots = np.linspace(0, 1, N + K + 1)
    point_curve = []
    for index, u in enumerate(np.linspace(knots[K], knots[N], points_num)):
        if index == points_num - 1:
            break
        point = np.zeros(point_dim)
        for i in range(N):
            point += control_points[i] * bspline_basis(i, K, u, knots)
        point_curve.append(point)
    return np.array(point_curve)


def bspline_usinglib(degree, control_points, points_num):
    n = len(control_points)
    knot_vector = np.concatenate(
        ([0] * degree, np.linspace(0, 1, n - degree + 1), [1] * degree)
    )
    bspline_x = BSpline(knot_vector, control_points[:, 0], degree)
    bspline_y = BSpline(knot_vector, control_points[:, 1], degree)

    t = np.linspace(0, 1, points_num)
    x = bspline_x(t)
    y = bspline_y(t)
    return np.column_stack((x, y))


def get_bspline_derivation(degree, control_points, points_num=1000):
    N = len(control_points)
    K = degree
    point_dim = control_points.shape[1]
    knots = np.concatenate(([0] * K, np.linspace(0, 1, N - K + 1), [1] * K))
    new_cps = []
    for i in range(N - 1):
        new_cps.append(
            (K / (knots[i + K + 1] - knots[i + 1]))
            * (control_points[i + 1] - control_points[i])
        )
    point_curve = []
    for index,u in enumerate(np.linspace(knots[K], knots[N], points_num)):
        if index == points_num - 1:
            break
        point = np.zeros(point_dim)
        for i in range(N - 1):
            point += new_cps[i] * bspline_basis(i + 1, K - 1, u, knots)
        point[1]=point[1]/point[0]
        point[0]=index
        point_curve.append(point)
    return np.array(point_curve)


def bezier_curve(control_points, num_points=1000):
    T = np.linspace(0, 1, num_points, dtype=np.float64)
    n = len(control_points) - 1
    point_dim = control_points.shape[1]
    curve = np.zeros((num_points, point_dim), dtype=np.float64)
    for num, t in enumerate(T):
        point = control_points.copy().astype(np.float64)
        for i in range(n):
            for j in range(len(point) - 1):
                point[j] = (1 - t) * point[j] + t * point[j + 1]
            point = np.delete(point, len(point) - 1, axis=0)
        curve[num, :] = point
    return curve


def bezier_curve_fast(control_points, num_points=1000):
    nums_cp = len(control_points)
    point_dim = control_points.shape[1]
    get_cmn = lambda n, m: math.factorial(m) / (
        math.factorial(m - n) * math.factorial(n)
    )
    curve = np.zeros((num_points, point_dim), dtype=np.float64)
    for index, t in enumerate(np.linspace(0, 1, num_points)):
        point = np.zeros(point_dim)
        for i in range(nums_cp):
            point += (
                get_cmn(i, nums_cp - 1)
                * (1 - t) ** (nums_cp - 1 - i)
                * t**i
                * control_points[i]
            )
        curve[index, :] = point
    return curve


def get_bezier_curve_derivation1(control_points, num_points=1000):
    degree = len(control_points) - 1
    new_cps = []
    for i in range(len(control_points) - 1):
        new_cps.append((control_points[i + 1] - control_points[i]) * degree)
    nums_cp = len(new_cps)
    point_dim = control_points.shape[1]
    get_cmn = lambda n, m: math.factorial(m) / (
        math.factorial(m - n) * math.factorial(n)
    )
    curve = np.zeros((num_points, point_dim), dtype=np.float64)
    for index, t in enumerate(np.linspace(0, 1, num_points)):
        point = np.zeros(point_dim)
        for i in range(nums_cp):
            point += (
                get_cmn(i, nums_cp - 1)
                * ((1 - t) ** (nums_cp - 1 - i))
                * (t**i)
                * new_cps[i]
            )
        curve[index, :] = np.array([index, point[1] / point[0]])
    return curve


def get_bezier_curve_derivation2(control_points, num_points=1000):
    degree = len(control_points) - 1
    get_cmn = lambda n, m: math.factorial(m) / (
        math.factorial(m - n) * math.factorial(n)
    )
    point_dim = control_points.shape[1]
    curve = np.zeros((num_points, point_dim), dtype=np.float64)
    curve1 = bezier_curve_fast(control_points=control_points[:-1])
    curve2 = bezier_curve_fast(control_points=control_points[1:])
    curve = (curve2 - curve1) * degree
    for index in range(num_points):
        curve[index, :] = np.array([index, curve[index, 1] / curve[index, 0]])
    return curve


def get_bspline_basis(degree, control_points, u):
    N = len(control_points)
    K = degree
    knots = np.concatenate(([0] * K, np.linspace(0, 1, N - K + 1), [1] * K))
    point_basis = []
    point = np.zeros(2)
    for i in range(N):
        point_basis.append(bspline_basis(i, K, u, knots))
    return np.array(point_basis)


control_points = np.array([(1, 2), (3, 6), (4, 2), (7, 10), (6, 12), (8, 12)])
degree = 5  # len(control_points) = degree + 1 cause bspline_curve == bezier_curve

beziercurve = bezier_curve_fast(control_points)
beziercurve_derivation = get_bezier_curve_derivation2(control_points)

bsplinecurve = bspline_curve(degree, control_points, 100)
bsplinelibcurve = bspline_usinglib(degree, control_points, 100)
bsplinecurve_derivation = get_bspline_derivation(degree, control_points, 100)
basis = get_bspline_basis(degree, control_points, 0)

fig, axes = plt.subplots(1, 4, figsize=(20, 10))
for i in range(len(axes)):
    if i == 0:
        axes[i].set_title("Bspline Curve")
        axes[i].grid()
        axes[i].plot(bsplinecurve[:, 0], bsplinecurve[:, 1], "o", color="blue")
        axes[i].scatter(
            control_points[:, 0],
            control_points[:, 1],
            color="red",
            label="Control Points",
        )
    if i == 1:
        axes[i].set_title("Bspline Curve Derivation")
        axes[i].grid()
        axes[i].plot(
            bsplinecurve_derivation[:, 0],
            bsplinecurve_derivation[:, 1],
            "o",
            color="blue",
        )
    if i == 2:
        axes[i].set_title("Bezier Curve")
        axes[i].grid()
        axes[i].plot(beziercurve[:, 0], beziercurve[:, 1], "o", color="blue")
        axes[i].plot(
            control_points[:, 0],
            control_points[:, 1],
            color="red",
            label="Control Points",
        )
    if i == 3:
        axes[i].set_title("Bezier Curve Derivation")
        axes[i].grid()
        axes[i].plot(
            beziercurve_derivation[:, 0],
            beziercurve_derivation[:, 1],
            "o",
            color="blue",
        )

plt.show()
