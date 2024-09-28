import numpy as np
import matplotlib.pyplot as plt

def bspline_basis(i, k, t, knots):
    """
    计算B样条基函数的值
    :param i: 节点索引
    :param k: 基函数的阶数
    :param t: 当前参数值
    :param knots: 节点向量
    :return: 基函数值
    """
    if k == 0:
        # 0阶基函数
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    else:
        # 递归计算k阶基函数
        coef1 = 0.0
        coef2 = 0.0
        
        # 避免除以零的情况
        if (knots[i + k] - knots[i]) != 0:
            coef1 = (t - knots[i]) / (knots[i + k] - knots[i]) * bspline_basis(i, k - 1, t, knots)
        if (knots[i + k + 1] - knots[i + 1]) != 0:
            coef2 = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) * bspline_basis(i + 1, k - 1, t, knots)
        
        return coef1 + coef2

def bspline_curve(control_points, degree, num_points=100):
    """
    生成B样条曲线
    :param control_points: 控制点数组
    :param degree: B样条的阶数
    :param num_points: 曲线上的采样点数量
    :return: B样条曲线点
    """
    n = len(control_points)
    knots = np.concatenate(([0] * degree, np.linspace(0, 1, n - degree + 1), [1] * degree))  # 生成节点向量
    curve_points = []

    for t in np.linspace(0, 1, num_points):
        point = np.zeros(2)  # 初始化曲线点
        for i in range(n):
            b = bspline_basis(i, degree, t, knots)  # 计算基函数值
            point += b * control_points[i]  # 线性组合生成曲线点
        curve_points.append(point)

    return np.array(curve_points)

# 控制点
control_points = np.array([
    [0, 0],
    [1, 2],
    [2, -1],
    [4, 3],
    [5, 0]
])

# 生成B样条曲线
degree = 3  # B样条的阶数
curve_points = bspline_curve(control_points, degree)

# 绘制B样条曲线和控制点
plt.plot(curve_points[:, 0], curve_points[:, 1], label='B-Spline Curve', color='blue')
plt.scatter(control_points[:, 0], control_points[:, 1], color='red', label='Control Points')
plt.plot(control_points[:, 0], control_points[:, 1], '--', color='gray', label='Control Polygon')
plt.title('B-Spline Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()
