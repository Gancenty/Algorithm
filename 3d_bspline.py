import numpy as np
import matplotlib.pyplot as plt

def bspline_basis(i, k, u, knots):
    if k == 0:
        return 1 if knots[i] <= u < knots[i + 1] else 0
    else:
        coefficient1 = 0.0
        coefficient2 = 0.0

        if knots[i + k] - knots[i] != 0:
            coefficient1 = ((u - knots[i]) / (knots[i + k] - knots[i])) * bspline_basis(i, k - 1, u, knots)
        if knots[i + k + 1] - knots[i + 1] != 0:
            coefficient2 = ((knots[i + k + 1] - u) / (knots[i + k + 1] - knots[i + 1])) * bspline_basis(i + 1, k - 1, u, knots)

        return coefficient1 + coefficient2

def bspline_curve(degree, control_points, points_num):
    N = len(control_points)
    K = degree
    knots = np.concatenate(([0] * K, np.linspace(0, 1, N - K + 1), [1] * K))
    point_curve = []
    for u in np.linspace(knots[K], knots[N], points_num):
        point = np.zeros(3)
        for i in range(N):
            point += control_points[i] * bspline_basis(i, K, u, knots)
        point_curve.append(point)
    return np.array(point_curve[:-1])

# 定义控制点和参数
control_points = np.array([(0, 0, 0), (10, 60, 1.2), (45, 80, 1.4), (60, 73, 1.6), (80, 60, 1.8), (90, 70, 2.0)])
degree = 3
points_num = 10000

# 计算B样条曲线及其导数曲线
bsplinecurve = bspline_curve(degree, control_points, points_num)
for i in range(len(bsplinecurve)):
    bsplinecurve[i] = bsplinecurve[i] + [9961.8,19353.7,280.201]
point = bsplinecurve.tolist()
with open("./output.txt", 'w+') as f:
    traj = 'traj = ' + str(point) + ';'
    f.write(traj)
# 绘制B样条曲线及其导数曲线
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(bsplinecurve[:, 0], bsplinecurve[:, 1], bsplinecurve[:, 2], label='B-spline curve')

# 绘制控制点
ax.scatter(control_points[:, 0], control_points[:, 1], control_points[:, 2], color='red')

# 设置标签和图例
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()
