import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# 控制点
control_points = np.array([(1, 2), (3, 6), (4, 2), (7, 10)])

# 节点向量（knot vector）
# 根据控制点和曲线阶数自动生成
degree = 3  # B样条的阶数（3表示立方B样条）
n = len(control_points)
knot_vector = np.concatenate(([0] * degree, np.linspace(0, 1, n - degree + 1), [1] * degree))

# 创建B样条对象
bspline_x = BSpline(knot_vector, control_points[:, 0], degree)
bspline_y = BSpline(knot_vector, control_points[:, 1], degree)

# 生成样本点
t = np.linspace(0, 1, 100)
x = bspline_x(t)
y = bspline_y(t)

# 绘制B样条曲线和控制点
plt.plot(x, y, label='B-Spline Curve', color='blue')
plt.scatter(control_points[:, 0], control_points[:, 1], color='red', label='Control Points')
plt.plot(control_points[:, 0], control_points[:, 1], '--', color='gray', label='Control Polygon')
plt.title('B-Spline Curve')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()
