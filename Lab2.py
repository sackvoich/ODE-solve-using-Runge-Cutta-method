import numpy as np
import matplotlib.pyplot as plt

def runge_kutta(f, x0, y0, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = np.zeros(n+1)
    y[0] = y0

    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x, y

def f(x, y):
    return (1 + x) * np.exp(-y)

# Аналитическое решение
def analytical_solution(x, x0, y0):
    return np.log((x**2 + 2*x)/2) - (15 - 24)/25

# Начальные условия
x0 = 0.4
y0 = 1
a = 0.4
b = 1.9
n = 15

# Решение методом Рунге-Кутты
x_rk, y_rk = runge_kutta(f, x0, y0, a, b, n)

# Аналитическое решение
x_analytic = np.linspace(a, b, n+1)
y_analytic = analytical_solution(x_analytic, x0, y0)

# Вывод таблицы с решением
print("Сравнение численного и аналитического решений уравнения y' = (1+x)*e**(-y):")
print("┌───────┬───────────┬───────────┬───────────┬")
print("│   x   │   y_RK    │   y_Anal  │ Difference│")
print("├───────┼───────────┼───────────┼───────────┤")
for i in range(n+1):
    print("│{:7.3f}│{:10.5f} │{:10.5f} │ {:9.5f} │".format(x_rk[i], y_rk[i], y_analytic[i], y_rk[i] - y_analytic[i]))
print("└───────┴───────────┴───────────┴───────────┘")

# Построение графиков
plt.figure(figsize=(8, 6))

# График численного решения
plt.plot(x_rk, y_rk, label='Метод Рунге-Кутты', marker='o')

# График аналитического решения
plt.plot(x_analytic, y_analytic, label='Аналитическое решение', linestyle='--')

# Настройки графика
plt.xlabel('x')
plt.ylabel('y')
plt.title('Сравнение численного и аналитического решений уравнения')
plt.grid(True)
plt.legend()

# Показать график
plt.show()