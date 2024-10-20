import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.optimize import fsolve

# Definir la función de la ecuación diferencial
def f(x, y):
    return -y + 10 * np.sin(3 * x)

# Solución analítica de la ecuación diferencial
def analytical_solution(x):
    return 3 * np.exp(-x) - 3 * np.cos(3 * x) + 3 * np.sin(3 * x)

# Parte 2.3: Método de Euler
def euler_method(x0, y0, h, x_end):
    N = int((x_end - x0) / h)  # Número de pasos
    x_values = [x0]
    y_values = [y0]

    for n in range(N):
        xn = x_values[n]
        yn = y_values[n]
        yn_plus_1 = yn + h * f(xn, yn)  # Calcular el siguiente valor de y
        x_values.append(xn + h)  # Incrementar x por el paso
        y_values.append(yn_plus_1)  # Almacenar el nuevo valor de y
    
    return x_values, y_values

# Parte 2.4: Polinomio interpolante
def plot_interpolating_polynomial(x_values, y_values):
    polynomial = lagrange(x_values, y_values)  # Obtener el polinomio interpolante
    x_interp = np.linspace(min(x_values), max(x_values), 100)
    y_interp = polynomial(x_interp)

    # Graficar el polinomio interpolante
    plt.plot(x_interp, y_interp, label='Polinomio Interpolante', color='orange')
    plt.scatter(x_values, y_values, color='blue', label='Puntos Aproximados')
    plt.title('Polinomio Interpolante')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Eje x
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # Eje y
    plt.grid()
    plt.legend()
    plt.show()
    
    return polynomial

# Parte 2.5: Encontrar raíces del polinomio interpolante
def find_roots_of_polynomial(polynomial):
    # Definir la función que representa el polinomio
    poly_func = lambda x: polynomial(x)

    # Encontrar raíces
    roots = fsolve(poly_func, x0=[0.1, 1.2])  # Puntos iniciales para la búsqueda de raíces
    positive_roots = [root for root in roots if root > 0]

    return positive_roots

# Parámetros
x0 = 0           # Inicio del intervalo
y0 = 0           # Valor inicial
h = 0.1          # Paso
x_end = 2        # Fin del intervalo

# Ejecutar el método de Euler
x_values, y_values = euler_method(x0, y0, h, x_end)

# Calcular la solución analítica
y_analytical = analytical_solution(np.array(x_values))

# Parte 2.2: Graficar la solución aproximada y la solución analítica
plt.plot(x_values, y_values, label='Aproximación de Euler', color='blue')
plt.plot(x_values, y_analytical, label='Solución Analítica', color='red', linestyle='--')
plt.title('Aproximación de la solución usando el método de Euler')
plt.xlabel('x')
plt.ylabel('y')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')  # Eje x
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # Eje y
plt.grid()
plt.xlim([0, 2])  # Limitar el eje x al intervalo [0, 2]
plt.ylim([min(min(y_values), min(y_analytical)) - 1, max(max(y_values), max(y_analytical)) + 1])  # Ajustar límites del eje y
plt.legend()
plt.show()

# Calcular el error
error = np.abs(np.array(y_values) - y_analytical)
percent_error = (error / np.abs(y_analytical)) * 100  # Porcentaje de error

# Imprimir los errores en la consola
for x, y_euler, y_analytic, err in zip(x_values, y_values, y_analytical, percent_error):
    print(f"x: {x:.2f}, Euler: {y_euler:.4f}, Analítico: {y_analytic:.4f}, % Error: {err:.2f}%")

# Graficar el polinomio interpolante
polynomial = plot_interpolating_polynomial(x_values, y_values)


# Mostrar la ecuación del polinomio interpolante
print("Ecuación del polinomio interpolante:")
coefficients = polynomial.coef
degree = len(coefficients) - 1
equation_terms = [f"{coefficients[i]:.4f} * x^{degree - i}" for i in range(degree + 1)]
equation = " + ".join(equation_terms).replace("x^1 ", "x ").replace("x^0", "").replace("x^0 ", "").replace("+ -", "- ")
print(equation)

# Encontrar las raíces positivas del polinomio interpolante
positive_roots = find_roots_of_polynomial(polynomial)
print("Raíces positivas del polinomio interpolante:", positive_roots)

# Mostrar las raíces conocidas de la solución de la ecuación
known_roots = [0, 1.36926249]  # Raíces que ya conoces
print("Raíces conocidas de la solución:", known_roots)

