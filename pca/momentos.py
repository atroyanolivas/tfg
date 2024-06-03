import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open("./pca/sample.png")
# print(image[:5, :5])
# image = image.resize((m := min(image.size), m))
image = image.resize((64, 64))
image = np.array(image) #/ 255
print(image[:5, :5])
print(image.shape)

# Set de interval to aproximate:
interval = (-1, 1)

def x_j(j: int, N: int) -> float:
    return interval[0] + j * ((interval[1] - interval[0]) / (N - 1))


def y_i(i: int, M: int) -> float:
    return interval[1] - i * ((interval[1] - interval[0]) / (M - 1))


def combinations(n, r):
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))


def tchev_pol(n: int):
    return np.polynomial.Chebyshev(np.polynomial.chebyshev.Chebyshev.basis(n).coef)

def momentos_tchebyshev(F, order=10):
    """
    Parameters:
    - F: Matriz de datos
    - N: Orden de la matriz de momentos (máximo el orden de la matriz F, por defecto 10)
    ---------------
    Returns:
    - Matriz de momentos de orden N
    """
    N, M = F.shape
    tch_pol = [tchev_pol(i) for i in range(0, order)]
    def calc_momentos_tchebyshev(n: int, m: int, F):
        """
        Parameters:

        - m, n : orden del momento
        - F: matriz de datos de la imagen
        """
        return (tch_pol[m](y_vals) @ (F @ tch_pol[n](x_vals)))  # Matrix multiplication operator, @

    res = np.zeros((order, order))
    x_vals = np.array([x_j(j, M) for j in range(0, M)])
    y_vals = np.array([y_i(i, N) for i in range(0, N)]).reshape((1, -1))

    for n in range(0, order):
        for m in range(0, order):
            calc = calc_momentos_tchebyshev(n, m, F)
            res[n, m] = (n / 2) * calc
    return res

def reconstruccion(M, B: list, original_shape): 
    """
    Parameters:

    - M: Matriz de momentos
    - B: Base respecto de la que reconstruir
    -----------------------
    Returns:

    - Matriz imagen aproximada
    """
    order = min(M.shape)

    def calc_coef_matrix(p_x, p_y):
        return p_x @ (M @ p_y)
    
    res = np.zeros(original_shape)
    for i in range(0, original_shape[0]): # Should be N = 68
        for j in range(0, original_shape[1]): # Should be M = 60
            xj = x_j(j, original_shape[1])
            yi = y_i(i, original_shape[0])

            px = np.array([B[k](xj) for k in range(0, order)]).reshape((1, -1))
            py = np.array([B[k](yi) for k in range(0, order)])
            res[i, j] = calc_coef_matrix(px, py)

    return res

n_momentos = 120
tche = momentos_tchebyshev(image, n_momentos)

print("finished")
reconstr = reconstruccion(tche, [tchev_pol(i) for i in range(0, n_momentos)], image.shape)

print(reconstr[:5, :5])

# Plot the original and reconstructed images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstr)
plt.title('Reconstructed Image with Tche')
plt.axis('off')

plt.show()
