import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from typing import Tuple, Optional, Union

plt.style.use(
    'seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')


class FourierSeries:
    def __init__(self, periodo: float = 2*np.pi):
        self.T = periodo
        self.omega0 = 2 * np.pi / periodo

    def calcular_coeficientes_trigonometricos(self,
                                              func,
                                              n_harmonicos: int = 10,
                                              metodo: str = 'cuadratura') -> Tuple[float, np.ndarray, np.ndarray]:
        if metodo == 'cuadratura':
            return self._coeficientes_numericos(func, n_harmonicos)
        else:
            raise NotImplementedError(
                "Método analítico debe implementarse por señal específica")

    def coeficientes_exponenciales(self,
                                   func,
                                   n_harmonicos: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        N = 10000
        t = np.linspace(-self.T/2, self.T/2, N)
        dt = self.T / N
        y = func(t)

        n_vector = np.arange(-n_harmonicos, n_harmonicos + 1)
        cn = np.zeros(len(n_vector), dtype=complex)

        for i, n in enumerate(n_vector):
            if n == 0:
                cn[i] = (1/self.T) * np.trapezoid(y, dx=dt)
            else:
                integrand = y * np.exp(-1j * n * self.omega0 * t)
                cn[i] = (1/self.T) * np.trapezoid(integrand, dx=dt)

        return n_vector, cn

    def _coeficientes_numericos(self, func, n_harmonicos: int) -> Tuple[float, np.ndarray, np.ndarray]:
        N = 10000
        t = np.linspace(-self.T/2, self.T/2, N)
        dt = self.T / N

        y = func(t)

        a0 = (2/self.T) * np.trapezoid(y, dx=dt)

        an = np.zeros(n_harmonicos)
        bn = np.zeros(n_harmonicos)

        for n in range(1, n_harmonicos + 1):
            integrand_cos = y * np.cos(n * self.omega0 * t)
            an[n-1] = (2/self.T) * np.trapezoid(integrand_cos, dx=dt)

            integrand_sin = y * np.sin(n * self.omega0 * t)
            bn[n-1] = (2/self.T) * np.trapezoid(integrand_sin, dx=dt)

        return a0, an, bn

    def sintetizar_serie(self,
                         a0: float,
                         an: np.ndarray,
                         bn: np.ndarray,
                         t: np.ndarray,
                         n_terminos: Optional[int] = None) -> np.ndarray:

        if n_terminos is None:
            n_terminos = len(an)
        else:
            n_terminos = min(n_terminos, len(an))

        y = np.ones_like(t) * a0/2

        for n in range(1, n_terminos + 1):
            y += an[n-1] * np.cos(n * self.omega0 * t)
            y += bn[n-1] * np.sin(n * self.omega0 * t)

        return y


def onda_cuadrada(t: np.ndarray, amplitud: float = 1, duty_cycle: float = 0.5) -> np.ndarray:
    periodo = 2 * np.pi if np.max(t) > 10 else 2
    return amplitud * signal.square(2 * np.pi * t / periodo, duty=duty_cycle)


def onda_triangular(t: np.ndarray, amplitud: float = 1) -> np.ndarray:
    periodo = 2 * np.pi if np.max(t) > 10 else 2
    return amplitud * signal.sawtooth(2 * np.pi * t / periodo, width=0.5)


def onda_diente_sierra(t: np.ndarray, amplitud: float = 1) -> np.ndarray:
    periodo = 2 * np.pi if np.max(t) > 10 else 2
    return amplitud * signal.sawtooth(2 * np.pi * t / periodo)


def impulso_periodico(t: np.ndarray, amplitud: float = 1, ancho: float = 0.1) -> np.ndarray:
    periodo = 2 * np.pi if np.max(t) > 10 else 2
    return amplitud * (signal.square(2 * np.pi * t / periodo, duty=ancho) > 0).astype(float)


def calcular_potencia(coeficientes: Union[Tuple, np.ndarray], tipo: str = 'trigonometrico') -> float:
    if tipo == 'trigonometrico':
        a0, an, bn = coeficientes
        potencia = (a0/2)**2 + 0.5 * np.sum(an**2 + bn**2)
    elif tipo == 'exponencial':
        cn = coeficientes
        potencia = np.sum(np.abs(cn)**2)
    else:
        raise ValueError("Tipo debe ser 'trigonometrico' o 'exponencial'")

    return potencia


def convertir_coeficientes(a0: float,
                           an: np.ndarray,
                           bn: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = len(an)
    n_vector = np.arange(-N, N + 1)
    cn = np.zeros(len(n_vector), dtype=complex)

    cn[N] = a0

    for n in range(1, N + 1):
        cn[N + n] = (an[n-1] - 1j * bn[n-1]) / 2  # n > 0
        cn[N - n] = (an[n-1] + 1j * bn[n-1]) / 2  # n < 0

    return n_vector, cn
