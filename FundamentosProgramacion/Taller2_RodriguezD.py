###################################################
### Taller 1 Curso: Fundamentos de Programación ###
###################################################

# Docente: Daniel Escobar
# Estudiante: Danilo Rodriguez Arango 
# Tema: Construyendo una librería de álgebra líneal


"""
Módulo principal de la librería de álgebra lineal
=================================================

Este módulo contiene las implementaciones de las clases Vector y Matrix,
así como las funciones de álgebra lineal asociadas.
"""

# from __future__ import annotations

import math
from typing import List, Union, Tuple, Optional

Number = Union[int, float]
_EPS = 1e-12  # Tolerancia numérica


class Vector:
    """
    Clase para representar y manipular vectores.
    
    Un vector es una lista de números que puede representar
    puntos en el espacio, direcciones, o cualquier secuencia ordenada de valores.
    """
    
    def __init__(self, components: List[Union[int, float]]):
        """
        Inicializa un vector con sus componentes.
        
        Args:
            components: Lista de números que representan las componentes del vector
        """
      # Implementación
        if not isinstance(components, (list, tuple)):
            raise TypeError("components debe ser una lista o tupla de números")
        comps: List[Number] = list(components)
        for x in comps:
            if not isinstance(x, (int, float)):
                raise TypeError("Todos los componentes deben ser numéricos (int o float)")
        self._components: List[Number] = comps[:]
    
    def __str__(self) -> str:
        """Representación en string del vector."""
        # Implementación
        return f"<{', '.join(str(x) for x in self._components)}>"
    
    def __repr__(self) -> str:
        """Representación detallada del vector."""
        # Implementación
        return f"Vector({self._components!r})"
    
    def __len__(self) -> int:
        """Retorna la dimensión del vector."""
        # Implementación
        return len(self._components)
    
    def __getitem__(self, index: int) -> Union[int, float]:
        """Permite acceder a los componentes del vector usando índices."""
        # Implementación
        return self._components[index]
    
    def __setitem__(self, index: int, value: Union[int, float]):
        """Permite modificar componentes del vector usando índices."""
        # Implementación
        if not isinstance(value, (int, float)):
            raise TypeError("El valor asignado debe ser numérico (int o float)")
        self._components[index] = value

    # Verificar tipo y dimensión
    def _check_same_dim(self, other: "Vector"):
        if not isinstance(other, Vector):
            raise TypeError("La operación requiere otro Vector")
        if len(self) != len(other):
            raise ValueError("Los vectores deben tener la misma dimensión")    

    def __add__(self, other: 'Vector') -> 'Vector':
        """Suma de vectores usando el operador +."""
        # Implementación
        self._check_same_dim(other)
        return Vector([a + b for a, b in zip(self._components, other._components)])
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        """Resta de vectores usando el operador -."""
        # Implementación
        self._check_same_dim(other)
        return Vector([a - b for a, b in zip(self._components, other._components)])
    
    def __mul__(self, scalar: Union[int, float]) -> 'Vector':
        """Multiplicación por escalar usando el operador *."""
        # Implementación
        if isinstance(scalar, (int, float)):
            return Vector([scalar * x for x in self._components])
        return NotImplemented
    
    def __rmul__(self, scalar: Union[int, float]) -> 'Vector':
        """Multiplicación por escalar (orden invertido)."""
        # Implementación
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar: Union[int, float]) -> 'Vector':
        """División por escalar usando el operador /."""
        # Implementación
        if not isinstance(scalar, (int, float)):
            raise TypeError("La división de un vector solo es por un escalar (int o float)")
        if abs(scalar) < _EPS:
            raise ZeroDivisionError("No se puede dividir por cero")
        return Vector([x / scalar for x in self._components])
    
    def __eq__(self, other: 'Vector') -> bool:
        """Igualdad entre vectores usando el operador ==."""
        # Implementación
        if not isinstance(other, Vector) or len(self) != len(other):
            return False
        return all(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(self._components, other._components))
    
    def __ne__(self, other: 'Vector') -> bool:
        """Desigualdad entre vectores usando el operador !=."""
        # Implementación
        return not (self == other)
    
    @property
    def magnitude(self) -> float:
        """Calcula y retorna la magnitud (norma) del vector."""
        # Implementación
        return math.sqrt(sum(x * x for x in self._components))
    
    @property
    def unit_vector(self) -> 'Vector':
        """Retorna el vector unitario (normalizado)."""
        # Implementación
        mag = self.magnitude
        if mag < _EPS:
            raise ValueError("No se puede normalizar el vector cero")
        return self / mag
    
    def dot(self, other: 'Vector') -> float:
        """
        Calcula el producto punto con otro vector.
        
        Args:
            other: Otro vector para el producto punto
            
        Returns:
            El producto punto como un número
        """
        # Implementación
        self._check_same_dim(other)
        return sum(a * b for a, b in zip(self._components, other._components))
    
    def cross(self, other: 'Vector') -> 'Vector':
        """
        Calcula el producto cruz con otro vector (solo para vectores 3D).
        
        Args:
            other: Otro vector para el producto cruz
            
        Returns:
            Un nuevo vector resultado del producto cruz
        """
        # Implementación
        if len(self) != 3 or len(other) != 3:
            raise ValueError("El producto cruz solo está definido para vectores de dimensión 3")
        a1, a2, a3 = self._components
        b1, b2, b3 = other._components
        return Vector([a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1])
    
    def angle_with(self, other: 'Vector') -> float:
        """
        Calcula el ángulo entre este vector y otro.
        
        Args:
            other: Otro vector
            
        Returns:
            El ángulo en radianes
        """
        # Implementación
        self._check_same_dim(other)
        mag_prod = self.magnitude * other.magnitude
        if mag_prod < _EPS:
            raise ValueError("No se puede calcular el ángulo con un vector de magnitud cero")
        cos_theta = self.dot(other) / mag_prod
        cos_theta = max(-1.0, min(1.0, cos_theta))
        return math.acos(cos_theta)


class Matrix:
    """
    Clase para representar y manipular matrices.
    
    Una matriz es una colección rectangular de números organizados en filas y columnas.
    """
    
    def __init__(self, data: List[List[Union[int, float]]]):
        """
        Inicializa una matriz con sus datos.
        
        Args:
            data: Lista de listas que representa las filas de la matriz
        """
         # Implementación
        if not isinstance(data, (list, tuple)) or len(data) == 0:
            raise TypeError("data debe ser una lista de listas no vacía")
        rows = [list(row) for row in data]
        n_cols = len(rows[0])
        if n_cols == 0:
            raise ValueError("La matriz debe tener al menos una columna")
        for r in rows:
            if len(r) != n_cols:
                raise ValueError("Todas las filas deben tener la misma longitud")
            for x in r:
                if not isinstance(x, (int, float)):
                    raise TypeError("Todos los elementos de la matriz deben ser numéricos")
        self._data: List[List[Number]] = [row[:] for row in rows] 

    
    def __str__(self) -> str:
        """Representación en string de la matriz."""
        # Implementación
        lines = ["[" + " ".join(f"{x}" for x in row) + "]" for row in self._data]
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Representación detallada de la matriz."""
        # Implementación
        return f"Matrix({self._data!r})"
    
    def __getitem__(self, key: Union[int, Tuple[int, int]]) -> Union[List[Union[int, float]], Union[int, float]]:
        """Permite acceder a filas o elementos específicos de la matriz."""
        # Implementación
        if isinstance(key, tuple):
            i, j = key
            return self._data[i][j]
        return self._data[key][:]
    
    def __setitem__(self, key: Union[int, Tuple[int, int]], value: Union[List[Union[int, float]], Union[int, float]]):
        """Permite modificar filas o elementos específicos de la matriz."""
        # Implementación
        if isinstance(key, tuple):
            i, j = key
            if not isinstance(value, (int, float)):
                raise TypeError("El valor del elemento debe ser numérico")
            self._data[i][j] = value
        else:
            if not isinstance(value, (list, tuple)) or len(value) != self.num_columns:
                raise ValueError("Debe asignar una fila con longitud correcta")
            for x in value:
                if not isinstance(x, (int, float)):
                    raise TypeError("Los elementos de la fila deben ser numéricos")
            self._data[key] = list(value)
    
    # Verificar tipo y dimensión
    def _check_same_shape(self, other: "Matrix"):
        if not isinstance(other, Matrix):
            raise TypeError("La operación requiere otra Matrix")
        if self.shape != other.shape:
            raise ValueError("Las matrices deben tener la misma forma")
        
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """Suma de matrices usando el operador +."""
        # Implementación
        self._check_same_shape(other)
        return Matrix([[a + b for a, b in zip(r1, r2)] for r1, r2 in zip(self._data, other._data)])
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Resta de matrices usando el operador -."""
        # Implementación
        self._check_same_shape(other)
        return Matrix([[a - b for a, b in zip(r1, r2)] for r1, r2 in zip(self._data, other._data)])
    
    def __mul__(self, other: Union['Matrix', 'Vector', int, float]) -> Union['Matrix', 'Vector']:
        """Multiplicación de matrices/vectores/escalares usando el operador *."""
        # Implementación
        if isinstance(other, (int, float)):
            return Matrix([[other * x for x in row] for row in self._data])
        if isinstance(other, Vector):
            if self.num_columns != len(other):
                raise ValueError("Dimensiones incompatibles para Matriz x Vector")
            res = []
            for i in range(self.num_rows):
                s = 0.0
                row = self._data[i]
                for j in range(self.num_columns):
                    s += row[j] * other[j]
                res.append(s)
            return Vector(res)
        if isinstance(other, Matrix):
            if self.num_columns != other.num_rows:
                raise ValueError("Dimensiones incompatibles para multiplicación de matrices")
            result = [[0.0 for _ in range(other.num_columns)] for _ in range(self.num_rows)]
            for i in range(self.num_rows):
                for k in range(self.num_columns):
                    aik = self._data[i][k]
                    if abs(aik) < _EPS:
                        continue
                    for j in range(other.num_columns):
                        result[i][j] += aik * other._data[k][j]
            return Matrix(result)
        return NotImplemented
    
    def __rmul__(self, scalar: Union[int, float]) -> 'Matrix':
        """Multiplicación por escalar (orden invertido)."""
        # Implementación
        if isinstance(scalar, (int, float)):
            return self * scalar
        return NotImplemented
    
    def __eq__(self, other: 'Matrix') -> bool:
        """Igualdad entre matrices usando el operador ==."""
        # Implementación
        if not isinstance(other, Matrix) or self.shape != other.shape:
            return False
        for r1, r2 in zip(self._data, other._data):
            for a, b in zip(r1, r2):
                if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12):
                    return False
        return True
    
    def __ne__(self, other: 'Matrix') -> bool:
        """Desigualdad entre matrices usando el operador !=."""
        # Implementación
        return not (self == other)
    
    @property
    def num_rows(self) -> int:
        """Retorna el número de filas de la matriz."""
        # Implementación
        return len(self._data)
    
    @property
    def num_columns(self) -> int:
        """Retorna el número de columnas de la matriz."""
        # Implementación
        return len(self._data[0]) if self._data else 0
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Retorna las dimensiones de la matriz como (filas, columnas)."""
        # Implementación
        return (self.num_rows, self.num_columns)
    
    @property
    def T(self) -> 'Matrix':
        """Retorna la transpuesta de la matriz."""
        # Implementación
        return Matrix([[self._data[i][j] for i in range(self.num_rows)] for j in range(self.num_columns)])
    
    @property
    def trace(self) -> Union[int, float]:
        """Calcula y retorna la traza de la matriz (suma de elementos diagonales)."""
        # Implementación
        if not self.is_square():
          raise ValueError("La traza solo está definida para matrices cuadradas")
        return sum(self._data[i][i] for i in range(self.num_rows))

    
    @property
    def determinant(self) -> Union[int, float]:
        """Calcula y retorna el determinante de la matriz."""
        # Implementación
        return _determinant_impl(self)
    
    @property
    def inverse(self) -> 'Matrix':
        """Calcula y retorna la matriz inversa."""
        # Implementación
        return _inverse_impl(self)
    
    def is_square(self) -> bool:
        """Verifica si la matriz es cuadrada."""
        # Implementación
        return self.num_rows == self.num_columns
    
    def is_symmetric(self) -> bool:
        """Verifica si la matriz es simétrica."""
        # Implementación
        if not self.is_square():
            return False
        return self == self.T
    
    def is_diagonal(self) -> bool:
        """Verifica si la matriz es diagonal."""
        # Implementación
        if not self.is_square():
            return False
        for i in range(self.num_rows):
            for j in range(self.num_columns):
                if i != j and abs(self._data[i][j]) > _EPS:
                    return False
        return True
    
    def get_row(self, index: int) -> 'Vector':
        """
        Obtiene una fila específica como vector.
        
        Args:
            index: Índice de la fila
            
        Returns:
            Vector con los elementos de la fila
        """
        # Implementación
        return Vector(self._data[index][:])
    
    def get_column(self, index: int) -> 'Vector':
        """
        Obtiene una columna específica como vector.
        
        Args:
            index: Índice de la columna
            
        Returns:
            Vector con los elementos de la columna
        """
        # Implementación
        return Vector([self._data[i][index] for i in range(self.num_rows)])


# =============================================================================
# FUNCIONES DE VECTOR
# =============================================================================

def dot_product(v1: Vector, v2: Vector) -> float:
    """
    Calcula el producto punto entre dos vectores.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        
    Returns:
        El producto punto como un número
    """
    # Implementación
    return v1.dot(v2)


def magnitude(v: Vector) -> float:
    """
    Calcula la magnitud (norma) de un vector.
    
    Args:
        v: El vector
        
    Returns:
        La magnitud del vector
    """
    # Implementación
    return v.magnitude


def normalize(v: Vector) -> Vector:
    """
    Normaliza un vector (lo convierte en vector unitario).
    
    Args:
        v: El vector a normalizar
        
    Returns:
        Un nuevo vector normalizado
    """
    # Implementación
    return v.unit_vector


def cross_product(v1: Vector, v2: Vector) -> Vector:
    """
    Calcula el producto cruz entre dos vectores 3D.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        
    Returns:
        Un nuevo vector resultado del producto cruz
    """
    # Implementación
    return v1.cross(v2)


def angle_between(v1: Vector, v2: Vector) -> float:
    """
    Calcula el ángulo entre dos vectores.
    
    Args:
        v1: Primer vector
        v2: Segundo vector
        
    Returns:
        El ángulo en radianes
    """
    # Implementación
    return v1.angle_with(v2)


# =============================================================================
# FUNCIONES DE MATRIZ
# =============================================================================

def scale(matrix: Matrix, scalar: Union[int, float]) -> Matrix:
    """
    Multiplica una matriz por un escalar.
    
    Args:
        matrix: La matriz
        scalar: El escalar
        
    Returns:
        Una nueva matriz escalada
    """
    # Implementación
    if not isinstance(scalar, (int, float)):
        raise TypeError("El escalar debe ser numérico (int o float)")
    return matrix * scalar


def add(m1: Matrix, m2: Matrix) -> Matrix:
    """
    Suma dos matrices.
    
    Args:
        m1: Primera matriz
        m2: Segunda matriz
        
    Returns:
        Una nueva matriz resultado de la suma
    """
    # Implementación
    return m1 + m2


def subtract(m1: Matrix, m2: Matrix) -> Matrix:
    """
    Resta dos matrices.
    
    Args:
        m1: Primera matriz
        m2: Segunda matriz
        
    Returns:
        Una nueva matriz resultado de la resta
    """
    # Implementación
    return m1 - m2


def vector_multiply(matrix: Matrix, vector: Vector) -> Vector:
    """
    Multiplica una matriz por un vector.
    
    Args:
        matrix: La matriz
        vector: El vector
        
    Returns:
        Un nuevo vector resultado de la multiplicación
    """
    # Implementación
    return matrix * vector


def matrix_multiply(m1: Matrix, m2: Matrix) -> Matrix:
    """
    Multiplica dos matrices.
    
    Args:
        m1: Primera matriz
        m2: Segunda matriz
        
    Returns:
        Una nueva matriz resultado de la multiplicación
    """
    # Implementación
    return m1 * m2


def transpose(matrix: Matrix) -> Matrix:
    """
    Calcula la transpuesta de una matriz.
    
    Args:
        matrix: La matriz
        
    Returns:
        Una nueva matriz transpuesta
    """
    # Implementación
    return matrix.T


def determinant(matrix: Matrix) -> Union[int, float]:
    """
    Calcula el determinante de una matriz cuadrada.
    
    Args:
        matrix: La matriz cuadrada
        
    Returns:
        El determinante
    """
    # Implementación
    return matrix.determinant


def inverse(matrix: Matrix) -> Matrix:
    """
    Calcula la matriz inversa.
    
    Args:
        matrix: La matriz cuadrada invertible
        
    Returns:
        Una nueva matriz inversa
    """
    # Implementación
    return matrix.inverse


def identity_matrix(size: int) -> Matrix:
    """
    Crea una matriz identidad de tamaño especificado.
    
    Args:
        size: El tamaño de la matriz (size x size)
        
    Returns:
        Una nueva matriz identidad
    """
    # Implementación
    if size <= 0:
        raise ValueError("El tamaño de la matriz identidad debe ser positivo")
    return Matrix([[1 if i == j else 0 for j in range(size)] for i in range(size)])



def zeros_matrix(rows: int, columns: int) -> Matrix:
    """
    Crea una matriz de ceros con las dimensiones especificadas.
    
    Args:
        rows: Número de filas
        columns: Número de columnas
        
    Returns:
        Una nueva matriz llena de ceros
    """
    # Implementación
    if rows <= 0 or columns <= 0:
        raise ValueError("Las dimensiones deben ser positivas")
    return Matrix([[0 for _ in range(columns)] for _ in range(rows)])



def ones_matrix(rows: int, columns: int) -> Matrix:
    """
    Crea una matriz de unos con las dimensiones especificadas.
    
    Args:
        rows: Número de filas
        columns: Número de columnas
        
    Returns:
        Una nueva matriz llena de unos
    """
    # Implementación
    if rows <= 0 or columns <= 0:
        raise ValueError("Las dimensiones deben ser positivas")
    return Matrix([[1 for _ in range(columns)] for _ in range(rows)])

# =============================================================================
# Implementaciones Internas para Determinante e Inversa 
# =============================================================================

def _determinant_impl(M: Matrix) -> Number:
    if not M.is_square():
        raise ValueError("El determinante solo está definido para matrices cuadradas")
    n = M.num_rows
    # Copia de trabajo en float
    a = [[float(x) for x in row] for row in M._data]
    sign = 1.0
    det = 1.0
    for col in range(n):
        # Búsqueda de pivote (pivoting parcial)
        pivot_row = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot_row][col]) < _EPS:
            return 0.0
        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]
            sign *= -1.0
        pivot = a[col][col]
        det *= pivot
        # Eliminación hacia abajo
        for r in range(col + 1, n):
            factor = a[r][col] / pivot
            if abs(factor) < _EPS:
                continue
            for c in range(col + 1, n):
                a[r][c] -= factor * a[col][c]
            a[r][col] = 0.0
    return sign * det


def _inverse_impl(M: Matrix) -> Matrix:
    if not M.is_square():
        raise ValueError("La inversa solo está definida para matrices cuadradas")
    n = M.num_rows
    # Matriz aumentada [A | I]
    a = [[float(x) for x in row] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M._data)]

    # Gauss-Jordan con pivoting parcial
    for col in range(n):
        # Seleccionar pivote
        pivot_row = max(range(col, n), key=lambda r: abs(a[r][col]))
        pivot = a[pivot_row][col]
        if abs(pivot) < _EPS:
            raise ValueError("La matriz es singular y no tiene inversa")
        # Intercambiar filas si es necesario
        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]
        # Normalizar fila pivote
        pivot = a[col][col]
        inv_pivot = 1.0 / pivot
        for j in range(2 * n):
            a[col][j] *= inv_pivot
        # Eliminar el resto de filas
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            if abs(factor) < _EPS:
                continue
            for j in range(2 * n):
                a[r][j] -= factor * a[col][j]

    # Extraer la inversa (parte derecha)
    inv = [row[n:] for row in a]
    return Matrix(inv)
"""
    Nota: 
      1. El determinante se calcula por eliminación Gaussiana 
      2. La inversa se calcula usando el método de Gauss-Jordan

    """