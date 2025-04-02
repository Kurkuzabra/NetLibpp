import numpy
from typing import List, TypeVar, Generic

T = TypeVar("T", numpy.float32, numpy.float64)

def filtrate(A: numpy.ndarray[numpy.float64], n: int, p: numpy.ndarray[numpy.float64], threads: int) -> numpy.ndarray[numpy.float64]:
    ...

class Point(Generic[T]):
    """
    A template class for points.

    Cannot be initialized within python __init__. 
    """

    def coords(self) -> List[T]:
        """
        Returns coordinates of point.

        for Point[unsigned] returns position in Matrix.
        
        :returns: coordinates.
        """
        ...

class PointIndex(Generic[T]):
    """
    A template class for point index in matrix.

    Cannot be initialized within python __init__. 
    """
    ...


Point_t = TypeVar("Point_t", Point[T], PointIndex[T])

class Simplex(Generic[Point_t, T]):
    """
    A template class for simplexes.

    Cannot be initialized within python __init__.    
    """

    def dim(self):
        """
        dimension of a simplex.

        currently returns number of points in simplex

        :returns: dimension.
        """
        ...
    
    def get_volume(self) -> T:
        """
        Gets volume of simplex in n dimensions. If its matrix is not full-rank, returns 0.

        :returns: volume of simplex
        """
        ...

    def projection(self, point: Point[T]) -> Point_t:
        """
        Gets the nearest point on Simplex.
            
        :returns: projection onto Simplex

        :requires:
            - point has to be the same type as simplex.
            - point must not be on any edge of simplex.
            - Point_t must be Point[T].
        """
        ...

    def distance(self, point: Point[T]) -> T:
        """
        Gets distance to the nearest point on Simplex.
            
        :returns: distance to Simplex

        :requires:
            - point has to be the same type as simplex.
            - point must not be on any edge of simplex.
            - Point_t must be Point[T].
        """
        ...

class Complex(Generic[Point_t, T]):
    """
    A template class for complexes.

    Cannot be initialized within python __init__.
    """

    def skeleton(self, p: int) -> Complex[Point_t, T]:
        """
        A p-skeleton is a subcomplex of a complex with simplices of dimension <= p.

        :param p: dimension of subcomplex.

        :returns: New Complex with applied condition.
        """
        ...
    
    def as_list(self) -> List[List[Simplex[Point_t, T]]]:
        """
        Returns complex as List[List[Simplex]]

        each node has List[Simplex], where Simplex dim = node index

        :returns: Complex as list
        """
        ...

    def projection(self, point: Point[T]) -> List[Point[T]]:
        """
        projection of any point in R^d to a complex
        defined as the (all) minimum distance projection to a complex's simplices
        finding and returning projection point(s)
        
        :returns: projection to complex
        """
        ...

    def distance(self, point: Point[T]) -> T:
        """
        distance of any point in R^d to a complex (its convex hull)
        defined as the (all) minimum distances to a complex's simplices
        computing the distance(s) between a point and its projection to a simplex
        
        :returns: distance to complex
        """
        ...

    def boundary_matrix() -> numpy.array[T]:
        """
        computes boundary matrix of s_dim - 1 and s_dim simplexes

        boundary_matrix[i][j] = simplexes[s_dim][j].contains(simplexes[s_dim - 1][i])

        :returns: boundary matrix of complex
        """
        ...

    def laplace_matrix() -> numpy.array[T]:
        """
        computes laplace matrix of s_dim - 1 and s_dim simplexes

        L_[k] = B_[k].T * W_[k] * B_[k] + B_[k+1] * W_[k] * B_[k+1].T

        L_[0] = B_[1] * B_[1]^T

        ??? says special case but can be computed by normal formula ???
        
        to be discussed 

        :returns: laplacian of complex
        """
        ...

class ComplexFromMatrix(Complex[Point_t, T]):
    """
    A class for complexes from matricies(distance/coordinates).

    Cannot be initialized within python __init__.
    """

    def as_index_list(self) -> List[List[List[numpy.unsigned]]]:
        """
        Returns simplexes as list of their points indexes in matrix rows.

        :returns: Complex as index list
        """
        ...

class ComplexFromCoordMatrix(ComplexFromMatrix[Point_t, T]):
    """
    A template class for complexes from coordinates matricies.

    Cannot be initialized within python __init__.
    """

    def as_simplex_list(self) -> List[List[Simplex[T]]]:
        """
        Returns complex as List[List[Simplex]]

        each node has List[Simplex], where Simplex dim = node index

        :returns: Complex as list.
        """

class ComplexFromDistMatrix(ComplexFromMatrix[Point_t, T]):
    """
    A template class for complexes from coordinates matricies.

    Cannot be initialized within python __init__.
    """


def get_VR_from_dist_matrix(A: numpy.ndarray[T], max_dist: int, max_dim: int) -> ComplexFromCoordMatrix[PointIndex[T], T]:
    ...

def get_VR_from_coord_matrix(A: numpy.ndarray[T], max_dist: int, max_dim: int) -> ComplexFromDistMatrix[PointIndex[T], T]:
    ...

def get_Lp_from_coord_matrix(A: numpy.ndarray[T], max_dist: int, p: numpy.float64, max_dim: int) -> ComplexFromCoordMatrix[PointIndex[T], T]:
    ...