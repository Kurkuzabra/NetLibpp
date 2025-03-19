import numpy

def filtrate(A: numpy.ndarray[numpy.float64], n: int, p: numpy.ndarray[numpy.float64], threads: int) -> numpy.ndarray[numpy.float64]:
    ...

class Complex:
    """
    A generic template class that can be instantiated with any type.
    """

    # def __init__(self, value: T) -> None:
    #     """
    #     Initialize the template with a value.

    #     :param value: The initial value.
    #     """
    #     ...

    def get_value(self) -> numpy.float64:
        """
        Get the value stored in the template.

        :return: The stored value.
        """
        ...

def get_VR_from_dist_matrix(A: numpy.ndarray[numpy.float64], max_dist: int, max_dim: int) -> Complex:
    ...

def get_VR_from_coord_matrix(A: numpy.ndarray[numpy.float64], max_dist: int, max_dim: int) -> Complex:
    ...

def get_Lp_from_coord_matrix(A: numpy.ndarray[numpy.float64], max_dist: int, p: numpy.float64, max_dim: int) -> Complex:
    ...