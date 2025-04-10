#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <semaphore>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "Point_t.cpp"
#include "Simplex_t.cpp"
#include "Complex_t.cpp"
#include "VRComplex.cpp"
#include "LpComplex.cpp"
#include "AlphaComplex.cpp"
#include "DelaunayRipsComplex.cpp"
#include "find_comb.cpp"

namespace py = pybind11;
namespace hg = hypergraph;

template <typename T>
std::unique_ptr<hg::VRComplexFromMatrix<hg::ComplexFromDistMatrix, T, hg::PointsType::DIST_PTR>> get_VR_from_dist_matrix(const py::array_t<T> &A, T min_dist, size_t sz)
{
    return std::unique_ptr<hg::VRComplexFromMatrix<hg::ComplexFromDistMatrix, T, hg::PointsType::DIST_PTR>>(
        new hg::VRComplexFromMatrix<hg::ComplexFromDistMatrix, T, hg::PointsType::DIST_PTR>(A, min_dist, sz));
}

template <typename T>
std::unique_ptr<hg::VRComplexFromMatrix<hg::ComplexFromCoordMatrix, T, hg::PointsType::POINT_PTR>> get_VR_from_coord_matrix(const py::array_t<T> &A, T min_dist, size_t sz)
{
    return std::unique_ptr<hg::VRComplexFromMatrix<hg::ComplexFromCoordMatrix, T, hg::PointsType::POINT_PTR>>(
        new hg::VRComplexFromMatrix<hg::ComplexFromCoordMatrix, T, hg::PointsType::POINT_PTR>(A, min_dist, sz));
}

template <typename T>
std::unique_ptr<hg::LpComplexFromMatrix<hg::ComplexFromCoordMatrix, T, hg::PointsType::POINT_PTR>> get_Lp_complex(const py::array_t<T> &A, T min_dist, double p, size_t sz)
{
    return std::unique_ptr<hg::LpComplexFromMatrix<hg::ComplexFromCoordMatrix, T, hg::PointsType::POINT_PTR>>(
        new hg::LpComplexFromMatrix<hg::ComplexFromCoordMatrix, T, hg::PointsType::POINT_PTR>(A, min_dist, p, sz));
}

template <typename T>
std::unique_ptr<hg::AlphaComplexFromMatrix<hg::ComplexFromCoordMatrix, T, hg::PointsType::POINT_PTR>> get_Alpha_complex(const py::array_t<T> &A, T max_rad)
{
    return std::unique_ptr<hg::AlphaComplexFromMatrix<hg::ComplexFromCoordMatrix, T, hg::PointsType::POINT_PTR>>(
        new hg::AlphaComplexFromMatrix<hg::ComplexFromCoordMatrix, T, hg::PointsType::POINT_PTR>(A, max_rad));
}

template <typename T>
std::unique_ptr<hg::DelaunayRipsComplexFromMatrix<hg::ComplexFromCoordMatrix, T, hg::PointsType::POINT_PTR>> get_DelaunayRips_complex(const py::array_t<T> &A, T max_dist)
{
    return std::unique_ptr<hg::DelaunayRipsComplexFromMatrix<hg::ComplexFromCoordMatrix, T, hg::PointsType::POINT_PTR>>(
        new hg::DelaunayRipsComplexFromMatrix<hg::ComplexFromCoordMatrix, T, hg::PointsType::POINT_PTR>(A, max_dist));
}

template <typename T>
hg::Point<T> getPoint(const py::array_t<T> &A)
{
    py::buffer_info A_arr = A.request();
    T *A_ptr = static_cast<T *>(A_arr.ptr);
    size_t N = A_arr.shape[0];
    std::vector<T> pt(N);
    for (size_t i = 0; i < N; i++)
    {
        pt[i] = A_ptr[i];
    }
    return hg::Point<T>(pt);
}

template <typename T>
hg::Simplex<hg::Point<T>, T, hg::PointsType::POINT> get_Simplex_by_points(const py::array_t<T> &points_)
{
    py::buffer_info A_arr = points_.request();
    T *A_ptr = static_cast<T *>(A_arr.ptr);
    size_t N = A_arr.shape[0];
    size_t M = A_arr.shape[1];
    std::vector<hg::Point<T>> points(N, hg::Point<T>(M));
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < M; j++)
        {
            points[i][j] = A_ptr[i * M + j];
        }
    }
    return hg::Simplex<hg::Point<T>, T, hg::PointsType::POINT>(points);
}

#define VAL(str) #str
#define TOSTRING(str) VAL(str)
#define STRCAT(str1, str2) str1 str2

#define declare_Point(Module, Type)                                           \
    {                                                                         \
        py::class_<hg::Point<Type>>(Module, STRCAT("Point_", TOSTRING(Type))) \
            .def(py::init<hg::Point<Type>>())                                 \
            .def("coords", &hg::Point<Type>::operator std::vector<Type> &);   \
    }

#define declare_Simplex(Module, Type1, Type2, Type3)                                                                                                \
    {                                                                                                                                               \
        py::class_<hg::Simplex<Type1, Type2, Type3>>(Module, STRCAT(STRCAT(STRCAT("Simplex_", TOSTRING(Type1)), TOSTRING(Type2)), TOSTRING(Type3))) \
            .def(py::init<hg::Simplex<Type1, Type2, Type3>>())                                                                                      \
            .def("dim", &hg::Simplex<Type1, Type2, Type3>::get_dim)                                                                                 \
            .def("projection", &hg::Simplex<Type1, Type2, Type3>::projection)                                                                       \
            .def("get_volume", &hg::Simplex<Type1, Type2, Type3>::get_volume)                                                                       \
            .def("contains", &hg::Simplex<Type1, Type2, Type3>::contains)                                                                           \
            .def("get_coords", &hg::Simplex<Type1, Type2, Type3>::get_coords)                                                                       \
            .def("get_circumradius", &hg::Simplex<Type1, Type2, Type3>::get_circumsphere_radius) \
            .def("distance", &hg::Simplex<Type1, Type2, Type3>::distance);                                                                          \
    }

#define declare_Simplex_for_all_point_types(Module, Type1, Type2)            \
    {                                                                        \
        declare_Simplex(Module, Type1, Type2, hg::PointsType::DIST_PTR)      \
            declare_Simplex(Module, Type1, Type2, hg::PointsType::POINT_PTR) \
                declare_Simplex(Module, Type1, Type2, hg::PointsType::POINT)}

#define declare_VRComplexFromDistMatrix(Module, Type)                                                                                                              \
    {                                                                                                                                                              \
        py::class_<hg::VRComplexFromMatrix<hg::ComplexFromDistMatrix, Type, hg::PointsType::DIST_PTR>>(Module, STRCAT("VRComplexFromDistMatrix_", TOSTRING(Type))) \
            .def(py::init<const py::array_t<Type> &, Type, size_t>())                                                                                              \
            .def("as_list", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::DIST_PTR>, size_t, Type>::as_list)                                              \
            .def("as_index_list", &hg::ComplexFromMatrix<hg::Simplex<size_t, Type, hg::PointsType::DIST_PTR>, Type>::as_index_list)                                \
            .def("filtration", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::DIST_PTR>, size_t, Type>::filtration)                                        \
            .def("boundary_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::DIST_PTR>, size_t, Type>::boundary_matrix)                              \
            .def("laplace_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::DIST_PTR>, size_t, Type>::laplace_matrix)                                \
            .def("weighted_laplace_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::DIST_PTR>, size_t, Type>::weighted_laplace_matrix)              \
            .def("skeleton", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::DIST_PTR>, size_t, Type>::skeleton);                                           \
    }

#define declare_VRComplexFromCoordMatrix(Module, Type)                                                                                                                \
    {                                                                                                                                                                 \
        py::class_<hg::VRComplexFromMatrix<hg::ComplexFromCoordMatrix, Type, hg::PointsType::POINT_PTR>>(Module, STRCAT("VRComplexFromCoordMatrix_", TOSTRING(Type))) \
            .def(py::init<const py::array_t<Type> &, Type, size_t>())                                                                                                 \
            .def("as_list", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::as_list)                                                \
            .def("as_index_list", &hg::ComplexFromMatrix<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, Type>::as_index_list)                                  \
            .def("as_simplex_list", &hg::VRComplexFromMatrix<hg::ComplexFromCoordMatrix, Type, hg::PointsType::POINT_PTR>::as_simplex_list)                           \
            .def("filtration", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::filtration)                                          \
            .def("simplex_from_indexes", &hg::ComplexFromCoordMatrix<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, Type>::simplex_from_indexes)               \
            .def("projection", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::projection)                                          \
            .def("distance", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::distance)                                              \
            .def("boundary_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::boundary_matrix)                                \
            .def("laplace_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::laplace_matrix)                                  \
            .def("weighted_laplace_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::weighted_laplace_matrix)                \
            .def("skeleton", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::skeleton);                                             \
    }

#define declare_LpComplexFromCoordMatrix(Module, Type)                                                                                                                \
    {                                                                                                                                                                 \
        py::class_<hg::LpComplexFromMatrix<hg::ComplexFromCoordMatrix, Type, hg::PointsType::POINT_PTR>>(Module, STRCAT("LpComplexFromCoordMatrix_", TOSTRING(Type))) \
            .def(py::init<const py::array_t<Type> &, Type, double, size_t>())                                                                                         \
            .def("as_list", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::as_list)                                                \
            .def("as_index_list", &hg::ComplexFromMatrix<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, Type>::as_index_list)                                  \
            .def("as_simplex_list", &hg::LpComplexFromMatrix<hg::ComplexFromCoordMatrix, Type, hg::PointsType::POINT_PTR>::as_simplex_list)                           \
            .def("filtration", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::filtration)                                          \
            .def("simplex_from_indexes", &hg::ComplexFromCoordMatrix<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, Type>::simplex_from_indexes)               \
            .def("projection", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::projection)                                          \
            .def("distance", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::distance)                                              \
            .def("boundary_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::boundary_matrix)                                \
            .def("laplace_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::laplace_matrix)                                  \
            .def("weighted_laplace_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::weighted_laplace_matrix)                \
            .def("skeleton", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::skeleton);                                             \
    }

#define declare_AlphaComplexFromCoordMatrix(Module, Type)                                                                                                                   \
    {                                                                                                                                                                       \
        py::class_<hg::AlphaComplexFromMatrix<hg::ComplexFromCoordMatrix, Type, hg::PointsType::POINT_PTR>>(Module, STRCAT("AlphaComplexFromCoordMatrix_", TOSTRING(Type))) \
            .def(py::init<const py::array_t<Type> &, Type>())                                                                                                               \
            .def("as_list", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::as_list)                                                      \
            .def("as_index_list", &hg::ComplexFromMatrix<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, Type>::as_index_list)                                        \
            .def("as_simplex_list", &hg::AlphaComplexFromMatrix<hg::ComplexFromCoordMatrix, Type, hg::PointsType::POINT_PTR>::as_simplex_list)                              \
            .def("filtration", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::filtration)                                                \
            .def("simplex_from_indexes", &hg::ComplexFromCoordMatrix<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, Type>::simplex_from_indexes)                     \
            .def("projection", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::projection)                                                \
            .def("distance", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::distance)                                                    \
            .def("boundary_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::boundary_matrix)                                      \
            .def("laplace_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::laplace_matrix)                                        \
            .def("weighted_laplace_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::weighted_laplace_matrix)                      \
            .def("skeleton", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::skeleton);                                                   \
    }

#define declare_DelaunayRipsComplexFromCoordMatrix(Module, Type)                                                                                                                               \
    {                                                                                                                                                                                          \
        py::class_<hg::DelaunayRipsComplexFromMatrix<hg::ComplexFromCoordMatrix, Type, hg::PointsType::POINT_PTR>>(Module, STRCAT("DelaunayRipsComplexFromCoordMatrix_", TOSTRING(Type))) \
            .def(py::init<const py::array_t<Type> &, Type>())                                                                                                                                  \
            .def("as_list", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::as_list)                                                                         \
            .def("as_index_list", &hg::ComplexFromMatrix<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, Type>::as_index_list)                                                           \
            .def("as_simplex_list", &hg::DelaunayRipsComplexFromMatrix<hg::ComplexFromCoordMatrix, Type, hg::PointsType::POINT_PTR>::as_simplex_list)                                     \
            .def("filtration", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::filtration)                                                                   \
            .def("simplex_from_indexes", &hg::ComplexFromCoordMatrix<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, Type>::simplex_from_indexes)                                        \
            .def("projection", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::projection)                                                                   \
            .def("distance", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::distance)                                                                       \
            .def("boundary_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::boundary_matrix)                                                         \
            .def("laplace_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::laplace_matrix)                                                           \
            .def("weighted_laplace_matrix", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::weighted_laplace_matrix)                                         \
            .def("skeleton", &hg::Complex<hg::Simplex<size_t, Type, hg::PointsType::POINT_PTR>, size_t, Type>::skeleton);                                                                      \
    }

#define declare_VR_complexes(Module, Type)                                   \
    {                                                                        \
        declare_Simplex(Module, size_t, Type, hg::PointsType::DIST_PTR)      \
            declare_Simplex(Module, size_t, Type, hg::PointsType::POINT_PTR) \
                declare_VRComplexFromDistMatrix(Module, Type)                \
                    declare_VRComplexFromCoordMatrix(Module, Type)           \
                        declare_LpComplexFromCoordMatrix(Module, Type)}

PYBIND11_MODULE(netlibpp_cpy, m)
{
    // AlphaComplex can only be double because of qhull lib
    declare_AlphaComplexFromCoordMatrix(m, double)
    declare_DelaunayRipsComplexFromCoordMatrix(m, double)
        declare_VR_complexes(m, float)
            declare_VR_complexes(m, double)

        // declare_VRComplexFromDistMatrix(m, double)
        // declare_VRComplexFromDistMatrix(m, float)

        // declare_VRComplexFromCoordMatrix(m, double)
        // declare_VRComplexFromCoordMatrix(m, float)

        m.def("get_VR_from_dist_matrix", py::overload_cast<const py::array_t<double> &, double, size_t>(&get_VR_from_dist_matrix<double>), "");
    m.def("get_VR_from_dist_matrix", py::overload_cast<const py::array_t<float> &, float, size_t>(&get_VR_from_dist_matrix<float>), "");

    m.def("get_VR_from_coord_matrix", py::overload_cast<const py::array_t<double> &, double, size_t>(&get_VR_from_coord_matrix<double>), "");
    m.def("get_VR_from_coord_matrix", py::overload_cast<const py::array_t<float> &, float, size_t>(&get_VR_from_coord_matrix<float>), "");

    m.def("get_Lp_from_coord_matrix", py::overload_cast<const py::array_t<double> &, double, double, size_t>(&get_Lp_complex<double>), "");
    m.def("get_Lp_from_coord_matrix", py::overload_cast<const py::array_t<float> &, float, double, size_t>(&get_Lp_complex<float>), "");

    m.def("get_Alpha_from_coord_matrix", py::overload_cast<const py::array_t<double> &, double>(&get_Alpha_complex<double>), "");
    m.def("get_DelaunayRips_from_coord_matrix", py::overload_cast<const py::array_t<double> &, double>(&get_DelaunayRips_complex<double>), "");

    declare_Point(m, double)
        declare_Point(m, float)
            m.def("Point", py::overload_cast<const py::array_t<double> &>(&getPoint<double>), "get Point");
    m.def("Point", py::overload_cast<const py::array_t<float> &>(&getPoint<float>), "get Point");

    // declare_Simplex(m, hg::Point<double>, double)
    // declare_Simplex(m, hg::Point<float>, float)
    declare_Simplex(m, hg::Point<double>, double, hg::PointsType::POINT)
        declare_Simplex(m, hg::Point<float>, float, hg::PointsType::POINT)
            m.def("Simplex", py::overload_cast<const py::array_t<double> &>(&get_Simplex_by_points<double>), "get Simplex");
    m.def("Simplex", py::overload_cast<const py::array_t<float> &>(&get_Simplex_by_points<float>), "get Simplex");
}
