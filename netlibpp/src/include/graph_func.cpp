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
#include "find_comb.cpp"

PYBIND11_MAKE_OPAQUE(std::vector<std::array<double, 3>>)

#define EPSILON 0.000001
#define MAX_SEM_VAL 100000

namespace py = pybind11;
namespace hg = hypergraph;

// std::binary_semaphore smphSignalThreadToMain{0};
// std::counting_semaphore free_sem{0};

template<typename T>
double d(double* A_ptr, int A_shape, T& simplex_ptr, std::vector<int>& perm, double p)
{
    Subsequences subs = Subsequences(perm.size(), 2);
    double ds = std::numeric_limits<double>::min();

    if (std::fabs(p - (double)1) < EPSILON)
    {
        while(subs.next())
        {
            double norm_sqr = 0.0;
            const std::vector<int>& item = subs.get_subseq();
            for (std::size_t i = 0; i < item.size() - 1; i++)
            {
                norm_sqr += A_ptr[simplex_ptr[perm[item[i]]] * A_shape + simplex_ptr[perm[item[i + 1]]]];
            }
            ds = std::max(ds, norm_sqr);
        }
    }
    else if (p == std::numeric_limits<double>::infinity())
    {
        while(subs.next())
        {
            double norm_sqr = std::numeric_limits<float>::min();
            const std::vector<int>& item = subs.get_subseq();
            for (std::size_t i = 0; i < item.size() - 1; i++)
            {
                norm_sqr = std::max(norm_sqr, A_ptr[simplex_ptr[perm[item[i]]] * A_shape + simplex_ptr[perm[item[i + 1]]]]);
            }
            ds = std::max(ds, norm_sqr);
        }
    }
    else
    {
        while(subs.next())
        {
            double norm_sqr = 0.0;
            const std::vector<int>& item = subs.get_subseq();
            for (std::size_t i = 0; i < item.size() - 1; i++)
            {
                norm_sqr += std::pow(std::fabs(A_ptr[simplex_ptr[perm[item[i]]] * A_shape + simplex_ptr[perm[item[i + 1]]]]), p);
            }
            ds = std::max(ds, std::pow(norm_sqr, 1.0 / p));
        }
    }
    return ds;
}

double f_single_thread_(double* A_ptr, int A_sz, std::vector<int> simplex, double p)
{
    std::vector<int> perm(simplex.size());
    double fs = std::numeric_limits<double>::max();

    for (std::size_t i = 0; i < perm.size(); i++) perm[i] = i;
    do
    {
        fs = std::min(fs, d(A_ptr, A_sz, simplex, perm, p));
    } while(std::next_permutation(perm.begin(), perm.end()));

    return fs;   
}

void f_multithread_part_
(
    std::vector<std::vector<double>>& result, double* A_ptr, int A_sz, double* p_ptr, int p_sz, std::vector<int>& beg_comb,
    long long start_offset, long long tasks, std::binary_semaphore& smphSignalThreadToMain, std::counting_semaphore<MAX_SEM_VAL>& free_sem)
{
    Combinations comb(A_sz, beg_comb.size(), beg_comb);
    smphSignalThreadToMain.release();
    long long i = 0;
    do
    {
        const std::vector<int>& simplex = comb.get_comb();

        for (int j = 0; j < p_sz; j++)
        {
            result[start_offset + i][j] = f_single_thread_(A_ptr, A_sz, simplex, p_ptr[j]);
        }

        i++;
    } while (comb.next() && i < tasks);
    free_sem.release();
}

py::array_t<double> filtrate(const py::array_t<double>& A, int simplex_sz, const py::array_t<double>& p, int num_threads = 1)
{
    py::buffer_info A_arr = A.request();
    py::buffer_info p_arr = p.request();
    int A_sz = A_arr.shape[0];
    const int p_sz = p_arr.shape[0];
    double* A_ptr = static_cast<double *>(A_arr.ptr);
    double* p_ptr = static_cast<double *>(p_arr.ptr);
    std::vector<std::vector<double>> result(nChoosek(A_sz, simplex_sz));

    

    for (int i = 0; i < result.size(); i++)
    {
        result[i] = std::vector<double>(p_sz);
    }
    Combinations comb(A_sz, simplex_sz);
    long long i = 0;
    if (num_threads == 1)
    {
        do
        {
            const std::vector<int>& simplex = comb.get_comb();
            for (int j = 0; j < p_sz; j++)
            {
                result[i][j] = f_single_thread_(A_ptr, A_sz, simplex, p_ptr[j]);
            }
            i++;
        } while(comb.next());
    }
    else
    {
        std::binary_semaphore smphSignalThreadToMain{0};
        std::counting_semaphore<MAX_SEM_VAL> free_sem{0};
        int64_t total_comb;
        compute_total_comb(A_sz, simplex_sz, total_comb);
        long long tasks = total_comb / num_threads;
        if (total_comb % tasks > 0)
        {
            num_threads++;
        }

        for (long long i = 0; i < num_threads; i++)
        {
            std::vector<int> curr_comb(simplex_sz);
            find_comb(A_sz, simplex_sz, tasks * i, curr_comb);
            std::thread thr
            (
                f_multithread_part_, std::ref(result), A_ptr, A_sz, p_ptr, p_sz, std::ref(curr_comb), i * tasks,
                tasks, std::ref(smphSignalThreadToMain), std::ref(free_sem)
            );
            thr.detach();
            smphSignalThreadToMain.acquire();
        }

        for (int i = 0; i < num_threads; i++)
        {
            free_sem.acquire();
        }
    }
    return py::array_t<double>(py::cast(std::ref(result)));
}


template <typename T>
std::unique_ptr<hg::VRComplexFromMatrix<hg::ComplexFromDistMatrix, T>> get_VR_from_dist_matrix(const py::array_t<T>& A, T min_dist, size_t sz)
{
    return std::unique_ptr<hg::VRComplexFromMatrix<hg::ComplexFromDistMatrix, T>>(new hg::VRComplexFromMatrix<hg::ComplexFromDistMatrix, T>(A, min_dist, sz));
}

template <typename T>
std::unique_ptr<hg::VRComplexFromMatrix<hg::ComplexFromCoordMatrix, T>> get_VR_from_coord_matrix(const py::array_t<T>& A, T min_dist, size_t sz)
{
    return std::unique_ptr<hg::VRComplexFromMatrix<hg::ComplexFromCoordMatrix, T>>(new hg::VRComplexFromMatrix<hg::ComplexFromCoordMatrix, T>(A, min_dist, sz));
}

template <typename T>
std::unique_ptr<hg::LpComplexFromMatrix<hg::ComplexFromCoordMatrix, T>> get_Lp_complex(const py::array_t<T>& A, T min_dist, double p, size_t sz)
{
    return std::unique_ptr<hg::LpComplexFromMatrix<hg::ComplexFromCoordMatrix, T>>(new hg::LpComplexFromMatrix<hg::ComplexFromCoordMatrix, T>(A, min_dist, p, sz));
}

template <typename T>
hg::Point<T> getPoint(const py::array_t<T>& A)
{
    py::buffer_info A_arr = A.request();
    T* A_ptr = static_cast<T *>(A_arr.ptr);
    size_t N = A_arr.shape[0];
    std::vector<T> pt(N);
    for (size_t i = 0; i < N; i++)
    {
        pt[i] = A_ptr[i];
    }
    return hg::Point<T>(pt);
}

template <typename T>
hg::Simplex<hg::Point<T>, T> get_Simplex_by_points(const py::array_t<T>& points_)
{
    py::buffer_info A_arr = points_.request();
    T* A_ptr = static_cast<T *>(A_arr.ptr);
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
    return hg::Simplex<hg::Point<T>, T>(points);
}

#define VAL(str) #str
#define TOSTRING(str) VAL(str)
#define STRCAT(str1, str2) str1 str2

#define declare_Point(Module, Type) \
{ \
    py::class_<hg::Point<Type>>(Module, STRCAT("Point_", TOSTRING(Type))) \
        .def(py::init<hg::Point<Type>>()) \
        .def("coords", &hg::Point<Type>::operator std::vector<Type>); \
}

#define declare_Simplex(Module, Type1, Type2) \
{ \
    py::class_<hg::Simplex<Type1, Type2>>(Module, STRCAT(STRCAT("Simplex_", TOSTRING(Type1)), TOSTRING(Type2))) \
        .def(py::init<hg::Simplex<Type1, Type2>>()) \
        .def("dim", &hg::Simplex<Type1, Type2>::get_dim) \
        .def("projection", &hg::Simplex<Type1, Type2>::projection); \
}

#define declare_VRComplexFromDistMatrix(Module, Type) \
{ \
    py::class_<hg::VRComplexFromMatrix<hg::ComplexFromDistMatrix, Type>>(Module, STRCAT("VRComplexFromDistMatrix_", TOSTRING(Type))) \
        .def(py::init<const py::array_t<Type>&, Type, size_t>()) \
        .def("volume_of", &hg::VRComplexFromMatrix<hg::ComplexFromDistMatrix, Type>::volume_of) \
        .def("as_list", &hg::VRComplexFromMatrix<hg::ComplexFromDistMatrix, Type>::as_list) \
        .def("filtration", &hg::Complex<hg::Simplex<size_t, Type>, size_t, Type>::filtration) \
        .def("skeleton", &hg::Complex<hg::Simplex<size_t, Type>, size_t, Type>::skeleton); \
}

#define declare_VRComplexFromCoordMatrix(Module, Type) \
{ \
    py::class_<hg::VRComplexFromMatrix<hg::ComplexFromCoordMatrix, Type>>(Module, STRCAT("VRComplexFromCoordMatrix_", TOSTRING(Type))) \
        .def(py::init<const py::array_t<Type>&, Type, size_t>()) \
        .def("volume_of", &hg::VRComplexFromMatrix<hg::ComplexFromCoordMatrix, Type>::volume_of) \
        .def("as_list", &hg::VRComplexFromMatrix<hg::ComplexFromCoordMatrix, Type>::as_list) \
        .def("filtration", &hg::Complex<hg::Simplex<size_t, Type>, size_t, Type>::filtration) \
        .def("skeleton", &hg::Complex<hg::Simplex<size_t, Type>, size_t, Type>::skeleton); \
}

#define declare_LpComplexFromCoordMatrix(Module, Type) \
{ \
    py::class_<hg::LpComplexFromMatrix<hg::ComplexFromCoordMatrix, Type>>(Module, STRCAT("LpComplexFromCoordMatrix_", TOSTRING(Type))) \
        .def(py::init<const py::array_t<Type>&, Type, double, size_t>()) \
        .def("as_list", &hg::LpComplexFromMatrix<hg::ComplexFromCoordMatrix, Type>::as_list) \
        .def("filtration", &hg::Complex<hg::Simplex<size_t, Type>, size_t, Type>::filtration) \
        .def("skeleton", &hg::Complex<hg::Simplex<size_t, Type>, size_t, Type>::skeleton); \
}

#define declare_VR_complexes(Module, Type) \
{ \
    declare_Simplex(Module, size_t, Type) \
    declare_VRComplexFromDistMatrix(Module, Type) \
    declare_VRComplexFromCoordMatrix(Module, Type) \
    declare_LpComplexFromCoordMatrix(Module, Type) \
}

PYBIND11_MODULE(netlibpp_cpy, m)
{
    m.doc() = "pybind11 graph filtration";
    m.def("filtrate", &filtrate, "filter complex", py::arg("A"), py::arg("n"), py::arg("p"), py::arg("threads"));

    declare_VR_complexes(m, float)
    declare_VR_complexes(m, double)

    // declare_VRComplexFromDistMatrix(m, double)
    // declare_VRComplexFromDistMatrix(m, float)

    // declare_VRComplexFromCoordMatrix(m, double)
    // declare_VRComplexFromCoordMatrix(m, float)

    m.def("get_VR_from_dist_matrix", py::overload_cast<const py::array_t<double>&, double, size_t>(&get_VR_from_dist_matrix<double>), "get VR complex");
    m.def("get_VR_from_dist_matrix", py::overload_cast<const py::array_t<float>&, float, size_t>(&get_VR_from_dist_matrix<float>), "get VR complex");

    m.def("get_VR_from_coord_matrix", py::overload_cast<const py::array_t<double>&, double, size_t>(&get_VR_from_coord_matrix<double>), "get VR complex");
    m.def("get_VR_from_coord_matrix", py::overload_cast<const py::array_t<float>&, float, size_t>(&get_VR_from_coord_matrix<float>), "get VR complex");

    m.def("get_Lp_from_coord_matrix", py::overload_cast<const py::array_t<double>&, double, double, size_t>(&get_Lp_complex<double>), "get Lp complex");
    m.def("get_Lp_from_coord_matrix", py::overload_cast<const py::array_t<float>&, float, double, size_t>(&get_Lp_complex<float>), "get Lp complex");

    declare_Point(m, double)
    declare_Point(m, float)
    m.def("Point", py::overload_cast<const py::array_t<double>&>(&getPoint<double>), "get Point");
    m.def("Point", py::overload_cast<const py::array_t<float>&>(&getPoint<float>), "get Point");

    declare_Simplex(m, hg::Point<double>, double)
    declare_Simplex(m, hg::Point<float>, float)
    m.def("Simplex", py::overload_cast<const py::array_t<double>&>(&get_Simplex_by_points<double>), "get Simplex");
    m.def("Simplex", py::overload_cast<const py::array_t<float>&>(&get_Simplex_by_points<float>), "get Simplex");

    // py::class_<hg::bind_Simplex<hg::Point<double>, double>>(m, "Simplex")
    //     .def(py::init<const py::array_t<double>&>())
    //     .def("projection", &hg::bind_Simplex<hg::Point<double>, double>::projection);
}
