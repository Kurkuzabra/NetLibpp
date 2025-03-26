#pragma once

#include <vector>
#include <limits>
#include <stdexcept>
#include <functional>
#include <thread>
#include <mutex>
#include "Point_t.cpp"
#include "Simplex_t.cpp"

#ifndef PYBIND11_INCLUDE
#define PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#endif

#define EPSILON 0.000001

namespace py = pybind11;

namespace hypergraph
{

    template <typename Simplex_t, typename Point_t, typename T>
    struct Complex
    {
    protected:
        std::mutex cmplx_mtx;

    public:
        size_t dim;
        // dimension of a complex (the largest dimension of a simplex in a complex)
        std::vector<std::vector<Simplex_t>> simplexes;
        // all simplexes of the complex
        // simplexes[simplexse.size() - 1] is always an empty vector

        Complex() : simplexes(std::vector<std::vector<Simplex_t>>(1)), dim(0) {}
        Complex(const Complex &cplx) : simplexes(cplx.simplexes), dim(cplx.dim) {}
        Complex(Complex &&cplx) : simplexes(std::move(cplx.simplexes)), dim(cplx.dim) {}
        Complex &operator=(const Complex &cplx)
        {
            simplexes = cplx.simplexes;
            dim = cplx.dim;
            return *this;
        }
        Complex &operator=(Complex &&cplx)
        {
            simplexes = std::move(cplx.simplexes);
            dim = cplx.dim;
            return *this;
        }
        ~Complex() = default;

        void append(const Simplex_t &simplex_);
        void safe_append(const Simplex_t &simplex_); // for multithreading

        const std::vector<Simplex_t> &filtration(size_t);
        // all n-simplexes

        Complex skeleton(size_t);
        // a p-skeleton is a subcomplex of a complex with simplices of dimension <= p
        // dimension of a p-skeleton is p

        std::vector<Point_t> projection(const Point_t &point);
        // projection \pi_p of any point in R^d to a complex
        // defined as the (all) minimum distance projection to a complex's simplices
        // finding and returning projection point(s)

        T distance(Point_t point);
        // distance of any point in R^d to a complex (its convex hull)
        // defined as the (all) minimum distances to a complex's simplices
        // computing the distance(s) between a point and its projection to a simplex

        py::list as_list()
        {
            return py::cast(simplexes);
        }
    };

    template <typename Simplex_t, typename T>
    struct ComplexFromMatrix : public Complex<Simplex_t, size_t, T>
    {
        T *dist_ptr;
        size_t N;
        size_t M;

        inline T *from_idx(const size_t &i)
        {
            return dist_ptr + i * M;
        }

        py::list as_index_list()
        {
            std::vector<std::vector<std::vector<size_t>>> indexes;
            for (size_t i = 0; i < this->simplexes.size(); i++)
            {
                indexes.push_back(std::vector<std::vector<size_t>>(0));
                for (size_t j = 0; j < this->simplexes[i].size(); j++)
                {
                    indexes[i].push_back(std::vector<size_t>(0));
                    std::vector<size_t> &vec = static_cast<std::vector<size_t> &>(this->simplexes[i][j]);
                    for (size_t k = 0; k < vec.size(); k++)
                    {
                        indexes[i][j].push_back(vec[k]);
                    }
                }
            }
            return py::cast(indexes);
        }

        ComplexFromMatrix() : dist_ptr(nullptr), N(0), M(0) {}
        ComplexFromMatrix(const py::array_t<T> &A);
        ComplexFromMatrix(const ComplexFromMatrix &other);
        ComplexFromMatrix(ComplexFromMatrix &&other);
        ComplexFromMatrix &operator=(const ComplexFromMatrix &other);
        ComplexFromMatrix &operator=(ComplexFromMatrix &&other);
        ~ComplexFromMatrix();
    };

    template <typename Simplex_t, typename T>
    struct ComplexFromDistMatrix : public ComplexFromMatrix<Simplex_t, T>
    {
        // functions to get distance between points
        std::function<T(const size_t, const size_t)> dist_idx =
            [&](const size_t i, const size_t j)
        {
            return this->dist_ptr[i * this->M + j];
        };
        std::function<T(const size_t &, const size_t &)> dist =
            [&](const size_t &A, const size_t &B)
        {
            return dist_idx(A, B);
        };

        T volume_of(Simplex<size_t, T> simplex)
        {
            return simplex.get_volume(this->dist);
        }

        // needed for pybind11 to properly copy/move objects
        ComplexFromDistMatrix(const py::array_t<T> &A) : ComplexFromMatrix<Simplex_t, T>(A) {}
        ComplexFromDistMatrix(const ComplexFromDistMatrix &other) : ComplexFromMatrix<Simplex_t, T>(other) {}
        ComplexFromDistMatrix(const ComplexFromDistMatrix &&other) : ComplexFromMatrix<Simplex_t, T>(std::move(other)) {}
        ComplexFromDistMatrix &operator=(const ComplexFromDistMatrix &other)
        {
            ComplexFromMatrix<Simplex_t, T>::operator=(other);
            return *this;
        }
        ComplexFromDistMatrix &operator=(const ComplexFromDistMatrix &&other)
        {
            ComplexFromMatrix<Simplex_t, T>::operator=(std::move(other));
            return *this;
        }
        ~ComplexFromDistMatrix() {}
    };

    template <typename Simplex_t, typename T>
    struct ComplexFromCoordMatrix : public ComplexFromMatrix<Simplex_t, T>
    {

        // functions to get distance between points
        std::function<T(const size_t, const size_t)> dist_idx =
            [&](const size_t i, const size_t j)
        {
            T res = 0.0;
            for (size_t k = 0; k < this->M; k++)
            {
                res += std::pow(this->dist_ptr[i * this->M + k] - this->dist_ptr[j * this->M + k], 2);
            }
            return std::sqrt(res);
        };
        std::function<T(const size_t &, const size_t &)> dist =
            [&](const size_t &A, const size_t &B)
        {
            return dist_idx(A, B);
        };
        // functions to get lp-distance between points
        std::function<T(const size_t &, const size_t &, const double &)> lp_dist_idx =
            [&](const size_t &i, const size_t &j, const double &p)
        {
            T res = 0.0;
            for (size_t k = 0; k < this->M; k++)
            {
                res += std::pow(this->dist_ptr[i * this->M + k] - this->dist_ptr[j * this->M + k], p);
            }
            return std::pow(res, 1.0 / p);
        };
        std::function<T(const size_t &, const size_t &, const double &)> lp_dist =
            [&](const size_t &A, const size_t &B, const double &p)
        {
            return lp_dist_idx(A, B, p);
        };

        Simplex<Point<T>, T> simplex_from_indexes(Simplex_t& splx)
        {
            const std::vector<size_t> &vec = static_cast<const std::vector<size_t> &>(splx);
            std::vector<T *> vec_(vec.size());
            for (size_t k = 0; k < vec.size(); k++)
            {
                vec_[k] = this->from_idx(vec[k]);
            }
            Simplex<Point<T>, T> pt_splx{};
            pt_splx.dim = splx.get_dim();
            pt_splx.volume = splx.get_volume_();
            pt_splx.filter = splx.get_filter_();
            pt_splx.set_vectors(vec_, vec_.size(), this->M);
            return pt_splx;
        }

        py::list as_simplex_list()
        {
            std::vector<std::vector<Simplex<Point<T>, T>>> indexes(this->simplexes.size());
            for (size_t i = 0; i < this->simplexes.size(); i++)
            {
                for (size_t j = 0; j < this->simplexes[i].size(); j++)
                {
                    // indexes[i].push_back(Simplex<std::vector<T>, T>(0));
                    const std::vector<size_t> &vec = static_cast<const std::vector<size_t> &>(this->simplexes[i][j]);
                    std::vector<T *> vec_(vec.size());
                    for (size_t k = 0; k < vec.size(); k++)
                    {
                        vec_[k] = this->from_idx(vec[k]);
                    }
                    Simplex<Point<T>, T> splx{};
                    splx.dim = this->simplexes[i][j].get_dim();
                    splx.volume = this->simplexes[i][j].get_volume_();
                    splx.filter = this->simplexes[i][j].get_filter_();
                    splx.set_vectors(vec_, vec_.size(), this->M);
                    indexes[i].push_back(std::move(splx));
                }
            }

            return py::cast(indexes);
        }

        T volume_of(Simplex<size_t, T> simplex)
        {
            return simplex.get_volume(this->dist);
        }

        // needed for pybind11 to properly copy/move objects
        ComplexFromCoordMatrix(const py::array_t<T> &A) : ComplexFromMatrix<Simplex_t, T>(A) {}
        ComplexFromCoordMatrix(const ComplexFromCoordMatrix &other) : ComplexFromMatrix<Simplex_t, T>(other) {}
        ComplexFromCoordMatrix(const ComplexFromCoordMatrix &&other) : ComplexFromMatrix<Simplex_t, T>(std::move(other)) {}
        ComplexFromCoordMatrix &operator=(const ComplexFromCoordMatrix &other)
        {
            ComplexFromMatrix<Simplex_t, T>::operator=(other);
            return *this;
        }
        ComplexFromCoordMatrix &operator=(const ComplexFromCoordMatrix &&other)
        {
            ComplexFromMatrix<Simplex_t, T>::operator=(std::move(other));
            return *this;
        }
        ~ComplexFromCoordMatrix() {}
    };

    template <typename Simplex_t, typename Point_t, typename T>
    void Complex<Simplex_t, Point_t, T>::append(const Simplex_t &simplex_)
    {
        size_t dim_ = simplex_.get_dim();
        if (dim_ > dim)
        {
            dim = dim_;
            simplexes.resize(dim + 1);
        }
        simplexes[dim_].push_back(simplex_);
    }

    template <typename Simplex_t, typename Point_t, typename T>
    void Complex<Simplex_t, Point_t, T>::safe_append(const Simplex_t &simplex_)
    {
        const std::lock_guard<std::mutex> lock(cmplx_mtx);
        append(simplex_);
    }

    template <typename Simplex_t, typename Point_t, typename T>
    const std::vector<Simplex_t> &Complex<Simplex_t, Point_t, T>::filtration(size_t n)
    {
        if (n > dim)
            return simplexes[simplexes.size() - 1];
        return simplexes[n];
    }

    template <typename Simplex_t, typename Point_t, typename T>
    Complex<Simplex_t, Point_t, T> Complex<Simplex_t, Point_t, T>::skeleton(size_t p)
    {
        Complex<Simplex_t, Point_t, T> skeleton_complex = Complex<Simplex_t, Point_t, T>();
        skeleton_complex.simplexes.insert(skeleton_complex.simplexes.end(), simplexes.begin(),
                                          p >= simplexes.size() ? simplexes.end() : simplexes.begin() + p + 1);
        skeleton_complex.dim = p >= simplexes.size() ? (simplexes.size() == 0 ? 0 : simplexes.size() - 1) : p;
        return skeleton_complex;
    }

    template <typename Simplex_t, typename Point_t, typename T>
    std::vector<Point_t> Complex<Simplex_t, Point_t, T>::projection(const Point_t &point)
    {
        std::vector<Point_t> res;
        T n_dist = 0.0;
        for (size_t i = 0; i < this->simplexes.size(); i++)
        {
            for (size_t j = 0; j < this->simplexes[i].size(); j++)
            {
                T c_dist = 0.0;
                std::vector<Point_t> simpl_projs = this->simplexes[i][j].projection_impl(point, c_dist);
                if (c_dist < n_dist - EPSILON || res.size() == 0)
                {
                    res = simpl_projs;
                }
                else if (c_dist < n_dist + EPSILON)
                {
                    res.insert(res.end(), simpl_projs.begin(), simpl_projs.end());
                    n_dist = std::min(n_dist, c_dist);
                }
            }
        }
        return res;
    }

    template <typename Simplex_t, typename Point_t, typename T>
    T Complex<Simplex_t, Point_t, T>::distance(Point_t point)
    {
        T res = std::numeric_limits<T>::max();
        for (size_t i = 0; i < this->simplexes.size(); i++)
        {
            for (size_t j = 0; j < this->simplexes[i].size(); j++)
            {
                T simpl_dist = this->simplexes[i][j].distance(point);
                res = std::min(res, simpl_dist);
            }
        }
        return res;
    }

    template <typename Simplex_t, typename T>
    ComplexFromMatrix<Simplex_t, T>::ComplexFromMatrix(const py::array_t<T> &A)
    {
        py::buffer_info A_arr = A.request();
        T *_dist_ptr = static_cast<T *>(A_arr.ptr);
        N = A_arr.shape[0];
        M = A_arr.shape[1];
        dist_ptr = new T[N * M];
        for (size_t i = 0; i < N * M; i++)
        {
            dist_ptr[i] = _dist_ptr[i];
        }
    }

    template <typename Simplex_t, typename T>
    ComplexFromMatrix<Simplex_t, T>::ComplexFromMatrix(ComplexFromMatrix<Simplex_t, T> &&other) : Complex<Simplex_t, size_t, T>::Complex(std::move(other))
    {
        py::print("Matrix move");
        N = other.N;
        M = other.M;
        dist_ptr = other.dist_ptr;
        other.dist_ptr = nullptr;
    }

    template <typename Simplex_t, typename T>
    ComplexFromMatrix<Simplex_t, T> &ComplexFromMatrix<Simplex_t, T>::operator=(ComplexFromMatrix<Simplex_t, T> &&other)
    {
        if (this == &other)
            return *this;
        Complex<Simplex_t, size_t, T>::operator=(std::move(other));
        N = other.N;
        M = other.M;
        if (dist_ptr != nullptr)
            delete[] dist_ptr;
        dist_ptr = other.dist_ptr;
        other.dist_ptr = nullptr;
        return *this;
    }

    template <typename Simplex_t, typename T>
    ComplexFromMatrix<Simplex_t, T>::ComplexFromMatrix(const ComplexFromMatrix<Simplex_t, T> &other) : Complex<Simplex_t, size_t, T>::Complex(other)
    {
        N = other.N;
        M = other.M;
        dist_ptr = new T[N * M];
        for (size_t i = 0; i < N * M; i++)
        {
            dist_ptr[i] = other.dist_ptr[i];
        }
    }

    template <typename Simplex_t, typename T>
    ComplexFromMatrix<Simplex_t, T> &ComplexFromMatrix<Simplex_t, T>::operator=(const ComplexFromMatrix<Simplex_t, T> &other)
    {
        if (this == &other)
            return *this;
        Complex<Simplex_t, size_t, T>::operator=(other);
        N = other.N;
        M = other.M;
        if (dist_ptr != nullptr)
            delete[] dist_ptr;
        dist_ptr = new T[N * M];
        for (size_t i = 0; i < N * M; i++)
        {
            dist_ptr[i] = other.dist_ptr[i];
        }
        return *this;
    }

    template <typename Simplex_t, typename T>
    ComplexFromMatrix<Simplex_t, T>::~ComplexFromMatrix()
    {
        if (dist_ptr != nullptr)
        {
            delete[] dist_ptr;
        }
    }

}