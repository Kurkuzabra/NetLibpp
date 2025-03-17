#pragma once

#include <vector>
#include <optional>
#include <algorithm>
#include <cmath>
#include <functional>
#include <type_traits>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "quadprog/Array.hpp"
#include "quadprog/QuadProg++.cpp"
#include "linalg/gramSchmidt.cpp"
#include "linalg/linalg.cpp"

#define EPS 0.001

// namespace py = pybind11;

namespace detail
{
    size_t factorial(size_t n)
    {
        if (n == 0)
            return 1;
        else
            return n * factorial(n - 1);
    }

    template <template <typename> typename vector_t, typename T>
    inline T dot_product(vector_t<T> x, T *y)
    {
        T sum = 0;
        for (std::size_t i = 0; i < x.size(); i++)
        {
            sum += x[i] * y[i];
        }
        return sum;
    }

    template <template <typename> typename vector_t, typename T>
    inline T dot_product(const vector_t<T> &x, const vector_t<T> &y)
    {
        T res = 0.0f;
        for (size_t i = 0; i < x.size(); i++)
        {
            res += x[i] * y[i];
        }
        return res;
    }

    template <template <typename> typename vector_t, typename T>
    inline void sum_vect_arr(vector_t<T> &lph, T *rph)
    {
        for (size_t i = 0; i < lph.size(); i++)
        {
            lph[i] += rph[i];
        }
    }

    // 2 * A * A_T
    template <typename T>
    inline void write_double_transpose_mul(T *m1, T *res, size_t N, size_t M, size_t M_res)
    {
        for (size_t i = 0; i < M; i++)
        {
            for (size_t j = 0; j < M; j++)
            {
                res[i * M_res + j] = 0.0;
                for (size_t k = 0; k < N; k++)
                {
                    res[i * M_res + j] += 2.0 * m1[i * N + k] * m1[j * N + k];
                }
            }
        }
    }

    template <typename T>
    inline void ATA(const quadprogpp::Matrix<T> &A, quadprogpp::Matrix<T> &res)
    {
        for (size_t i = 0; i < A.ncols(); i++)
        {
            for (size_t j = 0; j < A.ncols(); j++)
            {
                res[i][j] = 0.0;
                for (size_t k = 0; k < A.nrows(); k++)
                {
                    res[i][j] += A[k][i] * A[k][j];
                }
            }
        }
    }

    template <typename T>
    inline void write_transpose_mul(T *m1, T *m2, T *res, size_t N, size_t K, size_t M)
    {
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < M; j++)
            {
                for (size_t k = 0; k < K; k++)
                {
                    res[i * M + j] += m1[k * N + i] * m2[k * M + j];
                }
            }
        }
    }

}

namespace hypergraph
{

    template <typename Point_t, typename T>
    struct Simplex
    {
    private:
        // Point_t projection_impl(Point_t point, std::vector<size_t> pts_ord, size_t start_offset);

    public:
        size_t dim; // dimension of a simplex
        std::vector<Point_t> points;
        std::optional<T> volume;
        std::optional<T> filter;

        Simplex(const std::vector<Point_t> &points_) : points(points_), dim(points_.size() - 1) {}
        Simplex(size_t dim_) : dim(dim_ - 1), points(std::vector<Point_t>(dim_)) {}
        Simplex() : dim(0), points(std::vector<Point_t>(0)) {}
        // Simplex() = delete;

        size_t get_dim() const
        {
            return dim;
        }
        size_t size() const
        {
            return points.size();
        }
        operator std::vector<Point_t>()
        {
            return points;
        }

        T get_volume(std::function<T(const Point_t &, const Point_t &)>);
        // get volume from distance matrix
        T get_volume();
        // get volume from vector coordinates

        std::vector<Point_t> projection_impl(const Point_t &, T &);
        std::vector<Point_t> projection(const Point_t &);
        // projection \pi_p of any point in R^d to a simplex (its convex hull)
        // projection points can be multiple (think of a point equidistant to all edges of a triangle!)
        // finding and returning projection point(s?)
        // is actually a quadratic programming problem min||Ax - y|| with constraints Ix - e >= 0 and e * x = 1,

        T distance(const Point_t &point);
        // distance of any point in R^d to a simplex (its convex hull)
        // computing the distance between a point and its projection to a simplex
    };

    template <typename Point_t, typename T>
    std::vector<Point_t> Simplex<Point_t, T>::projection_impl(const Point_t &point, T &d_nearest)
    {
        if constexpr (std::is_same<Point_t, size_t>::value)
        {
            throw std::logic_error("this type of point has no size() method");
        }
        else
        {
            if (dim < points[0].size())
            {
                size_t M = points.size();
                quadprogpp::Matrix<T> CE = quadprogpp::Matrix<T>(1.0, M, 1);
                quadprogpp::Matrix<T> CI = quadprogpp::Matrix<T>(0.0, M, M);
                quadprogpp::Matrix<T> G_ = quadprogpp::Matrix<T>(points[0].size(), M);
                quadprogpp::Matrix<T> G = quadprogpp::Matrix<T>(M, M);
                quadprogpp::Vector<T> g0(0.0, M);
                quadprogpp::Vector<T> ce0(-1.0, 1);
                quadprogpp::Vector<T> ci0(0.0, M);
                for (size_t i = 0; i < M; i++)
                {
                    for (size_t j = 0; j < points[i].size(); j++)
                    {
                        G_[j][i] = points[i][j] - point[j];
                    }
                }
                for (size_t i = 0; i < M; i++)
                {
                    CI[i][i] = 1.0;
                }
                detail::ATA(G_, G);
                quadprogpp::Vector<T> x_ = quadprogpp::Vector<T>(M);
                T result = quadprogpp::solve_quadprog(G, g0, CE, ce0, CI, ci0, x_);
                x_ = quadprogpp::dot_prod(G_, x_);
                Point_t x = Point_t(points[0].size());
                for (size_t i = 0; i < x_.size(); i++)
                {
                    x[i] = point[i] + x_[i];
                }
                if (result == std::numeric_limits<T>::infinity())
                    throw std::logic_error("Could not find a projection");
                d_nearest = point.distance(x);
                return std::vector<Point_t>(1, x);
            }
            else if (dim == points[0].size())
            {
                size_t M = points.size() - 1;
                quadprogpp::Matrix<T> CE = quadprogpp::Matrix<T>(1.0, M, 1);
                quadprogpp::Matrix<T> CI = quadprogpp::Matrix<T>(0.0, M, M);
                quadprogpp::Matrix<T> G_ = quadprogpp::Matrix<T>(points[0].size(), M);
                quadprogpp::Matrix<T> G = quadprogpp::Matrix<T>(M, M);
                quadprogpp::Vector<T> g0(0.0, M);
                quadprogpp::Vector<T> ce0(-1.0, 1);
                quadprogpp::Vector<T> ci0(0.0, M);
                quadprogpp::Vector<T> swap_vec(points[0].size());
                std::vector<Point_t> projections(0);
                for (size_t i = 0; i < M; i++)
                {
                    for (size_t j = 0; j < points[i].size(); j++)
                    {
                        G_[j][i] = points[i][j] - point[j];
                    }
                }
                for (size_t i = 0; i < M; i++)
                {
                    CI[i][i] = 1.0;
                }
                for (size_t i = 0; i < swap_vec.size(); i++)
                {
                    swap_vec[i] = points[M][i] - point[i];
                }
                for (size_t k = 0; k <= M; k++)
                {
                    std::cout << "cycle" << std::endl;

                    for (int i = 0; i < G_.nrows(); i++)
                    {
                        for (int j = 0; j < G_.ncols(); j++)
                        {
                            std::cout << std::fixed << G_[i][j] << " ";
                        }
                        std::cout << std::endl;
                    }
                    std::cout << std::endl;
                    // for (int i = 0; i < CI.nrows(); i++)
                    // {
                    //     for (int j = 0; j < CI.ncols(); j++)
                    //     {
                    //         std::cout << std::fixed << CI[i][j] << " ";
                    //     }
                    //     std::cout << std::endl;
                    // }
                    // std::cout << std::endl;

                    detail::ATA(G_, G);
                    quadprogpp::Vector<T> x_ = quadprogpp::Vector<T>(M);
                    quadprogpp::Vector<T> _g0(g0);
                    quadprogpp::Matrix<T> _G(G);
                    T result = quadprogpp::solve_quadprog(_G, _g0, CE, ce0, CI, ci0, x_);
                    x_ = quadprogpp::dot_prod(G_, x_);
                    Point_t x = Point_t(points[0].size());
                    for (size_t i = 0; i < x_.size(); i++)
                    {
                        x[i] = point[i] + x_[i];
                    }
                    if (result == std::numeric_limits<T>::infinity())
                        throw std::logic_error("Could not find a projection");

                    for (int i = 0; i < x.size(); i++)
                    {
                        std::cout << std::fixed << x[i] << " ";
                    }
                    std::cout << std::endl;

                    if (projections.size() == 0)
                    {
                        projections.push_back(x);
                        d_nearest = point.distance(x);
                    }
                    else
                    {
                        T d_x = point.distance(x);
                        if (d_x + EPS < d_nearest)
                        {
                            projections.resize(0);
                            projections.push_back(x);
                            d_nearest = d_x;
                        }
                        else if (d_x < d_nearest + EPS)
                        {
                            projections.push_back(x);
                            d_nearest = std::min(d_nearest, d_x);
                        }
                    }
                    if (k != M)
                    {
                        for (size_t i = 0; i < swap_vec.size(); i++)
                        {
                            std::swap(G_[i][k], swap_vec[i]);
                        }
                    }
                }
                return projections;
            }
            else
            {
                throw std::logic_error("Can only find projections to convex forms");
            }
        }
    }

    template <typename Point_t, typename T>
    T Simplex<Point_t, T>::get_volume(std::function<T(const Point_t &, const Point_t &)> distance)
    {
        if (volume.has_value())
        {
            return volume.value();
        }
        else
        {
            size_t mx_size = dim + 2;
            T *matrix = new T[(mx_size) * (mx_size)];
            for (size_t i = 0; i < dim + 1; i++)
            {
                for (size_t j = i + 1; j < dim + 1; j++)
                {
                    matrix[i * (mx_size) + j] = pow(distance(points[i], points[j]), 2.0);
                    matrix[j * (mx_size) + i] = matrix[i * (mx_size) + j];
                }
            }
            for (size_t i = 0; i < mx_size; i++)
                matrix[i * (mx_size) + i] = 0.0f;
            for (size_t i = 0; i < dim + 1; i++)
                matrix[i * (mx_size) + dim + 1] = 1.0f;
            for (size_t i = 0; i < dim + 1; i++)
                matrix[(dim + 1) * (mx_size) + i] = 1.0f;
            volume = std::sqrt(fabs(determinant(matrix, mx_size)) / (T)std::pow(detail::factorial(dim), 2) / (T)std::pow(2, dim));
            delete[] matrix;
            return volume.value();
        }
    }

    template <typename Point_t, typename T>
    T Simplex<Point_t, T>::get_volume()
    {
        if (volume.has_value())
        {
            return volume.value();
        }
        else
        {
            size_t mx_size = dim + 2;
            T *matrix = new T[(mx_size) * (mx_size)];
            for (size_t i = 0; i < dim + 1; i++)
            {
                for (size_t j = i + 1; j < dim + 1; j++)
                {
                    matrix[i * (mx_size) + j] = pow(points[i].distance(points[j]), 2.0);
                    matrix[j * (mx_size) + i] = matrix[i * (mx_size) + j];
                }
            }
            for (size_t i = 0; i < mx_size; i++)
                matrix[i * (mx_size) + i] = 0.0f;
            for (size_t i = 0; i < dim + 1; i++)
                matrix[i * (mx_size) + dim + 1] = 1.0f;
            for (size_t i = 0; i < dim + 1; i++)
                matrix[(dim + 1) * (mx_size) + i] = 1.0f;
            volume = std::sqrt(fabs(determinant(matrix, mx_size)) / (T)std::pow(detail::factorial(dim), 2) / (T)std::pow(2, dim));
            delete[] matrix;
            return volume.value();
        }
    }

    template <typename Point_t, typename T>
    std::vector<Point_t> Simplex<Point_t, T>::projection(const Point_t &point)
    {
        T dist_ = 0.0;
        return projection_impl(point, dist_);
    }

    template <typename Point_t, typename T>
    T Simplex<Point_t, T>::distance(const Point_t &point)
    {
        return point.distance(projection(point)[0]);
    }

}