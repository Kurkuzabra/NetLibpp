#pragma once

#include <vector>
#include <limits>
#include <stdexcept>
#include <functional>

#ifndef PYBIND11_INCLUDE
#define PYBIND11_INCLUDE
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#endif

namespace py = pybind11;

namespace mtr
{

    template <typename T>
    struct Matrix
    {
        T *dist_ptr;
        size_t N;
        size_t M;

        inline T *from_idx(const size_t &i)
        {
            return dist_ptr + i * N;
        }

        py::array_t<T> to_py_array()
        {
            T *dist_ptr_ = dist_ptr;
            dist_ptr = nullptr;
            return py::array_t<T>(
                {N, M},
                {M * sizeof(T), sizeof(T)},
                dist_ptr_);
        }

        Matrix<T> operator+(const Matrix &) const;
        Matrix<T> operator*(const Matrix &) const;

        T& operator[](const size_t& i)
        {
            return dist_ptr[i];
        }
        const T& operator[](const size_t& i) const
        {
            return dist_ptr[i];
        }

        Matrix() : dist_ptr(nullptr), N(0), M(0) {}
        Matrix(size_t _N, size_t _M) : dist_ptr(new T[_N * _M]), N(_N), M(_M) {}
        Matrix(size_t _N, size_t _M, T *_dist_ptr) : dist_ptr(_dist_ptr), N(_N), M(_M) {}
        Matrix(const py::array_t<T> &A);
        Matrix(const Matrix &other);
        Matrix(Matrix &&other);
        Matrix &operator=(const Matrix &other);
        Matrix &operator=(Matrix &&other);
        ~Matrix();
    };

    template <typename T>
    Matrix<T> AAT(const Matrix<T> &A)
    {
        T *res = new T[A.N * A.N];
        for (size_t i = 0; i < A.N; i++)
        {
            for (size_t j = 0; j < A.N; j++)
            {
                T sum = 0.0;
                for (size_t k = 0; k < A.M; k++)
                {
                    sum += A.dist_ptr[i + k * A.N] * A.dist_ptr[j + k * A.N];
                }
                res[i * A.N + j] = sum;
            }
        }
        return Matrix<T>(A.N, A.N, res);
    }

    template <typename T>
    Matrix<T> ATA(const Matrix<T> &A)
    {
        T *res = new T[A.M * A.M];
        for (size_t i = 0; i < A.M; i++)
        {
            for (size_t j = 0; j < A.M; j++)
            {
                T sum = 0.0;
                for (size_t k = 0; k < A.N; k++)
                {
                    sum += A.dist_ptr[i * A.N + k] * A.dist_ptr[j * A.N + k];
                }
                res[i * A.M + j] = sum;
            }
        }
        return Matrix<T>(A.M, A.M, res);
    }

    template <typename T>
    Matrix<T> ABAT(const Matrix<T> &A, const Matrix<T> &B)
    {
        T *res = new T[A.N * A.N];
        for (size_t i = 0; i < A.N; i++)
        {
            for (size_t j = 0; j < A.N; j++)
            {
                T sum = 0;
                for (size_t k = 0; k < A.M; k++)
                {
                    for (size_t l = 0; l < A.M; l++)
                    {
                        sum += A[i * A.M + k] * B[k * A.M + l] * A[j * A.M + l];
                    }
                }
                res[i * A.N + j] = sum;
            }
        }
        return Matrix<T>(A.N, A.N, res);
    }

    template <typename T>
    Matrix<T> ATBA(const Matrix<T> &A, const Matrix<T> &B)
    {
        T *res = new T[A.M * A.M];
        for (size_t i = 0; i < A.M; i++)
        {
            for (size_t j = 0; j < A.M; j++)
            {
                T sum = 0.0;
                for (size_t k = 0; k < A.N; k++)
                {
                    for (size_t l = 0; l < A.N; l++)
                    {
                        sum += A[i * A.N + k] * B[k * A.N + l] * A[j * A.N + l];
                    }
                }
                res[i * A.M + j] = sum;
            }
        }
        return Matrix<T>(A.M, A.M, res);
    }

    template <typename T>
    Matrix<T> AB_diag_AT(const Matrix<T> &A, const Matrix<T> &B)
    {
        T *res = new T[A.N * A.N];
        for (size_t i = 0; i < A.N; i++)
        {
            for (size_t j = 0; j < A.N; j++)
            {
                T sum = 0;
                for (size_t k = 0; k < A.M; k++)
                {
                    sum += A[i * A.M + k] * B[k * A.M + k] * A[j * A.M + k];
                }
                res[i * A.N + j] = sum;
            }
        }
        return Matrix<T>(A.N, A.N, res);
    }

    template <typename T>
    Matrix<T> ATB_diag_A(const Matrix<T> &A, const Matrix<T> &B)
    {
        T *res = new T[A.M * A.M];
        for (size_t i = 0; i < A.M; i++)
        {
            for (size_t j = 0; j < A.M; j++)
            {
                T sum = 0.0;
                for (size_t k = 0; k < A.N; k++)
                {
                    sum += A[i * A.N + k] * B[k * A.N + k] * A[j * A.N + k];
                }
                res[i * A.M + j] = sum;
            }
        }
        return Matrix<T>(A.M, A.M, res);
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator+(const Matrix &mx) const
    {
        T *res = new T[N * M];
        for (size_t i = 0; i < N * M; i++)
        {
            res[i] = dist_ptr[i] + mx.dist_ptr[i];
        }
        return Matrix<T>(N, M, res);
    }

    template <typename T>
    Matrix<T> Matrix<T>::operator*(const Matrix &mx) const
    {
        T *res = new T[N * mx.M];
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < mx.M; j++)
            {
                T sum = 0;
                for (size_t k = 0; k < M; k++)
                {
                    sum += dist_ptr[i * M + k] * mx.dist_ptr[k * mx.M + j];
                }
                res[i * mx.M + j] = sum;
            }
        }
        return Matrix<T>(N, mx.M, res);
    }

    template <typename T>
    Matrix<T>::Matrix(const py::array_t<T> &A)
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

    template <typename T>
    Matrix<T>::Matrix(Matrix<T> &&other)
    {
        N = other.N;
        M = other.M;
        dist_ptr = other.dist_ptr;
        other.dist_ptr = nullptr;
    }

    template <typename T>
    Matrix<T> &Matrix<T>::operator=(Matrix<T> &&other)
    {
        if (this == &other)
            return *this;
        N = other.N;
        M = other.M;
        if (dist_ptr != nullptr)
            delete[] dist_ptr;
        dist_ptr = other.dist_ptr;
        other.dist_ptr = nullptr;
        return *this;
    }

    template <typename T>
    Matrix<T>::Matrix(const Matrix<T> &other)
    {
        N = other.N;
        M = other.M;
        dist_ptr = new T[N * M];
        for (size_t i = 0; i < N * M; i++)
        {
            dist_ptr[i] = other.dist_ptr[i];
        }
    }

    template <typename T>
    Matrix<T> &Matrix<T>::operator=(const Matrix<T> &other)
    {
        if (this == &other)
            return *this;
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

    template <typename T>
    Matrix<T>::~Matrix()
    {
        if (dist_ptr != nullptr)
        {
            delete[] dist_ptr;
        }
    }

}