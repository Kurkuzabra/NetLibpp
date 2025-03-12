#pragma once

#include <cinttypes>
#include <cmath>

template <typename T>
T norm(T *x, std::size_t length)
{
    T sum = 0;
    for (std::size_t i = 0; i < length; i++)
    {
        sum += x[i] * x[i];
    }

    return sqrt(sum);
}

template <typename T>
void scalar_div(T *x, T r, std::size_t length, T *y)
{
    for (std::size_t i = 0; i < length; i++)
    {
        y[i] = x[i] / r;
    }
}

template <typename T>
void scalar_mul(T *x, T r, std::size_t length, T *y)
{
    for (std::size_t i = 0; i < length; i++)
    {
        y[i] = x[i] * r;
    }
}

template <typename T>
void scalar_sub(T *x, T r, std::size_t length, T *y)
{
    for (std::size_t i = 0; i < length; i++)
    {
        y[i] -= r * x[i];
    }
}

template <typename T>
T dot_product(T *x, T *y, std::size_t length)
{
    T sum = 0;
    for (std::size_t i = 0; i < length; i++)
    {
        sum += x[i] * y[i];
    }
    return sum;
}

template <typename T>
void write_mul(T* m1, T* m2, T* res, size_t N, size_t K, size_t M)
{
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < M; j++)
        {
            for (size_t k = 0; k < K; k++)
            {
                res[i * M + j] += m1[i * K + k] * m2[k * M + j];
            }
        }
    }
}

template <typename T>
T determinant(T *matrix, size_t N)
{
    if (N == 1)
    {
        return matrix[0];
    }
    if (N == 2)
    {
        return (matrix[0] * matrix[3]) - (matrix[1] * matrix[2]);
    }

    T *m = new T[(N - 1) * (N - 1)];
    T det = 0.0f;
    for (size_t j1 = 0; j1 < N; j1++)
    {
        for (size_t i = 1; i < N; i++)
        {
            size_t j2 = 0;
            for (size_t j = 0; j < N; j++)
            {
                if (j == j1)
                {
                    continue;
                }
                m[(i - 1) * (N - 1) + j2] = matrix[i * N + j];
                j2++;
            }
        }
        T d = determinant(m, N - 1);
        if (j1 & 1)
        {
            d = -d;
        }
        det += matrix[j1] * d;
    }
    delete[] m;
    return det;
}

template<typename T>
void transpose(T* matrix, T* tr, size_t N, size_t M)
{
    for (size_t i = 0; i < M; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            tr[i * N + j] = matrix[j * M + i];
        }
    }
}


template<typename T>
void cofactor(T* matrix, T* cof, size_t p, size_t q, size_t N)
{
    size_t j = 0;
    for (size_t row = 0; row < N; row++)
    {
        for (size_t col = 0; col < N; col++)
        {
            if (row != p && col != q)
            {
                cof[j] = matrix[row * N + col];
                j++;
            }
        }
    }
}

template<typename T>
void adjoint(T* matrix, T* adj, size_t N)
{
    if (N == 1)
    {
        adj[0] = 1;
        return;
    }
    
    T sign = 1;
    T* cof = new T[(N - 1) * (N - 1)];
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            cofactor(matrix, cof, i, j, N);
            sign = ((i + j) % 2 == 0) ? 1 : -1;
            adj[j * N + i] = sign * determinant(cof, N - 1);
        }
    }
    delete[] cof;
}

template<typename T>
void inverse(T* matrix, T* inv, size_t N)
{
    T det = determinant(matrix, N);
    if (det == 0) {}

    T* adj = new T[N * N];
    adjoint(matrix, adj, N);

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            inv[i * N + j] = adj[i * N + j] / T(det);
        }
    }
    delete[] adj;
}