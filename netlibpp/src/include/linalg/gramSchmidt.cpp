#pragma once

#include "linalg.cpp"

template <typename T>
void gramSchmidt (T *a, T *r, std::size_t n, std::size_t m)
{
    T anorm, tol = 10e-7;

    for(std::size_t i = 0; i < n; i++)
    {
        r[i * m + i] = norm(&(a[i * m]), m);

        if(r[i * m + i] > tol)
        {
            scalar_div(&(a[i * m]), r[i * m + i], m, &(a[i * m]));
        }
        else if(i == 0)
        {
            a[i * m + 0] = 1;
            for(std::size_t j = 1; j < m; j++)
            {
                a[i * m + j] = 0;
            }
        }
        else
        {
            for(std::size_t j = 0; j < m; j++)
            {
                a[i * m + j] = -a[0 * m + i] * a[0 * m + j];
            }
            a[i * m + i] += 1;

            for(std::size_t j = 1; j < i; j++)
            {
                scalar_sub(&(a[j * m]), a[j * m + i], m, &(a[i * m]));
            }

            anorm = norm(&(a[i * m]), m);
            scalar_div(&(a[i * m]), anorm, m, &(a[i * m]));
        }

        for(std::size_t j = i + 1; j < n; j++)
        {
            r[j * m + i] = dot_product(&(a[i * m]), &(a[j * m]), m);
            scalar_sub(&(a[i * m]), r[j * m + i], m, &(a[j * m]));
        }
    }
}