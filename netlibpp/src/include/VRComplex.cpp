#pragma once

#include <vector>
#include <limits>
#include <omp.h>


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "Complex_t.cpp"
#include "Simplex_t.cpp"
#include "Point_t.cpp"

#define EPSILON 0.000001
#define MAX_SEM_VAL 100000

namespace py = pybind11;

namespace hypergraph
{

template <template<typename, typename> typename Derived, typename T>
struct VRComplexFromMatrix : public Derived<Simplex<size_t, T>, T>
{

    void add_cofaces
    (
        const std::vector<std::vector<size_t>>& N_lower, std::vector<size_t> tau,
        std::vector<size_t> clique_vertexes, size_t next, size_t max_dim
    )
    {
        tau.push_back(size_t(next));
        this->append(Simplex<size_t, T>(tau));
        if (tau.size() >= max_dim) return;
        for (size_t i = 0; i < clique_vertexes.size(); i++)
        {
            std::vector<size_t> updated_clique_vertexes(0);
            size_t next_vert_iter = 0;
            const std::vector<size_t>& N_lower_i = N_lower[clique_vertexes[i]];
            for (size_t j = 0; j < clique_vertexes.size(); j++)
            {
                while (next_vert_iter < N_lower_i.size() && N_lower_i[next_vert_iter] < clique_vertexes[j])
                {
                    next_vert_iter++;
                }
                if (next_vert_iter < N_lower_i.size() && N_lower_i[next_vert_iter] == clique_vertexes[j])
                {
                    updated_clique_vertexes.push_back(clique_vertexes[j]);        
                }
            }
            add_cofaces(N_lower, tau, updated_clique_vertexes, clique_vertexes[i], max_dim);
        }
    }

    VRComplexFromMatrix(const py::array_t<T>& A, T min_dist, size_t max_dim_) : Derived<Simplex<size_t, T>, T>(A)
    {
        std::vector<std::vector<size_t>> N_lower(this->N, std::vector<size_t>(0));
        for (size_t i = 0; i < this->N; i++)
        {
            for (size_t j = 0; j < i; j++)
            {
                if (this->dist_idx(i, j) < min_dist)
                {
                    N_lower[i].push_back(j);
                }
            }
        }

        #pragma omp parallel for
        for (size_t i = 0; i < this->N; i++)
        {
            std::vector<size_t> tau = std::vector<size_t>(0);
            add_cofaces(N_lower, tau, N_lower[i], i, max_dim_);
        }
    }

    VRComplexFromMatrix(const VRComplexFromMatrix& other) : Derived<Simplex<size_t, T>, T>(other) {}
    VRComplexFromMatrix(const VRComplexFromMatrix&& other) : Derived<Simplex<size_t, T>, T>(std::move(other)) {}
    VRComplexFromMatrix& operator=(const VRComplexFromMatrix& other)
    {
        Derived<Simplex<size_t, T>, T>::operator=(other);
        return *this;
    }
    VRComplexFromMatrix& operator=(const VRComplexFromMatrix&& other)
    {
        Derived<Simplex<size_t, T>, T>::operator=(std::move(other));
        return *this;
    }

};

}