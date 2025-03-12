#pragma once

#include <vector>
#include <limits>

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
struct VRComplexFromMatrix : public Derived<Simplex<PointIndex<T>, T>, T>
{
    T volume_of(Simplex<PointIndex<T>, T> simplex)
    {
        return simplex.get_volume(this->dist);
    }

    void add_cofaces
    (
        const std::vector<std::vector<size_t>>& N_lower, std::vector<PointIndex<T>> tau,
        std::vector<size_t> clique_vertexes, size_t next, size_t max_dim
    )
    {
        tau.push_back(PointIndex<T>(next));
        this->append(Simplex<PointIndex<T>, T>(tau));
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

    VRComplexFromMatrix(const py::array_t<T>& A, T min_dist, size_t max_dim_) : Derived<Simplex<PointIndex<T>, T>, T>(A)
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

        for (size_t i = 0; i < this->N; i++)
        {
            std::vector<PointIndex<T>> tau = std::vector<PointIndex<T>>(0);
            add_cofaces(N_lower, tau, N_lower[i], i, max_dim_);
        }
        py::print(this->N, this->M);
    }

    py::list as_list()
    {
        std::vector<std::vector<std::vector<size_t>>> indexes;
        for (size_t i = 0; i < this->simplexes.size(); i++)
        {
            indexes.push_back(std::vector<std::vector<size_t>>(0));
            for (size_t j = 0; j < this->simplexes[i].size(); j++)
            {
                indexes[i].push_back(std::vector<size_t>(0));
                std::vector<PointIndex<T>> vec = this->simplexes[i][j];
                for (size_t k = 0; k < vec.size(); k++)
                {
                    indexes[i][j].push_back(vec[k].index);
                }
            }
        }
        return py::cast(indexes);
        // return py::cast(Complex<Simplex<PointIndex<T>, T>, PointIndex<T>, T>::simplexes);
    }

    VRComplexFromMatrix(const VRComplexFromMatrix& other) : Derived<Simplex<PointIndex<T>, T>, T>(other) {}
    VRComplexFromMatrix(const VRComplexFromMatrix&& other) : Derived<Simplex<PointIndex<T>, T>, T>(std::move(other)) {}
    VRComplexFromMatrix& operator=(const VRComplexFromMatrix& other)
    {
        Derived<Simplex<PointIndex<T>, T>, T>::operator=(other);
        return *this;
    }
    VRComplexFromMatrix& operator=(const VRComplexFromMatrix&& other)
    {
        Derived<Simplex<PointIndex<T>, T>, T>::operator=(std::move(other));
        return *this;
    }

};

}