#pragma once

#include <vector>
#include <stdexcept>
#include <memory>
#include <limits>
#include <algorithm>
#include <set>
#include <iterator>
#include <omp.h>

#include <libqhullcpp/Qhull.h>
#include <libqhullcpp/QhullFacet.h>
#include <libqhullcpp/QhullVertex.h>

extern "C"
{
#include <libqhull_r/qhull_ra.h>
#include <libqhull_r/qset_r.h>
#include <libqhull_r/libqhull_r.h>
}

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

    template <template <typename, typename> typename Derived, typename T, PointsType PT>
    struct DelaunayRipsComplexFromMatrix : public Derived<Simplex<size_t, T, PT>, T>
    {
        template <typename InputIt>
        void k_clique_append(const std::vector<std::set<size_t>> &complex_graph, const std::set<std::vector<size_t>>& d_simplexes, std::vector<size_t> &tau,
                             InputIt beg_vert, InputIt end_vert, const size_t &next)
        {
            if (tau.size() <= this->M)
            {
                this->safe_append(Simplex<size_t, T, PT>(tau));
            }
            else
            {
                if (d_simplexes.contains(tau))
                {
                    this->safe_append(Simplex<size_t, T, PT>(tau));
                }
                return;
            }
            for (InputIt it = beg_vert; it != end_vert; it++)
            {

                if (*it > next)
                {
                    tau.push_back(*it);
                    std::vector<size_t> v_intersection;
                    std::set_intersection(end_vert.base(), beg_vert.base(), complex_graph[*it].begin(), complex_graph[*it].end(),
                                          std::back_inserter(v_intersection));
                    k_clique_append(complex_graph, d_simplexes, tau, v_intersection.rbegin(), v_intersection.rend(), *it);
                    tau.pop_back();
                }
                else
                {
                    return;
                }
            }
        }

        DelaunayRipsComplexFromMatrix(const py::array_t<T> &A, T max_dist) : Derived<Simplex<size_t, T, PT>, T>(A)
        {
            py::buffer_info A_arr = A.request();
            T *A_ptr = static_cast<T *>(A_arr.ptr);

            qhT qh_qh;
            qhT *qh = &qh_qh;
            qh_zero(qh, stderr);
            const char *flags = "qhull d Qbb Qc Qt Qz";
            qh_new_qhull(qh, static_cast<int>(this->M),
                         static_cast<int>(this->N),
                         const_cast<T *>(A_ptr),
                         0, const_cast<char *>(flags), nullptr, stderr);
            std::vector<std::set<size_t>> complex_graph(this->N);
            std::set<std::vector<size_t>> d_simplexes;
            facetT *facet;
            for (facet = qh->facet_list; facet && facet->next; facet = facet->next)
            {
                if (!facet->upperdelaunay)
                {
                    std::vector<size_t> simplex;
                    vertexT *vertex, **vertexp;

                    FOREACHvertex_(facet->vertices)
                    {
                        int vertex_id = qh_pointid(qh, vertex->point);
                        simplex.push_back(static_cast<size_t>(vertex_id));
                    }
                    std::sort(simplex.begin(), simplex.end());
                    d_simplexes.insert(simplex);
                    for (size_t i = 0; i < simplex.size(); i++)
                    {
                        for (size_t j = i + 1; j < simplex.size(); j++)
                        {
                            if (this->dist_idx(simplex[i], simplex[j]) <= max_dist)
                            {
                                complex_graph[simplex[i]].insert(simplex[j]);
                                complex_graph[simplex[j]].insert(simplex[i]);
                            }
                        }
                    }
                }
            }
            #pragma omp parallel for
            #ifndef _WIN32
            for (size_t i = 0; i < this->N; i++)
            #else
            for (int i = 0; i < this->N; i++)
            #endif
            {
                std::vector<size_t> tau(1, i);
                k_clique_append(complex_graph, d_simplexes, tau, complex_graph[i].rbegin(), complex_graph[i].rend(), i);
            }
            qh_freeqhull(qh, !qh_ALL);
            int curlong, totlong;
            qh_memfreeshort(qh, &curlong, &totlong);
            if (curlong || totlong)
            {
                std::cerr << "Qhull memory leak: " << curlong << " " << totlong << std::endl;
            }
        }

        DelaunayRipsComplexFromMatrix(const DelaunayRipsComplexFromMatrix &other) : Derived<Simplex<size_t, T, PT>, T>(other) {}
        DelaunayRipsComplexFromMatrix(const DelaunayRipsComplexFromMatrix &&other) : Derived<Simplex<size_t, T, PT>, T>(std::move(other)) {}
        DelaunayRipsComplexFromMatrix &operator=(const DelaunayRipsComplexFromMatrix &other)
        {
            Derived<Simplex<size_t, T, PT>, T>::operator=(other);
            return *this;
        }
        DelaunayRipsComplexFromMatrix &operator=(const DelaunayRipsComplexFromMatrix &&other)
        {
            Derived<Simplex<size_t, T, PT>, T>::operator=(std::move(other));
            return *this;
        }
    };

}