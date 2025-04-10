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

    namespace util
    {
        template <typename T>
        struct Sphere
        {
            Point<T> center;
            T radius;
        };
    }

    template <template <typename, typename> typename Derived, typename T, PointsType PT>
    struct AlphaComplexFromMatrix : public Derived<Simplex<size_t, T, PT>, T>
    {
    private:
        T max_rad;

        T circumscribed_sphere_radius_nd(const std::vector<size_t> &support)
        {
            Simplex<size_t, T, PT> splx(support, static_cast<mtr::Matrix<T> *>(this));
            return splx.get_circumsphere_radius();
        }        

        // Base case: Minimal sphere for up to (n+1) points in n-D space
        T minimal_sphere_radius(const std::vector<size_t> &support)
        {
            if (support.empty() || support.size() == 1)
            {
                // Undefined case
                return 0;
            }
            else if (support.size() == 2)
            {
                // Sphere with diameter = distance between two points
                return this->dist_idx(support[0], support[1]) / 2.0;
            }
            else
            {
                // For 2 dims use formula
                // T a = this->dist_idx(support[0], support[1]);
                // T b = this->dist_idx(support[1], support[2]);
                // T c = this->dist_idx(support[2], support[0]);
                // return (a * b * c) / sqrt((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c));
                return circumscribed_sphere_radius_nd(support);
                // For d+1 points, solve the circumscribed sphere (using linear algebra)
            }
        }

        void process_facet(std::vector<std::set<Simplex<size_t, T, PT>>> &simplexes_,
                           const std::vector<size_t> &simplex, bool already_satisfy = false)
        {
            if (simplex.size() == 1)
                return;

            if (!already_satisfy)
            {
                already_satisfy = minimal_sphere_radius(simplex) <= max_rad;
            }
            if (already_satisfy)
            {
                simplexes_[simplex.size() - 1].insert(Simplex<size_t, T, PT>(simplex, static_cast<mtr::Matrix<T> *>(this)));
            }

            size_t sz = simplex.size();
            for (size_t i = 0; i < sz; i++)
            {
                std::vector<size_t> points_(sz - 1);
                size_t j_ = 0;
                for (size_t j = 0; j < sz - 1; j++)
                {
                    if (j_ == i)
                    {
                        j_++;
                    }
                    points_[j] = simplex[j_];
                    j_++;
                }
                process_facet(simplexes_, points_, already_satisfy);
            }
        }

    public:
        AlphaComplexFromMatrix(const py::array_t<T> &A, T _max_rad) : Derived<Simplex<size_t, T, PT>, T>(A), max_rad(_max_rad)
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
            std::vector<std::set<Simplex<size_t, T, PT>>> simplexes_(this->N);
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
                    process_facet(simplexes_, simplex);
                }
            }
            for (size_t i = 0; i < this->N; i++)
            {
                this->append(Simplex<size_t, T, PT>(std::vector<size_t>(1, i)));
            }
            for (size_t i = 1; i < simplexes_.size(); i++)
            {
                for (typename std::set<Simplex<size_t, T, PT>>::iterator it = simplexes_[i].begin(); it != simplexes_[i].end(); it++)
                {
                    this->append(const_cast<Simplex<size_t, T, PT> &>(*it));
                }
            }
            qh_freeqhull(qh, !qh_ALL);
            int curlong, totlong;
            qh_memfreeshort(qh, &curlong, &totlong);
            if (curlong || totlong)
            {
                std::cerr << "Qhull memory leak: " << curlong << " " << totlong << std::endl;
            }
        }

        AlphaComplexFromMatrix(const AlphaComplexFromMatrix &other) : Derived<Simplex<size_t, T, PT>, T>(other) {}
        AlphaComplexFromMatrix(const AlphaComplexFromMatrix &&other) : Derived<Simplex<size_t, T, PT>, T>(std::move(other)) {}
        AlphaComplexFromMatrix &operator=(const AlphaComplexFromMatrix &other)
        {
            Derived<Simplex<size_t, T, PT>, T>::operator=(other);
            return *this;
        }
        AlphaComplexFromMatrix &operator=(const AlphaComplexFromMatrix &&other)
        {
            Derived<Simplex<size_t, T, PT>, T>::operator=(std::move(other));
            return *this;
        }
    };

}