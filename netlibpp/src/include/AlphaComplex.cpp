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

        T __squared_norm(const size_t &i)
        {
            T sum = 0.0;
                for (size_t k = 0; k < this->M; k++)
                {
                    T diff = this->dist_ptr[i * this->M + k];
                    sum += diff * diff;
                }
            return sum;
        }

        T __pt_distnce(const Point<T> &pt, const size_t &i)
        {
            T dist = 0;
            for (size_t k = 0; k < this->M; k++)
            {
                T diff = this->dist_ptr[i * this->M + k] - pt[k];
                dist += diff * diff;
            }
            return sqrt(dist);
        }

        T circumscribed_sphere_radius_nd(const std::vector<size_t> &points)
        {
            const int d = this->M;
            assert(points.size() == d + 1);

            // Construct matrix A and vector b
            std::vector<std::vector<T>> A(d, std::vector<T>(d));
            std::vector<T> b(d);

            const size_t p0 = points[0];
            for (int i = 0; i < d; ++i)
            {
                const size_t pi = points[i + 1];
                for (int j = 0; j < d; ++j)
                {
                    A[i][j] = this->dist_ptr[pi * this->M + j] - this->dist_ptr[p0 * this->M + j];
                }
                b[i] = 0.5 * (__squared_norm(pi) - __squared_norm(p0));
            }

            // Solve A * c = b (Gaussian elimination)
            std::vector<T> c(d);
            for (int col = 0; col < d; ++col)
            {
                // Partial pivoting
                int max_row = col;
                for (int row = col + 1; row < d; ++row)
                {
                    if (std::abs(A[row][col]) > std::abs(A[max_row][col]))
                    {
                        max_row = row;
                    }
                }
                std::swap(A[col], A[max_row]);
                std::swap(b[col], b[max_row]);

                // Eliminate column
                for (int row = col + 1; row < d; ++row)
                {
                    T factor = A[row][col] / A[col][col];
                    for (int k = col; k < d; ++k)
                    {
                        A[row][k] -= factor * A[col][k];
                    }
                    b[row] -= factor * b[col];
                }
            }

            // Back substitution
            for (int row = d - 1; row >= 0; --row)
            {
                c[row] = b[row];
                for (int col = row + 1; col < d; ++col)
                {
                    c[row] -= A[row][col] * c[col];
                }
                c[row] /= A[row][row];
            }

            // Compute radius
            Point<T> center(d);
            for (int i = 0; i < d; ++i)
                center[i] = c[i] + this->dist_ptr[p0 * this->M + i];
            T radius = __pt_distnce(center, p0);

            return radius;
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
                py::print("cnt sphere");
                // return 0;
                return circumscribed_sphere_radius_nd(support);
            }
            // For d+1 points, solve the circumscribed sphere (using linear algebra)
            // (Implementation depends on solving a linear system)
            // Placeholder: Fallback to Ritter's approximation if not implemented
        }

        void process_facet(std::vector<std::set<Simplex<size_t, T, PT>>> &simplexes_, const std::vector<size_t> &simplex)
        {
            if (simplex.size() == 2)
                return;

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
                process_facet(simplexes_, points_);
                simplexes_[sz - 1].insert(Simplex<size_t, T, PT>(std::move(points_), static_cast<mtr::Matrix<T> *>(this)));
            }
        }

    public:
        AlphaComplexFromMatrix(const py::array_t<T> &A, T min_dist, size_t max_dim_) : Derived<Simplex<size_t, T, PT>, T>(A)
        {
            py::buffer_info A_arr = A.request();
            T *A_ptr = static_cast<T *>(A_arr.ptr);

            qhT qh_qh;
            qhT *qh = &qh_qh;
            qh_zero(qh, stderr);
            const char *flags = "qhull d Qbb Qc Qt";
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
                        simplex.push_back(static_cast<size_t>(vertex->id - 1));
                    }
                    std::reverse(simplex.begin(), simplex.end());
                    this->append(Simplex<size_t, T, PT>(simplex));
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