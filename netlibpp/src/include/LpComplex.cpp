#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <semaphore>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "Complex_t.cpp"
#include "Simplex_t.cpp"
#include "Point_t.cpp"
#include "VRComplex.cpp"
#include "find_comb.cpp"

namespace hypergraph
{

    template <template <typename, typename> typename Derived, typename T>
    struct LpComplexFromMatrix : public Derived<Simplex<PointIndex<T>, T>, T>
    {
    private:
        template <typename T_>
        T d(T_ &simplex_ptr, std::vector<int> &perm, T &min_dist, double &p)
        {
            Subsequences subs = Subsequences(perm.size(), 2);
            T ds = std::numeric_limits<T>::min();
            while (subs.next())
            {
                T norm_sqr = 0.0;
                const std::vector<int> &item = subs.get_subseq();
                for (std::size_t i = 0; i < item.size() - 1; i++)
                {
                    py::print(simplex_ptr[perm[item[i]]], simplex_ptr[perm[item[i + 1]]]);
                    norm_sqr = std::max(norm_sqr, this->lp_dist_idx(simplex_ptr[perm[item[i]]], simplex_ptr[perm[item[i + 1]]], p));
                }
                ds = std::max(ds, norm_sqr);
            }
            return ds;
        }

        void f_single_thread_(std::vector<int> simplex, T min_dist, double p)
        {
            pybind11::print(simplex, min_dist, p);
            std::vector<int> perm(simplex.size());
            T fs = std::numeric_limits<T>::max();

            for (std::size_t i = 0; i < perm.size(); i++)
                perm[i] = i;
            do
            {
                fs = std::min(fs, d(simplex, perm, min_dist, p));
            } while (std::next_permutation(perm.begin(), perm.end()));

            if (fs < min_dist)
            {
                std::vector<PointIndex<T>> app_simplex;
                app_simplex.reserve(simplex.size());
                for (size_t i = 0; i < simplex.size(); i++)
                {
                    app_simplex.push_back(simplex[i]);
                }
                this->append(app_simplex); // data race
            }
        }

        void f_multithread_part_(
            T min_dist, double p, std::vector<int> &beg_comb,
            long long start_offset, long long tasks, std::binary_semaphore &smphSignalThreadToMain, std::counting_semaphore<MAX_SEM_VAL> &free_sem)
        {
            Combinations comb(this->N, beg_comb.size(), beg_comb);
            smphSignalThreadToMain.release();
            long long i = 0;
            do
            {
                const std::vector<int> &simplex = comb.get_comb();
                f_single_thread_(simplex, min_dist, p);
            } while (comb.next() && i < tasks);
            free_sem.release();
        }

    public:
        LpComplexFromMatrix(const py::array_t<T> &A, T min_dist, double p, size_t max_dim_) : Derived<Simplex<PointIndex<T>, T>, T>(A)
        {
            py::buffer_info A_arr = A.request();
            int A_sz = A_arr.shape[0];
            T *A_ptr = static_cast<T *>(A_arr.ptr);
            for (size_t simplex_sz = 2; simplex_sz <= max_dim_; simplex_sz++)
            {
                Combinations comb(A_sz, simplex_sz);
                long long i = 0;
                ////
                int num_threads = 1;
                ////
                if (num_threads == 1)
                if(true)
                {
                    do
                    {
                        const std::vector<int> &simplex = comb.get_comb();
                        f_single_thread_(simplex, min_dist, p);
                    } while (comb.next());
                }
                else
                {
                    std::binary_semaphore smphSignalThreadToMain{0};
                    std::counting_semaphore<MAX_SEM_VAL> free_sem{0};
                    int64_t total_comb;
                    compute_total_comb(A_sz, simplex_sz, total_comb);
                    long long tasks = total_comb / num_threads;
                    if (total_comb % tasks > 0)
                    {
                        num_threads++;
                    }

                    for (long long i = 0; i < num_threads; i++)
                    {
                        std::vector<int> curr_comb(simplex_sz);
                        find_comb(A_sz, simplex_sz, tasks * i, curr_comb);
                        std::thread thr(
                            &LpComplexFromMatrix::f_multithread_part_, this, min_dist, p, std::ref(curr_comb), i * tasks,
                            tasks, std::ref(smphSignalThreadToMain), std::ref(free_sem));
                        thr.detach();
                        smphSignalThreadToMain.acquire();
                    }

                    for (int i = 0; i < num_threads; i++)
                    {
                        free_sem.acquire();
                    }
                }
            }
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

        LpComplexFromMatrix(const LpComplexFromMatrix &other) : Derived<Simplex<PointIndex<T>, T>, T>(other) {}
        LpComplexFromMatrix(const LpComplexFromMatrix &&other) : Derived<Simplex<PointIndex<T>, T>, T>(std::move(other)) {}
        LpComplexFromMatrix &operator=(const LpComplexFromMatrix &other)
        {
            Derived<Simplex<PointIndex<T>, T>, T>::operator=(other);
            return *this;
        }
        LpComplexFromMatrix &operator=(const LpComplexFromMatrix &&other)
        {
            Derived<Simplex<PointIndex<T>, T>, T>::operator=(std::move(other));
            return *this;
        }
    };

}