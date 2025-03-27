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

#define LP_VR_CMPLX_OPTIM

// for now multithreading here is not supported for win32

namespace hypergraph
{

    template <template <typename, typename> typename Derived, typename T>
    struct LpComplexFromMatrix : public Derived<Simplex<size_t, T>, T>
    {
    private:
        template <typename T_>
        T d(const T_ &simplex_ptr, const std::vector<int> &perm, const double& p)
        {
            Subsequences subs = Subsequences(perm.size(), 2);
            T ds = std::numeric_limits<T>::min();

            if (std::fabs(p - (double)1) < EPSILON)
            {
                while (subs.next())
                {
                    T norm_sqr = 0.0;
                    const std::vector<int> &item = subs.get_subseq();
                    for (std::size_t i = 0; i < item.size() - 1; i++)
                    {
                        norm_sqr += this->lp_dist_idx(simplex_ptr[perm[item[i]]], simplex_ptr[perm[item[i + 1]]], p);
                    }
                    ds = std::max(ds, norm_sqr);
                }
            }
            else if (p == std::numeric_limits<double>::infinity())
            {
                while (subs.next())
                {
                    T norm_sqr = std::numeric_limits<float>::min();
                    const std::vector<int> &item = subs.get_subseq();
                    for (std::size_t i = 0; i < item.size() - 1; i++)
                    {
                        norm_sqr = std::max(norm_sqr, this->lp_dist_idx(simplex_ptr[perm[item[i]]], simplex_ptr[perm[item[i + 1]]], p));
                    }
                    ds = std::max(ds, norm_sqr);
                }
            }
            else
            {
                while (subs.next())
                {
                    T norm_sqr = 0.0;
                    const std::vector<int> &item = subs.get_subseq();
                    for (std::size_t i = 0; i < item.size() - 1; i++)
                    {
                        norm_sqr += std::pow(std::fabs(this->lp_dist_idx(simplex_ptr[perm[item[i]]], simplex_ptr[perm[item[i + 1]]], p)), p);
                    }
                    ds = std::max(ds, static_cast<T>(std::pow(norm_sqr, 1.0 / p)));
                }
            }
            return ds;
        }

        template <bool actual_single_thread = false>
        void f_single_thread_(const std::vector<int>& simplex, const T& min_dist, const double& p)
        {
            std::vector<int> perm(simplex.size());
            T fs = std::numeric_limits<T>::max();

            for (std::size_t i = 0; i < perm.size(); i++)
                perm[i] = i;
            do
            {
                fs = std::min(fs, d(simplex, perm, p));
            } while (std::next_permutation(perm.begin(), perm.end()));

            if (fs < min_dist)
            {
                std::vector<size_t> app_simplex;
                app_simplex.reserve(simplex.size());
                for (size_t i = 0; i < simplex.size(); i++)
                {
                    app_simplex.push_back(simplex[i]);
                }
                if constexpr (actual_single_thread)
                {
                    this->append(app_simplex);
                }
                else
                {
                    this->safe_append(app_simplex);
                }
            }
        }

        #ifndef _WIN32
        void f_multithread_part_(
            T min_dist, double p, std::vector<int> &beg_comb,
            long long tasks, std::binary_semaphore &smphSignalThreadToMain, std::counting_semaphore<MAX_SEM_VAL> &free_sem)
        {
            Combinations comb(this->N, beg_comb.size(), beg_comb);
            smphSignalThreadToMain.release();
            long long i = 0;
            do
            {
                const std::vector<int> &simplex = comb.get_comb();
                f_single_thread_<false>(simplex, min_dist, p);
                i++;
            } while (comb.next() && i < tasks);
            free_sem.release();
        }
        #endif

    public:
        LpComplexFromMatrix(const py::array_t<T> &A, T min_dist, double p, size_t max_dim_) : Derived<Simplex<size_t, T>, T>(A)
        {
            int num_threads = std::thread::hardware_concurrency();

            std::vector<size_t> app_simplex(1);
            for (size_t i = 0; i < this->N; i++)
            {
                app_simplex[0] = i;
                this->append(app_simplex);
            }

            for (size_t simplex_sz = 2; simplex_sz <= max_dim_; simplex_sz++)
            {
                long long i = 0;
                int64_t total_comb;
                Combinations comb(this->N, simplex_sz);
                compute_total_comb(this->N, simplex_sz, total_comb);
                #ifndef _WIN32
                if (num_threads == 1 || total_comb < num_threads)
                #else
                if (true)
                #endif
                {

                    do
                    {
                        const std::vector<int> &simplex = comb.get_comb();
                        f_single_thread_<true>(simplex, min_dist, p);
                    } while (comb.next());
                }
                #ifndef _WIN32
                else
                {
                    std::binary_semaphore smphSignalThreadToMain{0};
                    std::counting_semaphore<MAX_SEM_VAL> free_sem{0};
                    long long tasks = total_comb / num_threads;
                    if (total_comb % tasks > 0)
                    {
                        num_threads++;
                    }
                    for (long long i = 0; i < num_threads; i++)
                    {
                        std::vector<int> curr_comb(simplex_sz);
                        find_comb(this->N, simplex_sz, tasks * i, curr_comb);
                        std::thread thr(
                            &LpComplexFromMatrix::f_multithread_part_, this, min_dist, p, std::ref(curr_comb),
                            tasks, std::ref(smphSignalThreadToMain), std::ref(free_sem));
                        thr.detach();
                        smphSignalThreadToMain.acquire();
                    }

                    for (int i = 0; i < num_threads; i++)
                    {
                        free_sem.acquire();
                    }
                }
                #endif
            }
        }
        
        LpComplexFromMatrix(const LpComplexFromMatrix &other) : Derived<Simplex<size_t, T>, T>(other) {}
        LpComplexFromMatrix(const LpComplexFromMatrix &&other) : Derived<Simplex<size_t, T>, T>(std::move(other)) {}
        LpComplexFromMatrix &operator=(const LpComplexFromMatrix &other)
        {
            Derived<Simplex<size_t, T>, T>::operator=(other);
            return *this;
        }
        LpComplexFromMatrix &operator=(const LpComplexFromMatrix &&other)
        {
            Derived<Simplex<size_t, T>, T>::operator=(std::move(other));
            return *this;
        }
    };

}

#undef LP_VR_CMPLX_OPTIM