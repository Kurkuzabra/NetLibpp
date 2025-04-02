#pragma once

#include <vector>
#include <cmath>
#include <functional>
#include <exception>

namespace hypergraph
{

template<typename T>
struct Point
{
    std::vector<T> coordinates;
    Point() : coordinates(std::vector<T>(0)) {}
    Point(const std::vector<T>& coordinates_) : coordinates(coordinates_) {}
    Point(size_t N) : coordinates(std::vector<T>(N)) {}
    Point(size_t N, T* c_arr) : coordinates(std::vector<T>(N))
    {
        for (size_t i = 0; i < N; i++)
        {
            coordinates[i] = c_arr[i];
        }
    }
    T distance(const Point<T>& point) const
    {
        T dist = 0;
        for (size_t i = 0; i < coordinates.size(); i++)
        {
            dist += std::pow((coordinates[i] - point.coordinates[i]), 2.0);
        }
        return std::sqrt(dist);
    }
    T distance(const T* point) const
    {
        T dist = 0;
        for (size_t i = 0; i < coordinates.size(); i++)
        {
            dist += std::pow((coordinates[i] - point[i]), 2.0);
        }
        return std::sqrt(dist);
    }
    inline T& operator[](const size_t& i)
    {
        return coordinates[i];
    }
    inline const T& operator[](const size_t& i) const
    {
        return coordinates[i];
    }
    inline size_t size()
    {
        return coordinates.size();
    }
    explicit operator std::vector<T>&()
    {
        return coordinates;
    }
    std::vector<T> coords()
    {
        return coordinates;
    }
    bool operator ==(const Point& pt) const
    {
        return coordinates == pt.coordinates;
    }
};

template<typename T>
struct PointIndex
{
    size_t index;

    PointIndex() {}
    PointIndex(const size_t& index_) : index(index_) {}
    inline const size_t& get_index() const
    {
        return index;
    }
    operator size_t()
    {
        return index;
    }
    
    inline size_t size()
    {
        throw std::logic_error("this type of point has no size() method");
    }
    inline T& operator[](const size_t& i)
    {
        throw std::logic_error("this type of point has no [size_t] method");
    }
    inline const T& operator[](const size_t& i) const
    {
        throw std::logic_error("this type of point has no [size_t] method");
    }
    inline T distance(const PointIndex<T>& point) const
    {
        throw std::logic_error("this type of point has no distance method");
    }
    T distance(const T* point) const
    {
        throw std::logic_error("this type of point has no distance method");
    }
    inline std::vector<T> coords()
    {
        throw std::logic_error("this type of point has no coords method");
    }
};


}
