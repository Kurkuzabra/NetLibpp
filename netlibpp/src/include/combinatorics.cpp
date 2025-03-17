#pragma once

#include <iterator>
#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric>
#include <cstdint>
#include <cassert>

long long nChoosek(long long n, long long k)
{
    if (k > n) return 0;
    if (k * 2 > n) k = n-k;
    if (k == 0) return 1;

    int result = n;
    for( int i = 2; i <= k; ++i ) {
        result *= (n-i+1);
        result /= i;
    }
    return result;
}

//combinations n | k
class Combinations
{
    std::vector<int> _offset;
    int _global_offset;
    int _seq_size;

public:

    Combinations(int seq_size, int starting_offset)
    {
        _seq_size = seq_size;
        _global_offset = starting_offset;
        _offset.resize(_global_offset);
        for (int i = 0; i < _global_offset; i++)
        {
            _offset[i] = i;
        }
    }
    Combinations(int seq_size, int starting_offset, std::vector<int>& start)
    {
        _seq_size = seq_size;
        _global_offset = starting_offset;
        _offset.resize(_global_offset);
        for (int i = 0; i < _global_offset; i++)
        {
            _offset[i] = start[i];
        }
    }
    bool next()
    {
        _offset[_global_offset - 1]++;
        int i = _global_offset - 1;
        while (_offset[i] >= _seq_size - (_global_offset - i - 1) && i > 0)
        {
            _offset[i - 1]++;
            i--;
        }
        for (int j = i + 1; j < _global_offset; j++)
        {
            _offset[j] = _offset[j - 1] + 1;
        }
        if (_offset[_global_offset - 1] >= _seq_size)
        {
            return false;
        }
        return true;
    }
    const std::vector<int>& get_comb()
    {
        return _offset;
    }
};

// combinations n | k where k >= starting offset
class Subsequences
{
    std::vector<int> _offset;
    int _global_offset;
    int _seq_size;

    void clear_comb()
    {
        _offset.resize(_global_offset);
        for(int i = 0; i < _global_offset; i++)
        {
            _offset[i] = i;
        }
    }

public:

    Subsequences(int seq_size, int starting_offset)
    {
        _seq_size = seq_size;
        _global_offset = starting_offset;
        _offset.resize(_global_offset);
        for (int i = 0; i < _global_offset; i++)
        {
            _offset[i] = i;
        }
        _offset[_global_offset - 1]--;
    }
    bool next()
    {
        if (_global_offset > _seq_size)
        {
            return false;
        }
        _offset[_global_offset - 1]++;
        int i = _global_offset - 1;
        while (_offset[i] >= _seq_size && i > 0)
        {
            _offset[i - 1]++;
            _offset[i] -= _seq_size - 1;
            i--;
        }
        for (int j = i + 1; j < _global_offset; j++)
        {
            _offset[j] = _offset[j - 1] + 1;
        }
        if (_offset[_global_offset - 1] == _seq_size)
        {
            _global_offset++;
            if (_global_offset > _seq_size)
            {
                return false;
            }
            else
            {
                clear_comb();
            }
        }
        return true;
    }
    const std::vector<int>& get_subseq()
    {
        return _offset;
    }
};