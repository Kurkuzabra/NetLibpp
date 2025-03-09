#pragma once

#include "combinatorics.cpp"

template<typename int_type>
void compute_total_comb(int n, int k, int_type& result)
{
    if (k > n) 
    {
        result = 0;
        return;
    }
    if (k * 2 > n) k = n-k;
    if (k == 0)
    {
        result = 1;
        return;
    }

    result = n;
    for (int i = 2; i <= k; i++)
    {
        result *= (n - i + 1);
        result /= i;
    }
}

template<typename int_type>
bool find_comb(int n, int k, int_type index_to_find, std::vector<int>& results )
{
	if (k > n || n == 0 || k == 0)
    {
        return false;
    }
	int remaining_set = n - 1;  
	int remaining_comb = k - 1;  

	for (int i = 0; i < k; i++)
	{

		if (i == k - 1)
		{
			while (index_to_find != 0)
			{
                index_to_find -= 1;
				--remaining_set;
			}
            results[i] = n - remaining_set - 1;
		}
		else
		{
			int_type total_comb = 0;
			int_type prev = 0;

			int loop = remaining_set - remaining_comb;
			bool found = false;
			int i_prev = 0;

			if (i > 0)
            {
                i_prev = results[i - 1] + 1;
            }

			int j = 0;
			for (; j < loop; j++)
			{
				compute_total_comb(remaining_set, remaining_comb, total_comb);

				total_comb += prev;
				if (total_comb > index_to_find)
				{
					index_to_find -= prev;
					results[i] = j + i_prev;
					found = true;
					break;
				}
				prev = total_comb;
				--remaining_set;
			}

			if (!found)
			{
				index_to_find -= total_comb;
				results[i] = j + i_prev;
			}
			--remaining_set;
			--remaining_comb;
		}
	}

	return true;
};

int main()
{
    std::vector<int> t(3);
    for (int j = 0; j < 100; j++)
    {
        std::iota(t.begin(), t.end(), 0);
        if (find_comb(6, 3, j, t)) std::cout << "found\n";
        for (int i = 0; i < t.size(); i++)
        {
            std::cout << t[i] << " ";
        }
        std::cout << std::endl;
    }
}