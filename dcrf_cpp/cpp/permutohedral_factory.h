
#include <stdbool.h>
#include <vector>
#include <iostream>
#include <string.h>

using namespace std;

class HashTable
{
protected:
    size_t key_size_, filled_, capacity_;
    std::vector<short> keys_;
    std::vector<int> table_;
    void grow()
    {
        // Create the new memory and copy the values in
        int old_capacity = capacity_;
        capacity_ *= 2;
        std::vector<short> old_keys((old_capacity + 10) * key_size_);
        std::copy(keys_.begin(), keys_.end(), old_keys.begin());
        std::vector<int> old_table(capacity_, -1);

        // Swap the memory
        table_.swap(old_table);
        keys_.swap(old_keys);

        // Reinsert each element
        for (int i = 0; i < old_capacity; i++)
            if (old_table[i] >= 0)
            {
                int e = old_table[i];
                size_t h = hash(getKey(e)) % capacity_;
                for (; table_[h] >= 0; h = h < capacity_ - 1 ? h + 1 : 0)
                    ;
                table_[h] = e;
            }
    }
    size_t hash(const short *k)
    {
        size_t r = 0;
        for (size_t i = 0; i < key_size_; i++)
        {
            r += k[i];
            r *= 1664525;
        }
        return r;
    }

public:
    explicit HashTable(int key_size, int n_elements) : key_size_(key_size), filled_(0), capacity_(2 * n_elements), keys_((capacity_ / 2 + 10) * key_size_), table_(2 * n_elements, -1)
    {
    }
    int size() const
    {
        return filled_;
    }
    void reset()
    {
        filled_ = 0;
        std::fill(table_.begin(), table_.end(), -1);
    }
    int find(const short *k, bool create = false)
    {
        if (2 * filled_ >= capacity_)
            grow();
        // Get the hash value
        size_t h = hash(k) % capacity_;
        // Find the element with he right key, using linear probing
        while (1)
        {
            int e = table_[h];
            if (e == -1)
            {
                if (create)
                {
                    // Insert a new key and return the new id
                    for (size_t i = 0; i < key_size_; i++)
                        keys_[filled_ * key_size_ + i] = k[i];
                    return table_[h] = filled_++;
                }
                else
                    return -1;
            }
            // Check if the current key is The One
            bool good = true;
            for (size_t i = 0; i < key_size_ && good; i++)
                if (keys_[e * key_size_ + i] != k[i])
                    good = false;
            if (good)
                return e;
            // Continue searching
            h++;
            if (h == capacity_)
                h = 0;
        }
    }
    const short *getKey(int i) const
    {
        return &keys_[i * key_size_];
    }
};

void cppInit(const float *features, const int N, const int d, int *os, float *ws, int *blur_neighborsT, int *M)
{
    // Compute the lattice coordinates for each feature [there is going to be a lot of magic here]
    HashTable hash_table(d, N * (d + 1));

    // Allocate local memories
    float *scale_factor = new float[d];
    float *elevated = new float[d + 1];
    float *rem0 = new float[d + 1];
    float *barycentric = new float[d + 2];
    short *rank = new short[d + 1];
    short *canonical = new short[(d + 1) * (d + 1)];
    short *key = new short[d + 1];

    // Compute the canonical simplex
    for (int i = 0; i < d + 1; i++)
    {
        for (int j = 0; j < d - i + 1; j++)
        {
            canonical[i * (d + 1) + j] = i;
        }
        for (int j = d - i + 1; j < d + 1; j++)
        {
            canonical[i * (d + 1) + j] = i - (d + 1);
        }
    }

    // Expected standard deviation of our filter (p.6 in [Adams et al., 2010])
    float inv_std_dev = sqrt(2. / 3.) * (d + 1);
    // Compute the diagonal part of E (p.5 in [Adams et al., 2010])
    for (int i = 0; i < d; i++)
    {
        scale_factor[i] = 1. / sqrt(double((i + 2) * (i + 1))) * inv_std_dev;
    }

    // Compute the simplex each feature lies in
    for (int n = 0; n < N; n++)
    {
        // Elevate the feature (y = Ep, see p.5 in [Adams et al., 2010])
        const float *f = &features[n * d];

        // sm contains the sum of 1...n of our feature vector
        float sm = 0;
        for (int j = d; j > 0; j--)
        {
            float cf = f[j - 1] * scale_factor[j - 1];
            elevated[j] = sm - j * cf;
            sm += cf;
        }
        elevated[0] = sm;

        // Find the closest 0-colored simplex through rounding
        float down_factor = 1.f / (d + 1);
        float up_factor = (d + 1);
        int sum = 0;
        for (int i = 0; i < d + 1; i++)
        {
            int rd2;
            float v = down_factor * elevated[i];
            float up = ceilf(v) * up_factor;
            float down = floorf(v) * up_factor;
            if (up - elevated[i] < elevated[i] - down)
            {
                rd2 = (short)up;
            }
            else
            {
                rd2 = (short)down;
            }

            rem0[i] = rd2;
            sum += rd2 * down_factor;
        }

        // Find the simplex we are in and store it in rank (where rank describes what position coordinate i has in the sorted order of the feature values)
        for (int i = 0; i < d + 1; i++)
        {
            rank[i] = 0;
        }
        for (int i = 0; i < d; i++)
        {
            double di = elevated[i] - rem0[i];
            for (int j = i + 1; j < d + 1; j++)
            {
                if (di < elevated[j] - rem0[j])
                {
                    rank[i]++;
                }
                else
                {
                    rank[j]++;
                }
            }
        }

        // If the point doesn't lie on the plane (sum != 0) bring it back
        for (int i = 0; i < d + 1; i++)
        {
            rank[i] += sum;
            if (rank[i] < 0)
            {
                rank[i] += d + 1;
                rem0[i] += d + 1;
            }
            else if (rank[i] > d)
            {
                rank[i] -= d + 1;
                rem0[i] -= d + 1;
            }
        }

        // Compute the barycentric coordinates (p.10 in [Adams et al., 2010])
        for (int i = 0; i < d + 2; i++)
        {
            barycentric[i] = 0;
        }
        for (int i = 0; i < d + 1; i++)
        {
            float v = (elevated[i] - rem0[i]) * down_factor;
            barycentric[d - rank[i]] += v;
            barycentric[d - rank[i] + 1] -= v;
        }
        // Wrap around
        barycentric[0] += 1. + barycentric[d + 1];

        // Compute all vertices and their offset
        for (int remainder = 0; remainder < d + 1; remainder++)
        {
            for (int i = 0; i < d; i++)
            {
                key[i] = rem0[i] + canonical[remainder * (d + 1) + rank[i]];
            }

            os[n * (d + 1) + remainder] = hash_table.find(key, true) + 1;

            ws[n * (d + 1) + remainder] = barycentric[remainder];
        }
    }

    // Delete
    delete[] scale_factor;
    delete[] elevated;
    delete[] rem0;
    delete[] barycentric;
    delete[] rank;
    delete[] canonical;
    delete[] key;

    // Find the neighbors of each lattice point

    // Get the number of vertices in the lattice
    *M = hash_table.size();

    short *n1 = new short[d + 1];
    short *n2 = new short[d + 1];

    // For each of d + 1 axes,
    for (int j = 0; j < d + 1; j++)
    {
        for (int i = 0; i < *M; i++)
        {
            const short *key = hash_table.getKey(i);
            for (int k = 0; k < d; k++)
            {
                n1[k] = key[k] - 1;
                n2[k] = key[k] + 1;
            }
            n1[j] = key[j] + d;
            n2[j] = key[j] - d;

            blur_neighborsT[j * (*M) * 2 + i * 2 + 0] = hash_table.find(n1) + 1;
            blur_neighborsT[j * (*M) * 2 + i * 2 + 1] = hash_table.find(n2) + 1;
        }
    }

    delete[] n1;
    delete[] n2;
}

void cppCompute(const float *inp, const int N, const int value_size, const int M, const int d, const int *os, const float *ws, const int *blur_neighborsT, const bool reverse, float *out)
{
    float *values = new float[(M + 2) * value_size];
    fill(values, values + (M + 2) * value_size, 0.f);

    // Splatting
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < d + 1; j++)
        {
            int o = os[i * (d + 1) + j];
            float w = ws[i * (d + 1) + j];
            for (int k = 0; k < value_size; k++)
            {
                values[o * value_size + k] += w * inp[i * value_size + k];
            }
        }
    }

    // Blurring
    for (int j = reverse ? d : 0; j <= d && j >= 0; reverse ? j-- : j++)
    {
        for (int i = 0; i < M; i++)
        {
            int n1 = blur_neighborsT[j * M * 2 + i * 2 + 0];
            int n2 = blur_neighborsT[j * M * 2 + i * 2 + 1];
            for (int k = 0; k < value_size; k++)
            {
                float n1_val = values[n1 * value_size + k];
                float n2_val = values[n2 * value_size + k];

                values[(i + 1) * value_size + k] += 0.5 * (n1_val + n2_val);
            }
        }
    }

    // Slicing
    float alpha = 1.f / (1 + powf(2, -d));

    fill(out, out + N * value_size, 0.f);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < d + 1; j++)
        {
            int o = os[i * (d + 1) + j];
            float w = ws[i * (d + 1) + j];
            for (int k = 0; k < value_size; k++)
            {
                out[i * value_size + k] += w * values[o * value_size + k] * alpha;
            }
        }
    }

    delete[] values;
}