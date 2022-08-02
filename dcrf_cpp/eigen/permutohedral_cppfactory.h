
#include <stdbool.h>
#include <vector>
#include <iostream>
#include <string.h>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

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

class Permutohedral
{
protected:
    int N_ = 0, d_ = 0, M_ = 0;
    float alpha_ = 0.f;
    MatrixXi os_, blur_neighbors1T_, blur_neighbors2T_;
    MatrixXf ws_;

public:
    Permutohedral(int N, int d)
    {
        N_ = N;
        d_ = d;

        // Slicing
        alpha_ = 1.f / (1 + powf(2, -d_));

        os_.resize(N_, d_ + 1);
        ws_.resize(N_, d_ + 1);

        // Allocate predefined factors

    }

    void init(const float *features)
    {
        // Compute the lattice coordinates for each feature [there is going to be a lot of magic here]
        HashTable hash_table(d_, N_ * (d_ + 1));

        // Allocate local memories
        float *elevated = new float[d_ + 1];
        int *rem0 = new int[d_ + 1];
        float *barycentric = new float[d_ + 2];
        short *rank = new short[d_ + 1];
        short *key = new short[d_ + 1];
        short *canonical = new short[(d_ + 1) * (d_ + 1)];
        float *scale_factor = new float[d_];

        // Compute the canonical simplex
        for (int i = 0; i < d_ + 1; i++)
        {
            for (int j = 0; j < d_ - i + 1; j++)
            {
                canonical[i * (d_ + 1) + j] = i;
            }
            for (int j = d_ - i + 1; j < d_ + 1; j++)
            {
                canonical[i * (d_ + 1) + j] = i - (d_ + 1);
            }
        }

        // Expected standard deviation of our filter (p.6 in [Adams et al., 2010])
        float inv_std_dev = sqrt(2. / 3.) * (d_ + 1);
        // Compute the diagonal part of E (p.5 in [Adams et al., 2010])
        for (int i = 0; i < d_; i++)
        {
            scale_factor[i] = 1. / sqrt(double((i + 2) * (i + 1))) * inv_std_dev;
        }

        // Compute the simplex each feature lies in
        for (int n = 0; n < N_; n++)
        {
            // Elevate the feature (y = Ep, see p.5 in [Adams et al., 2010])
            const float *f = &features[n * d_];

            // sm contains the sum of 1...n of our feature vector
            float sm = 0;
            for (int j = d_; j > 0; j--)
            {
                float cf = f[j - 1] * scale_factor[j - 1];
                elevated[j] = sm - j * cf;
                sm += cf;
            }
            elevated[0] = sm;

            // Find the closest 0-colored simplex through rounding
            float down_factor = 1.f / (d_ + 1);
            float up_factor = (d_ + 1);
            int sum = 0;
            for (int i = 0; i < d_ + 1; i++)
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
            fill(rank, rank + d_ + 1, 0);
            for (int i = 0; i < d_; i++)
            {
                double di = elevated[i] - rem0[i];
                for (int j = i + 1; j < d_ + 1; j++)
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
            for (int i = 0; i < d_ + 1; i++)
            {
                rank[i] += sum;
                if (rank[i] < 0)
                {
                    rank[i] += d_ + 1;
                    rem0[i] += d_ + 1;
                }
                else if (rank[i] > d_)
                {
                    rank[i] -= d_ + 1;
                    rem0[i] -= d_ + 1;
                }
            }

            // Compute the barycentric coordinates (p.10 in [Adams et al., 2010])
            fill(barycentric, barycentric + d_ + 2, 0);
            for (int i = 0; i < d_ + 1; i++)
            {
                float v = (elevated[i] - rem0[i]) * down_factor;
                barycentric[d_ - rank[i]] += v;
                barycentric[d_ - rank[i] + 1] -= v;
            }
            // Wrap around
            barycentric[0] += 1. + barycentric[d_ + 1];

            // Compute all vertices and their offset
            for (int remainder = 0; remainder < d_ + 1; remainder++)
            {
                for (int i = 0; i < d_; i++)
                {
                    key[i] = rem0[i] + canonical[remainder * (d_ + 1) + rank[i]];
                }

                os_(n, remainder) = hash_table.find(key, true) + 1;
                ws_(n, remainder) = barycentric[remainder];
            }
        }

        // Delete
        delete[] elevated;
        delete[] rem0;
        delete[] barycentric;
        delete[] rank;
        delete[] key;
        delete[] canonical;
        delete[] scale_factor;

        // Find the neighbors of each lattice point

        // Get the number of vertices in the lattice
        M_ = hash_table.size();
        blur_neighbors1T_.resize(d_ + 1, M_);
        blur_neighbors2T_.resize(d_ + 1, M_);

        short *n1 = new short[d_ + 1];
        short *n2 = new short[d_ + 1];

        // For each of d + 1 axes,
        for (int j = 0; j < d_ + 1; j++)
        {
            for (int i = 0; i < M_; i++)
            {
                const short *key = hash_table.getKey(i);
                for (int k = 0; k < d_; k++)
                {
                    n1[k] = key[k] - 1;
                    n2[k] = key[k] + 1;
                }
                n1[j] = key[j] + d_;
                n2[j] = key[j] - d_;

                blur_neighbors1T_(j, i) = hash_table.find(n1) + 1;
                blur_neighbors2T_(j, i) = hash_table.find(n2) + 1;
            }
        }

        delete[] n1;
        delete[] n2;
    }

    void compute(const MatrixXf &inp_flat, const bool reversal, MatrixXf &out_flat)
    {
        int value_size = inp_flat.cols();

        MatrixXf values(M_ + 2, value_size);
        values.setZero();

        // Splatting
        for (int i = 0; i < N_; i++)
        {
            for (int j = 0; j < d_ + 1; j++)
            {
                int o = os_(i, j);
                float w = ws_(i, j);
                values(o, all) += w * inp_flat(i, all);
            }
        }

        // Blurring
        for (int j = reversal ? d_ : 0; j <= d_ && j >= 0; reversal ? j-- : j++)
        {
            VectorXi n1s = blur_neighbors1T_(j, all);
            VectorXi n2s = blur_neighbors2T_(j, all);

            MatrixXf n1_vals = values(n1s, all);
            MatrixXf n2_vals = values(n2s, all);

            values(seq(1, M_), all) += 0.5 * (n1_vals + n2_vals);
        }

        out_flat.setZero();

        for (int i = 0; i < N_; i++)
        {
            for (int j = 0; j < d_ + 1; j++)
            {
                int o = os_(i, j);
                float w = ws_(i, j);
                out_flat(i, all) += w * values(o, all) * alpha_;
            }
        }
    }

    void testCompute(const float *inp_1d, const int N, const int value_size, const bool reversal, float *out_1d)
    {
        const MatrixXf inp_flat = Map< const MatrixXf >(inp_1d, N, value_size);
        MatrixXf out_flat(N, value_size);
        compute(inp_flat, reversal, out_flat);

        for (int i = 0; i < out_flat.size(); i++)
        {
            out_1d[i] = out_flat.data()[i];
        }
    }

    ~Permutohedral(){}
};
