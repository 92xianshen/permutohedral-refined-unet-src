#include <iostream>
#include <Eigen/Dense>

#include "permutohedral_cppfactory.h"

using namespace std;
using namespace Eigen;

class DenseCRF
{
protected:
    int N_, n_iterations_;
    Permutohedral *bilateral_filter_, *spatial_filter_;
    float bilateral_compat_, spatial_compat_;
    MatrixXf compatibility_matrix_;

public:
    DenseCRF(
        int N,
        int n_classes,
        int d_bifeats,
        int d_spfeats,
        float bilateral_compat,
        float spatial_compat
    )
    {
        N_ = N;
        n_classes_ = n_classes;
        bilateral_filter = new Permutohedral(N_, d_bifeats);
        spatial_filter = new Permutohedral(N_, d_spfeats);
    }
    ~DenseCRF()
    {
        delete bilateral_filter;
        delete spatial_filter;
    }
    void softmax(const MatrixXf &in, MatrixXf &out)
    {
        // in and out share the shape of [N, d], channel-last
        out = (in.colwise() - in.rowwise().maxCoeff()).array().exp();
        VectorXf sm = out.rowwise().sum();
        out.array().colwise() /= sm.array();
    }
    void inference(
        const float *unary_1d,
        const float *bilateral_feats_1d,
        const float *spatial_feats_1d,
        const bool reversal,
        float *out_1d
    )
    {
        bilateral_filter_.init(bilateral_feats_1d);
        spatial_filter_.init(spatial_feats_1d);

        MatrixXf all_ones(N_, 1), tmp(N_, 1), bilateral_norm_vals(N_, 1), spatial_norm_vals(N_, 1);
        all_ones.setOnes();
        tmp.setZero();
        bilateral_filter_.compute(all_ones, reversal, tmp);
        bilateral_norm_vals = (tmp.array().pow(.5f) + 1e-20).inverse();
        tmp.setZero();
        spatial_filter_.compute(all_ones, reversal, tmp);
        spatial_norm_vals = (tmp.array().pow(.5f) + 1e-20).inverse();

        const Map<const MatrixXf> unary_flat(unary_1d, N_, n_classes_);
        Map<MatrixXf> out_flat(out_1d, N_, n_classes_);

        softmax(-unary_flat, Q);
        for (int i = 0; i < n_iterations_; i++)
        {
            MatrixXf tmp1 = -unary_flat;

            bilateral_filter_.compute()
        }
    }
};





//void inference(const MatrixXf &unary, int n_iterations, MatrixXf &Q)
//{
//
//
//    softmax(-unary, Q);
//    for (int i = 0; i < n_iterations; i++)
//    {
//        MatrixXf tmp1 = -unary;
//
//        compute(Q.array() * spatial_norm_vals.array(), false, bilateral_out);
//        bilateral_out.array() *= bilateral_norm_vals.array();
//
//        compute(Q.cwiseProduct(spatial_norm_vals), false, spatial_out);
//        spatial_out.array() *= spatial_norm_vals.array();
//
//        MatrixXf message_passing = bilateral_compat * bilateral_out +  spatial_compat * spatial_out;
//
//        MatrixXf pairwise = message_passing * compatibility_matrix;
//
//        tmp1.array() -= pairwise.array();
//
//        softmax(tmp1, Q);
//    }
//}
