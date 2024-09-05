#include "geometry.h"
#include <deal.II/base/vectorization.h>

using namespace warpii;

namespace warpii {

/**
 * Given a normal vector to a plane, computes two basis vectors for that plane,
 * which we call the tangent and binormal vectors.
 */
std::pair<Tensor<1, 3, double>, Tensor<1, 3, double>> tangent_and_binormal(
    const Tensor<1, 3, double> n) {
    // Construct a reference vector pointing in an arbitrary direction
    Tensor<1, 3, double> a;
    a[0] = 0.;
    a[1] = 0.;
    a[2] = 1.;
    
    // If n is too nearly colinear with a, pick a different a.
    // By the Brouwer fixed point theorem, no continuous mapping from S^2 to itself
    // can avoid having a fixed point. This introduced a discontinuity if it detects
    // that n is at the fixed point.
    if (std::abs(a * n) >= 0.9) {
        a[1] = 1.;
        a[2] = 0.;
    }

    auto tangent = a - (a * n) * n;
    tangent = tangent / tangent.norm();

    auto binormal = cross_product_3d(n, tangent);
    return std::make_pair(tangent, binormal);
}

template std::pair<Tensor<1, 3, VectorizedArray<double>>, Tensor<1, 3, VectorizedArray<double>>>
         tangent_and_binormal<VectorizedArray<double>>(
                 const Tensor<1, 3, VectorizedArray<double>>);

}
