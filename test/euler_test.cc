#include "five_moment/euler.h"
#include "euler_test_helpers.h"
#include "utilities.h"

#include <gtest/gtest.h>


using namespace warpii::five_moment;

TEST(EulerTest, EntropyVariablesSatisfyDerivativeIdentity) {
    double gamma = 1.4;
    Tensor<1, 5, double> q;
    q[0] = 2.3;
    q[1] = 0.58;
    q[2] = 0.33;
    q[3] = 0.0;
    q[4] = 10.0;

    const auto w = euler_entropy_variables<3>(q, gamma);
    //const auto eta = euler_mathematical_entropy<2>(q, gamma);

    // Perform a finite difference test of the relationship w = 
    double h = 1e-5;
    for (unsigned int i = 0; i < 4; i++) {
        auto q_L = Tensor(q);
        q_L[i] -= h;
        const double eta_L = euler_mathematical_entropy<2>(q_L, gamma);
        auto q_R = Tensor(q);
        q_R[i] += h;
        const double eta_R = euler_mathematical_entropy<2>(q_R, gamma);

        const double d_eta_di = (eta_R - eta_L) / (2*h);
        EXPECT_NEAR(d_eta_di, w[i], 1e-5);
    }
}

/**
 * Tests the identity psi = rho*u, which is eqn (4.5) from Chandrashekar (2012).
 */
TEST(EulerTest, EulerEntropyFluxPotentialTest) {
    double gamma = 1.4;
    random_euler_state<1>(gamma);
    for (unsigned int i = 0; i < 100; i++) {
        auto stateL = random_euler_state<1>(gamma);
        auto wL = euler_entropy_variables<1>(stateL, gamma);
        auto fL = euler_flux<1>(stateL, gamma);
        auto qL = euler_entropy_flux<1>(stateL, gamma);

        // Entropy flux potential
        Tensor<1, 3, double> psiL;
        for (unsigned int d = 0; d < 1; d++) {
            for (unsigned int c = 0; c < 5; c++) {
                psiL[d] += wL[c] * fL[c][d];
            }
            psiL[d] -= qL[d];
        }

        EXPECT_NEAR(psiL[0], stateL[1], 1e-8 * std::abs(stateL[0]));
    }
}

double ln_avg_reference_impl(double a, double b) {
    if (std::abs(a - b) < std::max(a, b) * 1e-6) {
        return (a + b) / 2.0;
    } else {
        return (b - a) / (std::log(b) - std::log(a));
    }
}

TEST(EulerFluxTests, LnAvgTest) {
    // Equal values
    EXPECT_EQ(ln_avg(0.4, 0.4), 0.4);
    EXPECT_NEAR(ln_avg(0.4, 0.6), ln_avg_reference_impl(0.4, 0.6), 1e-12);

    // Very small
    EXPECT_NEAR(ln_avg(1e-10, 1e-12), 2.1497576854210972e-11, 1e-16);
    EXPECT_NEAR(ln_avg(0.4, 0.4 + 1e-8), (0.8 + 1e-8) / 2.0, 1e-16);

    EXPECT_NEAR(ln_avg(1.0, 0.5), 0.7213475204444817, 1e-15);
}

TEST(EulerFluxTests, EulerCHECTest) {
    double gamma = 5.0 / 3.0;
    // Sod shocktube left and right states
    Tensor<1, 5, double> left({1.0, 0.0, 0.0, 0.0, 1.0 / (gamma - 1.0)});
    Tensor<1, 5, double> right({0.1, 0.0, 0.0, 0.0, 0.125 / (gamma - 1.0)});

    Tensor<1, 5, Tensor<1, 1, double>> actual = euler_CH_EC_flux<1>(left, right, gamma);
    EXPECT_EQ(actual[0][0], 0.0);
    EXPECT_NEAR(actual[1][0], 0.5 + 1.0 / 9, 1e-15);
    EXPECT_EQ(actual[4][0], 0.0);

    // Left: rho, u, p = [1.0, 1.0, 1.0]
    // Right: rho, u, p = [0.5, 0.5, 0.5]
    left = Tensor<1, 5, double>({1.0, 1.0, 0.0, 0.0, 0.5 * 1.0 + 1.0 / (gamma - 1.0)});
    right = Tensor<1, 5, double>({0.5, 0.5, 0.0, 0.0, 0.5 * 0.5 + 0.5 / (gamma - 1.0)});
    actual = euler_CH_EC_flux<1>(left, right, gamma);
    EXPECT_NEAR(actual[0][0], 0.7213475204444817, 1e-15);
    EXPECT_NEAR(actual[1][0],  1.4713475204444817, 1e-15);
    EXPECT_NEAR(actual[4][0], 2.192695040888963, 1e-15);
}

TEST(EulerFluxTests, EulerCHESTest) {
    double gamma = 5.0 / 3.0;
    // Sod shocktube left and right states
    Tensor<1, 5, double> left({1.0, 0.0, 0.0, 0.0, 1.0 / (gamma - 1.0)});
    Tensor<1, 5, double> right({0.1, 0.0, 0.0, 0.0, 0.125 / (gamma - 1.0)});

    Tensor<1, 1, double> normal;
    normal[0] = 1.0;
    Tensor<1, 5, double> actual = euler_CH_entropy_dissipating_flux<1>(
            left, right, normal, gamma);
    EXPECT_NEAR(actual[0], 0.6495190528383291, 1e-15);
    EXPECT_NEAR(actual[1], 0.5 + 1.0 / 9, 1e-15);
    EXPECT_NEAR(actual[4],  0.9381717944489488, 1e-15);
}

// This tests that the numerical entropy production,
// eqn (B.56) in Hennemann et al., is nearly zero.
TEST(EulerFluxTests, EntropyConservationTest1D) {
    double gamma = 1.4;
    random_euler_state<1>(gamma);
    for (unsigned int i = 0; i < 100; i++) {
        auto stateL = random_euler_state<1>(gamma);
        auto stateR = random_euler_state<1>(gamma);

        auto wL = euler_entropy_variables<1>(stateL, gamma);
        auto wR = euler_entropy_variables<1>(stateR, gamma);

        auto fL = euler_flux<1>(stateL, gamma);
        auto fR = euler_flux<1>(stateR, gamma);

        auto qL = euler_entropy_flux<1>(stateL, gamma);
        auto qR = euler_entropy_flux<1>(stateR, gamma);

        // Entropy flux potential
        Tensor<1, 1, double> psiL;
        Tensor<1, 1, double> psiR;
        for (unsigned int d = 0; d < 1; d++) {
            psiL[d] = 0.0;
            psiR[d] = 0.0;
            for (unsigned int c = 0; c < 5; c++) {
                psiL[d] += wL[c] * fL[c][d];
                psiR[d] += wR[c] * fR[c][d];
            }
            psiL[d] -= qL[d];
            psiR[d] -= qR[d];

            EXPECT_NEAR(psiL[d], stateL[d+1], 1e-10*stateL.norm());
            EXPECT_NEAR(psiR[d], stateR[d+1], 1e-10*stateR.norm());
        }

        Tensor<1, 1, double> n;
        n[0] = 1.0;

        const auto wjump = wR - wL;
        const auto psijump = psiR - psiL;
        const auto numerical_flux = euler_CH_EC_flux<1>(stateL, stateR, gamma) * n;

        const double r = wjump * numerical_flux - psijump * n;
        EXPECT_NEAR(r, 0.0, 1e-10 * (qL.norm() + qR.norm()));
    }
}
// This tests that the numerical entropy production,
// eqn (B.56) in Hennemann et al., is nearly zero.
TEST(EulerFluxTests, EntropyConservationTest2D) {
    double gamma = 1.4;
    random_euler_state<2>(gamma);
    for (unsigned int i = 0; i < 100; i++) {
        auto stateL = random_euler_state<2>(gamma);
        auto stateR = random_euler_state<2>(gamma);
        SHOW(stateL);
        SHOW(stateR);

        auto wL = euler_entropy_variables<2>(stateL, gamma);
        auto wR = euler_entropy_variables<2>(stateR, gamma);
        SHOW(wL);
        SHOW(wR);

        auto fL = euler_flux<2>(stateL, gamma);
        auto fR = euler_flux<2>(stateR, gamma);

        auto qL = euler_entropy_flux<2>(stateL, gamma);
        SHOW(qL);
        auto qR = euler_entropy_flux<2>(stateR, gamma);
        SHOW(qR);

        // Entropy flux potential
        Tensor<1, 2, double> psiL;
        Tensor<1, 2, double> psiR;
        for (unsigned int d = 0; d < 2; d++) {
            psiL[d] = 0.0;
            psiR[d] = 0.0;
            for (unsigned int c = 0; c < 5; c++) {
                psiL[d] += wL[c] * fL[c][d];
                psiR[d] += wR[c] * fR[c][d];
            }
            psiL[d] -= qL[d];
            psiR[d] -= qR[d];

            EXPECT_NEAR(psiL[d], stateL[d+1], 1e-10*stateL.norm());
            EXPECT_NEAR(psiR[d], stateR[d+1], 1e-10*stateR.norm());
        }

        SHOW(psiL);
        SHOW(psiR);

        // Normal direction
        Tensor<1, 2, double> n;
        n[0] = rand_01() + 0.1;
        n[1] = rand_01() + 0.1;
        n = n / n.norm();
        EXPECT_NEAR(n*n, 1.0, 1e-15);

        const auto wjump = wR - wL;
        const auto psijump = psiR - psiL;
        const auto numerical_flux = euler_CH_EC_flux<2>(stateL, stateR, gamma) * n;

        const double r = wjump * numerical_flux - n * psijump;
        EXPECT_NEAR(r, 0.0, 1e-10 * (qL.norm() + qR.norm()));
    }
}

// This tests that the numerical entropy production,
// eqn (B.56) in Hennemann et al., is negative for the entropy dissipating flux.
TEST(EulerFluxTests, EntropyDissipationTest2D) {
    double gamma = 1.4;
    random_euler_state<2>(gamma);
    for (unsigned int i = 0; i < 100; i++) {
        auto stateL = random_euler_state<2>(gamma);
        auto stateR = random_euler_state<2>(gamma);
        SHOW(stateL);
        SHOW(stateR);

        auto wL = euler_entropy_variables<2>(stateL, gamma);
        auto wR = euler_entropy_variables<2>(stateR, gamma);
        SHOW(wL);
        SHOW(wR);

        auto fL = euler_flux<2>(stateL, gamma);
        auto fR = euler_flux<2>(stateR, gamma);

        auto qL = euler_entropy_flux<2>(stateL, gamma);
        SHOW(qL);
        auto qR = euler_entropy_flux<2>(stateR, gamma);
        SHOW(qR);

        // Entropy flux potential
        Tensor<1, 2, double> psiL;
        Tensor<1, 2, double> psiR;
        for (unsigned int d = 0; d < 2; d++) {
            psiL[d] = 0.0;
            psiR[d] = 0.0;
            for (unsigned int c = 0; c < 5; c++) {
                psiL[d] += wL[c] * fL[c][d];
                psiR[d] += wR[c] * fR[c][d];
            }
            psiL[d] -= qL[d];
            psiR[d] -= qR[d];

            EXPECT_NEAR(psiL[d], stateL[d+1], 1e-10*stateL.norm());
            EXPECT_NEAR(psiR[d], stateR[d+1], 1e-10*stateR.norm());
        }

        SHOW(psiL);
        SHOW(psiR);

        // Normal direction
        Tensor<1, 2, double> n;
        n[0] = rand_01() + 0.1;
        n[1] = rand_01() + 0.1;
        n = n / n.norm();
        EXPECT_NEAR(n*n, 1.0, 1e-15);

        const auto wjump = wR - wL;
        const auto psijump = psiR - psiL;
        const auto numerical_flux = euler_CH_entropy_dissipating_flux<2>(stateL, stateR, n, gamma);

        const double r = wjump * numerical_flux - n * psijump;
        EXPECT_LE(r, 0.0);
    }
}

TEST(EulerFluxTests, RoeFlux1D) {
    double gamma = 5.0 / 3.0;
    Tensor<1, 5, double> q_in({1.0, 0.0, 0.0, 0.0, 1.0});
    Tensor<1, 5, double> q_out({0.1, 0.0, 0.0, 0.0, 0.1});
    Tensor<1, 1, double> n({1.0});

    auto flux = euler_roe_flux<1>(q_in, q_out, n, gamma, false);
    // Definitely the mass flux should be positive.
    ASSERT_GT(flux[0], 0.0);

    // The function should give the outward directed flux no matter the orientation of n
    flux = euler_roe_flux<1>(q_in, q_out, Tensor<1, 1, double>({-1.0}), gamma, false);
    ASSERT_GT(flux[0], 0.0);

    q_in = Tensor<1, 5, double>({1.0, 1.0, 0.0, 0.0, 1.0});
    q_out = Tensor<1, 5, double>({1.0, 1.0, 0.0, 0.0, 1.0});

    flux = euler_roe_flux<1>(q_in, q_out, Tensor<1, 1, double>({1.0}), gamma, false);
    ASSERT_GT(flux[0], 0.0);

    flux = euler_roe_flux<1>(q_in, q_out, Tensor<1, 1, double>({-1.0}), gamma, false);
    ASSERT_LT(flux[0], 0.0);
}

// qL and qR are [p, rho, u]
void test_roe_property_1d_x(Tensor<1, 3, double> qL, Tensor<1, 3, double> qR, 
        double n, double gamma) {
    double pR = qR[0];
    double rhoR = qR[1];
    double uR = qR[2];
    double pL = qL[0];
    double rhoL = qL[1];
    double uL = qL[2];
    double eR = (1.0 / (gamma - 1.0)) * pR + 0.5 * uR * uR * rhoR;
    double eL = (1.0 / (gamma - 1.0)) * pL + 0.5 * uL * uL * rhoL;

    Tensor<1, 5, double> region_L({rhoL, rhoL*uL, 0.0, 0.0, eL});
    SHOW(region_L);
    Tensor<1, 5, double> region_R({rhoR, rhoR*uR, 0.0, 0.0, eR});
    SHOW(region_R);

    Tensor<1, 1, double> normal({n});
    const auto FL = euler_flux<1>(region_L, gamma) * normal;
    const auto FR = euler_flux<1>(region_R, gamma) * normal;

    const auto flux_jump = FR - FL;
    SHOW(flux_jump);
    const auto jump = region_R - region_L;
    SHOW(jump);
    ASSERT_NEAR(flux_jump[0] / jump[0], flux_jump[1] / jump[1], 1e-14);
    ASSERT_NEAR(flux_jump[0] / jump[0], flux_jump[4] / jump[4], 1e-14);
    const auto s = flux_jump[0] / jump[0];

    const auto roe_flux = euler_roe_flux<1>(region_L, region_R, normal, gamma, true);

    const auto avg_flux = 0.5 * (FL + FR);
    SHOW(roe_flux - avg_flux);
    const auto viscous_correction = roe_flux - avg_flux;
    const auto expected = -0.5 * s * jump;

    SHOW(viscous_correction);
    SHOW(expected);
    for (unsigned int i = 0; i < 5; i++) {
        ASSERT_NEAR(viscous_correction[i], expected[i], 1e-14);
    }
}

// Values are computed using SodShockTube.jl
TEST(EulerFluxTests, RoeFluxHasRoeProperty) {
    // Classic Sod shocktube
    double gamma = 1.4;
    double p4 = 0.30313017804679177;
    double u4 = 0.9274526200369384;
    double rho4 = 0.2655737117032393;

    double p5 = 0.1;
    double u5 = 0.0;
    double rho5 = 0.125;

    test_roe_property_1d_x(Tensor<1, 3, double>({p4, rho4, u4}), 
            Tensor<1, 3, double>({p5, rho5, u5}), 
            1.0, gamma);

    test_roe_property_1d_x(
            Tensor<1, 3, double>({p4, rho4, -u4}), 
            Tensor<1, 3, double>({p5, rho5, u5}), 
            -1.0, gamma);

    // Contact discontinuity
    double p3 = p4;
    double u3 = u4;
    double rho3 = 0.42631942817462254;

    test_roe_property_1d_x(Tensor<1, 3, double>({p3, rho3, u3}), 
            Tensor<1, 3, double>({p4, rho4, u4}), 
            1.0, gamma);

    test_roe_property_1d_x(
            Tensor<1, 3, double>({p3, rho3, -u3}), 
            Tensor<1, 3, double>({p4, rho4, -u4}), 
            -1.0, gamma);
}

