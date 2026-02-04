#include "simple_ols.hpp"
#include <cmath>

Eigen::VectorXd simpleOLS(
    int ns,
    int nf,
    const Eigen::MatrixXd &A,
    const Eigen::VectorXd &y,
    double eigenThreshold)
{
    Eigen::VectorXd v = Eigen::VectorXd::Zero(nf);
    if (ns <= 0 || nf <= 0)
    {
        return v;
    }

    // X'X (matriz Gram)
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(nf, nf);
    for (int j = 0; j < nf; ++j)
    {
        for (int jj = 0; jj < nf; ++jj)
        {
            double sum = 0.0;
            for (int i = 0; i < ns; ++i)
            {
                sum += A(i, j) * A(i, jj);
            }
            B(j, jj) = sum;
        }
    }

    // Eigendescomposicion de X'X
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(B);
    if (solver.info() != Eigen::Success)
    {
        return v;
    }

    Eigen::VectorXd wvec = solver.eigenvalues();
    Eigen::MatrixXd zmat = solver.eigenvectors();

    // (X'X)^-1 via pseudo-inversa con umbral
    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(nf, nf);
    for (int i = 0; i < nf; ++i)
    {
        for (int k = 0; k < nf; ++k)
        {
            double sum = 0.0;
            for (int j = 0; j < nf; ++j)
            {
                if (std::abs(wvec(j)) > eigenThreshold)
                {
                    sum += zmat(k, j) * (1.0 / wvec(j)) * zmat(i, j);
                }
            }
            C(i, k) = sum;
        }
    }

    // (X'X)^-1 X'
    Eigen::MatrixXd BB = Eigen::MatrixXd::Zero(nf, ns);
    for (int i = 0; i < ns; ++i)
    {
        for (int j = 0; j < nf; ++j)
        {
            double sum = 0.0;
            for (int jj = 0; jj < nf; ++jj)
            {
                sum += C(j, jj) * A(i, jj);
            }
            BB(j, i) = sum;
        }
    }

    // beta = (X'X)^-1 X' y
    for (int jj = 0; jj < nf; ++jj)
    {
        double sum = 0.0;
        for (int j = 0; j < ns; ++j)
        {
            sum += BB(jj, j) * y(j);
        }
        v(jj) = sum;
    }

    return v;
}
