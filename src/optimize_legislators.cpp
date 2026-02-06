#include "optimize_legislators.hpp"
#include <cmath>
#include <stdexcept>

namespace
{
    double computeStep(double gradNormSq, double stepUnit)
    {
        if (gradNormSq <= 0.0)
        {
            return 0.0;
        }
        return stepUnit / std::sqrt(gradNormSq);
    }

    Eigen::MatrixXd invertInformationMatrix(
        const Eigen::MatrixXd &info,
        double eigenThreshold)
    {
        const int n = static_cast<int>(info.rows());
        if (n == 0)
        {
            return Eigen::MatrixXd();
        }

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(info);
        if (solver.info() != Eigen::Success)
        {
            return Eigen::MatrixXd::Zero(n, n);
        }

        Eigen::VectorXd wvec = solver.eigenvalues();
        Eigen::MatrixXd zmat = solver.eigenvectors();

        Eigen::MatrixXd inv = Eigen::MatrixXd::Zero(n, n);
        for (int i = 0; i < n; ++i)
        {
            for (int k = 0; k < n; ++k)
            {
                double sum = 0.0;
                for (int j = 0; j < n; ++j)
                {
                    double eig = wvec(n - 1 - j);
                    if (std::abs(eig) > eigenThreshold)
                    {
                        sum += zmat(k, n - 1 - j) * (1.0 / eig) * zmat(i, n - 1 - j);
                    }
                }
                inv(i, k) = sum;
            }
        }
        return inv;
    }

    TimeTrends buildLegendreTimeTrends(const std::vector<int> &servedPeriods)
    {
        const int kk = static_cast<int>(servedPeriods.size());
        TimeTrends trends(kk);
        double xinc = 0.0;
        if (kk > 1)
        {
            xinc = 2.0 / (static_cast<double>(kk) - 1.0);
        }

        for (int i = 0; i < kk; ++i)
        {
            double xtime = -1.0 + static_cast<double>(i) * xinc;
            trends.values(i, 0) = 1.0;
            trends.values(i, 1) = xtime;
            trends.values(i, 2) = (3.0 * xtime * xtime - 1.0) / 2.0;
            trends.values(i, 3) = (5.0 * xtime * xtime * xtime - 3.0 * xtime) / 2.0;
        }

        return trends;
    }

    Eigen::VectorXd computeInitialBetasForDimension(
        int kk,
        const TimeTrends &trends,
        const Eigen::VectorXd &yyy)
    {
        Eigen::VectorXd vvv = Eigen::VectorXd::Zero(4);
        if (kk < 5)
        {
            vvv(0) = yyy(0);
            return vvv;
        }

        int nf = 2;
        if (kk == 6)
        {
            nf = 3;
        }
        else if (kk >= 7)
        {
            nf = 4;
        }

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(kk, nf);
        for (int i = 0; i < kk; ++i)
        {
            for (int j = 0; j < nf; ++j)
            {
                A(i, j) = trends.values(i, j);
            }
        }

        Eigen::VectorXd beta = simpleOLS(kk, nf, A, yyy);
        for (int j = 0; j < nf; ++j)
        {
            vvv(j) = beta(j);
        }
        return vvv;
    }

    void updateCoefficientsFromVector(
        TemporalCoefficients &coefficients,
        int termIndex,
        const Eigen::VectorXd &values)
    {
        for (int k = 0; k < values.size(); ++k)
        {
            coefficients(termIndex, k) = values(k);
        }
    }

    Eigen::VectorXd getCoefficientsVector(
        const TemporalCoefficients &coefficients,
        int termIndex,
        int numDim)
    {
        Eigen::VectorXd values(numDim);
        for (int k = 0; k < numDim; ++k)
        {
            values(k) = coefficients(termIndex, k);
        }
        return values;
    }

    void storeDervish(
        LegislatorOptimizationResult &result,
        const LegislatorDerivativesResult &derivs)
    {
        const int ns = static_cast<int>(derivs.derivatives0.size());
        if (derivs.totalVotes <= 0)
        {
            result.dervish.setZero();
            return;
        }

        for (int k = 0; k < ns; ++k)
        {
            result.dervish(0, k) = derivs.derivatives0(k) / derivs.totalVotes;
            result.dervish(1, k) = derivs.derivatives1(k) / derivs.totalVotes;
            result.dervish(2, k) = derivs.derivatives2(k) / derivs.totalVotes;
            result.dervish(3, k) = derivs.derivatives3(k) / derivs.totalVotes;
        }
    }
}

LegislatorOptimizationResult optimizeLegislator(
    int legislatorIndex,
    const LegislatorPeriodInfo &periodInfo,
    const Eigen::MatrixXd &legislatorDataCoords,
    const Eigen::MatrixXd &rollCallMidpoints,
    const Eigen::MatrixXd &rollCallSpreads,
    const VoteMatrix &votes,
    const std::vector<bool> &validRollCalls,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    TemporalModel maxModel,
    int firstPeriod,
    int lastPeriod,
    const LegislatorOptimizerConfig &config)
{
    const int numDim = static_cast<int>(weights.size()) - 1;
    LegislatorOptimizationResult result(numDim);

    std::vector<int> servedPeriods;
    for (int j = firstPeriod; j <= lastPeriod; ++j)
    {
        if (periodInfo.servedIn(j))
        {
            servedPeriods.push_back(j);
        }
    }

    const int nepcog = static_cast<int>(servedPeriods.size());
    if (nepcog == 0)
    {
        return result;
    }

    TimeTrends timeTrends = buildLegendreTimeTrends(servedPeriods);

    TemporalCoefficients coefficients(numDim);
    for (int k = 0; k < numDim; ++k)
    {
        Eigen::VectorXd yyy(nepcog);
        for (int idx = 0; idx < nepcog; ++idx)
        {
            int period = servedPeriods[idx];
            int dataIndex = periodInfo.dataIndices[period];
            yyy(idx) = legislatorDataCoords(dataIndex, k);
        }

        Eigen::VectorXd vvv = computeInitialBetasForDimension(nepcog, timeTrends, yyy);
        coefficients(0, k) = vvv(0);
        coefficients(1, k) = vvv(1);
        coefficients(2, k) = vvv(2);
        coefficients(3, k) = vvv(3);
    }

    double sum = 0.0;
    for (int k = 0; k < numDim; ++k)
    {
        sum += coefficients(0, k) * coefficients(0, k);
    }
    if (sum > 1.0)
    {
        double scale = config.unitSphereScale / std::sqrt(sum);
        for (int k = 0; k < numDim; ++k)
        {
            coefficients(0, k) *= scale;
        }
    }

    LegislatorDerivativesResult finalDerivs(numDim, nepcog);
    double saveGMP = 0.0;

    auto runOptimizationTerm = [&](int termIndex, TemporalModel model, int numSearchPoints, bool constrainSphere)
    {
        std::vector<double> yGMP(numSearchPoints, 0.0);
        std::vector<Eigen::VectorXd> yGamma(numSearchPoints, Eigen::VectorXd::Zero(numDim));
        std::vector<double> yyGMP(config.maxIterations, 0.0);

        for (int iter = 0; iter < config.maxIterations; ++iter)
        {
            Eigen::VectorXd xbetaSv = getCoefficientsVector(coefficients, termIndex, numDim);

            LegislatorDerivativesResult derivs = computeLegislatorDerivatives(
                legislatorIndex,
                periodInfo,
                timeTrends,
                coefficients,
                rollCallMidpoints,
                rollCallSpreads,
                votes,
                validRollCalls,
                weights,
                normalCDF,
                model,
                firstPeriod,
                lastPeriod);

            if (iter == 0)
            {
                saveGMP = derivs.geometricMeanProb;
            }

            if (derivs.totalVotes <= 0)
            {
                break;
            }

            Eigen::VectorXd grad(numDim);
            if (termIndex == 0)
            {
                grad = derivs.derivatives0 / static_cast<double>(derivs.totalVotes);
            }
            else if (termIndex == 1)
            {
                grad = derivs.derivatives1 / static_cast<double>(derivs.totalVotes);
            }
            else if (termIndex == 2)
            {
                grad = derivs.derivatives2 / static_cast<double>(derivs.totalVotes);
            }
            else
            {
                grad = derivs.derivatives3 / static_cast<double>(derivs.totalVotes);
            }

            double gradNormSq = grad.squaredNorm();
            double step = computeStep(gradNormSq, config.stepUnit);
            if (step == 0.0)
            {
                break;
            }

            double xinc = 0.0;
            int nninc = numSearchPoints;
            for (int kk = 0; kk < numSearchPoints; ++kk)
            {
                Eigen::VectorXd candidate = xbetaSv - xinc * grad;

                if (constrainSphere)
                {
                    double sphereSum = candidate.squaredNorm();
                    if (sphereSum > 1.0)
                    {
                        candidate /= std::sqrt(sphereSum);
                        updateCoefficientsFromVector(coefficients, termIndex, candidate);
                        LegislatorDerivativesResult d = computeLegislatorDerivatives(
                            legislatorIndex,
                            periodInfo,
                            timeTrends,
                            coefficients,
                            rollCallMidpoints,
                            rollCallSpreads,
                            votes,
                            validRollCalls,
                            weights,
                            normalCDF,
                            model,
                            firstPeriod,
                            lastPeriod);

                        yGMP[kk] = d.geometricMeanProb;
                        yGamma[kk] = candidate;
                        nninc = kk + 1;
                        break;
                    }
                }

                updateCoefficientsFromVector(coefficients, termIndex, candidate);
                LegislatorDerivativesResult d = computeLegislatorDerivatives(
                    legislatorIndex,
                    periodInfo,
                    timeTrends,
                    coefficients,
                    rollCallMidpoints,
                    rollCallSpreads,
                    votes,
                    validRollCalls,
                    weights,
                    normalCDF,
                    model,
                    firstPeriod,
                    lastPeriod);

                yGMP[kk] = d.geometricMeanProb;
                yGamma[kk] = candidate;
                xinc += step;
            }

            std::vector<double> gmpSubset(yGMP.begin(), yGMP.begin() + nninc);
            std::vector<size_t> indices = argsort(gmpSubset);
            size_t bestIdx = indices.back();

            updateCoefficientsFromVector(coefficients, termIndex, yGamma[bestIdx]);
            yyGMP[iter] = gmpSubset[bestIdx];

            if (iter >= 2)
            {
                double stopper = yyGMP[iter] - yyGMP[iter - 1];
                if (stopper <= config.convergenceTol)
                {
                    break;
                }
            }
        }

        finalDerivs = computeLegislatorDerivatives(
            legislatorIndex,
            periodInfo,
            timeTrends,
            coefficients,
            rollCallMidpoints,
            rollCallSpreads,
            votes,
            validRollCalls,
            weights,
            normalCDF,
            model,
            firstPeriod,
            lastPeriod);

        if (saveGMP - finalDerivs.geometricMeanProb > config.gmpDropTol)
        {
            throw std::runtime_error("optimizeLegislator: GMP decreased unexpectedly");
        }
    };

    runOptimizationTerm(0, TemporalModel::Constant, config.numSearchPointsConst, true);

    result.logLikelihood0 = finalDerivs.logLikelihood;
    result.logLikelihood1 = finalDerivs.logLikelihood;
    result.logLikelihood2 = finalDerivs.logLikelihood;
    result.logLikelihood3 = finalDerivs.logLikelihood;

    if (maxModel >= TemporalModel::Linear && nepcog >= 5)
    {
        for (int k = 0; k < numDim; ++k)
        {
            coefficients(1, k) = 0.0;
        }
        runOptimizationTerm(1, TemporalModel::Linear, config.numSearchPointsTemporal, false);
        result.logLikelihood1 = finalDerivs.logLikelihood;
        result.logLikelihood2 = finalDerivs.logLikelihood;
        result.logLikelihood3 = finalDerivs.logLikelihood;
    }

    if (maxModel >= TemporalModel::Quadratic && nepcog >= 6)
    {
        for (int k = 0; k < numDim; ++k)
        {
            coefficients(2, k) = 0.0;
        }
        runOptimizationTerm(2, TemporalModel::Quadratic, config.numSearchPointsTemporal, false);
        result.logLikelihood2 = finalDerivs.logLikelihood;
        result.logLikelihood3 = finalDerivs.logLikelihood;
    }

    if (maxModel >= TemporalModel::Cubic && nepcog >= 7)
    {
        for (int k = 0; k < numDim; ++k)
        {
            coefficients(3, k) = 0.0;
        }
        runOptimizationTerm(3, TemporalModel::Cubic, config.numSearchPointsTemporal, false);
        result.logLikelihood3 = finalDerivs.logLikelihood;
    }

    result.totalVotes = finalDerivs.totalVotes;
    result.periodCoordinates = finalDerivs.periodCoordinates;
    result.coefficients = coefficients;

    result.covariance0 = invertInformationMatrix(finalDerivs.infoMatrix0, config.eigenThreshold);
    if (nepcog >= 5)
    {
        result.covariance1 = invertInformationMatrix(finalDerivs.infoMatrix1, config.eigenThreshold);
    }
    if (nepcog >= 6)
    {
        result.covariance2 = invertInformationMatrix(finalDerivs.infoMatrix2, config.eigenThreshold);
    }
    if (nepcog >= 7)
    {
        result.covariance3 = invertInformationMatrix(finalDerivs.infoMatrix3, config.eigenThreshold);
    }

    storeDervish(result, finalDerivs);

    return result;
}
