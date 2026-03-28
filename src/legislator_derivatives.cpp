/**
 * Implementacion de computeLegislatorDerivatives (PROX).
 * Calcula derivadas del log-likelihood respecto a las coordenadas espaciales de un legislador especifico,
 * considerando modelos temporales.
 */

#include "legislator_derivatives.hpp"
#include <cmath>
#include <algorithm>
#include <array>

namespace
{

    /**
     * Reconstruye coordenadas del legislador para un periodo segun modelo temporal.
     * Implementa:
     * - Constante: x_k = beta_0k
     * - Lineal: x_k = beta_0k + beta_1k * t
     * - Cuadratico: x_k = beta_0k + beta_1k * t + beta_2k * t^2
     * - Cubico: x_k = beta_0k + beta_1k * t + beta_2k * t^2 + beta_3k * t^3
     *
     * @param coefficients Coeficientes del modelo temporal
     * @param timeTrends Valores temporales [1, t, t^2, t^3]
     * @param periodIndex Indice del periodo
     * @param model Tipo de modelo temporal
     * @param numDimensions Numero de dimensiones
     * @return Coordenadas reconstruidas
     */
    Eigen::VectorXd reconstructCoordinates(
        const TemporalCoefficients &coefficients,
        const TimeTrends &timeTrends,
        int periodIndex,
        TemporalModel model,
        int numDimensions)
    {
        Eigen::VectorXd coords(numDimensions);

        for (int k = 0; k < numDimensions; ++k)
        {
            double x = timeTrends(periodIndex, 0) * coefficients(0, k);

            if (model >= TemporalModel::Linear)
            {
                x += timeTrends(periodIndex, 1) * coefficients(1, k);
            }
            if (model >= TemporalModel::Quadratic)
            {
                x += timeTrends(periodIndex, 2) * coefficients(2, k);
            }
            if (model >= TemporalModel::Cubic)
            {
                x += timeTrends(periodIndex, 3) * coefficients(3, k);
            }

            coords(k) = x;
        }

        return coords;
    }

    /**
     * Calcula distancias al cuadrado y lineales hacia puntos Yes/No.
     * Geometria DW-NOMINATE:
     * - Punto Yes = zmid - dyn
     * - Punto No = zmid + dyn
     * Optimizacion: Usa raw arrays para evitar heap allocations.
     */
    void computeDistancesOpt(
        const Eigen::VectorXd &coords,
        const double *midpoint,
        const double *spread,
        int numDim,
        double *dyesSquared,
        double *dnoSquared,
        double *dyesLinear,
        double *dnoLinear)
    {
        for (int k = 0; k < numDim; ++k)
        {
            // Punto Yes = zmid - dyn
            double diffYes = coords(k) - midpoint[k] + spread[k];
            // Punto No = zmid + dyn
            double diffNo = coords(k) - midpoint[k] - spread[k];

            dyesSquared[k] = diffYes * diffYes;
            dnoSquared[k] = diffNo * diffNo;
            dyesLinear[k] = diffYes;
            dnoLinear[k] = diffNo;
        }
    }

    /**
     * Calcula utilidades ponderadas para ambas opciones de voto.
     * Optimizacion: Usa raw arrays y pesos pre-calculados.
     */
    std::pair<double, double> computeUtilitiesOpt(
        const double *dyesSquared,
        const double *dnoSquared,
        const double *weightsSquared,
        int numDim,
        bool voteYes)
    {
        double utilityChosen = 0.0;
        double utilityOther = 0.0;

        for (int k = 0; k < numDim; ++k)
        {
            double wk2 = weightsSquared[k];
            if (voteYes)
            {
                utilityChosen -= wk2 * dyesSquared[k];
                utilityOther -= wk2 * dnoSquared[k];
            }
            else
            {
                utilityChosen -= wk2 * dnoSquared[k];
                utilityOther -= wk2 * dyesSquared[k];
            }
        }

        return {utilityChosen, utilityOther};
    }

    /**
     * Calcula derivadas base respecto a coordenadas del legislador.
     * Optimizacion: Usa raw arrays, pesos pre-calculados, escribe directamente en array de salida.
     */
    void computeBaseDerivativesOpt(
        const double *dyesLinear,
        const double *dnoLinear,
        const double *weightsSquared,
        double expDC,
        double expDB,
        double millsRatio,
        bool voteYes,
        int numDim,
        double *derivs)
    {
        for (int k = 0; k < numDim; ++k)
        {
            double wk2 = weightsSquared[k];
            double dcc1, dbb1;

            if (voteYes)
            {
                dcc1 = dyesLinear[k] * wk2;
                dbb1 = dnoLinear[k] * wk2;
            }
            else
            {
                dcc1 = dnoLinear[k] * wk2;
                dbb1 = dyesLinear[k] * wk2;
            }

            derivs[k] = millsRatio * (dcc1 * expDC - dbb1 * expDB);
        }
    }

    /**
     * Expande derivadas base a derivadas para modelo temporal.
     * Optimizacion: Usa raw arrays.
     */
    void expandDerivativesForModelOpt(
        const double *baseDerivs,
        const TimeTrends &timeTrends,
        int periodIndex,
        TemporalModel model,
        int numDim,
        double *expanded)
    {
        // Derivadas respecto a beta_0 (constante)
        for (int k = 0; k < numDim; ++k)
        {
            expanded[k] = baseDerivs[k];
        }

        // Derivadas respecto a beta_1 (lineal)
        if (model >= TemporalModel::Linear)
        {
            double t = timeTrends(periodIndex, 1);
            for (int k = 0; k < numDim; ++k)
            {
                expanded[numDim + k] = t * baseDerivs[k];
            }
        }

        // Derivadas respecto a beta_2 (cuadratico)
        if (model >= TemporalModel::Quadratic)
        {
            double t2 = timeTrends(periodIndex, 2);
            for (int k = 0; k < numDim; ++k)
            {
                expanded[2 * numDim + k] = t2 * baseDerivs[k];
            }
        }

        // Derivadas respecto a beta_3 (cubico)
        if (model >= TemporalModel::Cubic)
        {
            double t3 = timeTrends(periodIndex, 3);
            for (int k = 0; k < numDim; ++k)
            {
                expanded[3 * numDim + k] = t3 * baseDerivs[k];
            }
        }
    }

    /**
     * Actualiza la matriz de informacion con producto externo.
     * Optimizacion: Usa raw arrays.
     */
    void updateInfoMatrixOpt(
        Eigen::MatrixXd &infoMatrix,
        const double *derivs,
        int size)
    {
        for (int j = 0; j < size; ++j)
        {
            for (int i = 0; i < size; ++i)
            {
                infoMatrix(i, j) += derivs[j] * derivs[i];
            }
        }
    }

} // namespace anonimo

// Implementacion de computeLegislatorDerivatives
LegislatorDerivativesResult computeLegislatorDerivatives(
    int legislatorIndex,
    const LegislatorPeriodInfo &periodInfo,
    const TimeTrends &timeTrends,
    const TemporalCoefficients &coefficients,
    const Eigen::MatrixXd &rollCallMidpoints,
    const Eigen::MatrixXd &rollCallSpreads,
    const VoteMatrix &votes,
    const std::vector<bool> &validRollCalls,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    TemporalModel model,
    int firstPeriod,
    int lastPeriod)
{
    int numDim = static_cast<int>(weights.size()) - 1;
    double beta = weights(numDim);

    // Opt: Pre-calcular pesos al cuadrado
    std::array<double, MAX_DIMENSIONS> weightsSquared;
    for (int k = 0; k < numDim && k < MAX_DIMENSIONS; ++k)
    {
        weightsSquared[k] = weights(k) * weights(k);
    }

    // Contar periodos en que sirvio el legislador
    int numPeriodsServed = 0;
    for (int j = firstPeriod; j <= lastPeriod; ++j)
    {
        if (periodInfo.servedIn(j))
        {
            ++numPeriodsServed;
        }
    }

    LegislatorDerivativesResult result(numDim, numPeriodsServed);

    // Calcular offset inicial de roll calls
    int rollCallOffset = 0;
    for (int j = 0; j < firstPeriod; ++j)
    {
        rollCallOffset += periodInfo.rollCallCounts[j];
    }

    // Construir lista de periodos en que sirvio (MARK en Fortran)
    std::vector<PeriodInfo> servedPeriods;
    servedPeriods.reserve(numPeriodsServed);

    int currentOffset = rollCallOffset;
    int kk = 0; // Contador de periodos servidos

    for (int j = firstPeriod; j <= lastPeriod; ++j)
    {
        int nq = periodInfo.rollCallCounts[j];

        if (periodInfo.servedIn(j))
        {
            PeriodInfo pi;
            pi.dataIndex = periodInfo.dataIndices[j];
            pi.numRollCalls = nq;
            pi.rollCallOffset = currentOffset;
            pi.periodIndex = kk;
            servedPeriods.push_back(pi);

            // Reconstruir coordenadas para este periodo
            Eigen::VectorXd coords = reconstructCoordinates(
                coefficients, timeTrends, kk, model, numDim);

            // Guardar en periodCoordinates (XMARK)
            result.periodCoordinates.row(kk) = coords.transpose();

            ++kk;
        }

        currentOffset += nq;
    }

    // Opt: Variables temporales en stack (evita heap allocations)
    std::array<double, MAX_DIMENSIONS> dyesSquared;
    std::array<double, MAX_DIMENSIONS> dnoSquared;
    std::array<double, MAX_DIMENSIONS> dyesLinear;
    std::array<double, MAX_DIMENSIONS> dnoLinear;
    std::array<double, MAX_DIMENSIONS> baseDerivs;
    std::array<double, 4 * MAX_DIMENSIONS> expandedDerivs; // 4*NS para modelo cubico
    std::array<double, MAX_DIMENSIONS> midpoint_arr;
    std::array<double, MAX_DIMENSIONS> spread_arr;

    // Iterar sobre cada periodo en que sirvio el legislador
    for (size_t periodIdx = 0; periodIdx < servedPeriods.size(); ++periodIdx)
    {
        const PeriodInfo &pi = servedPeriods[periodIdx];
        int mwhere = pi.dataIndex;
        int nq = pi.numRollCalls;
        int ktotq = pi.rollCallOffset;
        int jj = pi.periodIndex;

        // Coordenadas del legislador en este periodo
        Eigen::VectorXd coords = result.periodCoordinates.row(jj).transpose();

        // Log-likelihood de este periodo
        double periodLL = 0.0;
        int periodVoteCount = 0;
        int periodErrorCount = 0;

        // Iterar sobre votaciones en este periodo
        for (int j = 0; j < nq; ++j)
        {
            int globalRC = ktotq + j;

            // Verificar si la votacion es valida (RCBAD)
            if (globalRC >= static_cast<int>(validRollCalls.size()) ||
                !validRollCalls[globalRC])
            {
                continue;
            }

            // Verificar si el voto es missing - VERSION SIN BOUNDS CHECK
            if (votes.isMissingUnsafe(mwhere, globalRC))
            {
                continue;
            }

            // Opt: Obtener parametros del roll call sin heap allocation
            for (int k = 0; k < numDim; ++k)
            {
                midpoint_arr[k] = rollCallMidpoints(globalRC, k);
                spread_arr[k] = rollCallSpreads(globalRC, k);
            }

            // Calcular distancias (version optimizada con raw arrays)
            computeDistancesOpt(coords, midpoint_arr.data(), spread_arr.data(), numDim,
                                dyesSquared.data(), dnoSquared.data(),
                                dyesLinear.data(), dnoLinear.data());

            // Obtener voto - VERSION SIN BOUNDS CHECK
            bool voteYes = votes.getVoteUnsafe(mwhere, globalRC);

            // Calcular utilidades (version optimizada con pesos pre-calculados)
            auto [utilChosen, utilOther] = computeUtilitiesOpt(
                dyesSquared.data(), dnoSquared.data(), weightsSquared.data(), numDim, voteYes);

            // Calcular ZS (diferencia de utilidades escalada)
            double expDC = std::exp(utilChosen);
            double expDB = std::exp(utilOther);
            double zs = beta * (expDC - expDB);

            // Clasificacion
            if (std::abs(utilChosen) <= std::abs(utilOther))
            {
                result.correctClassified++;
            }
            if (zs > 0.0)
            {
                result.positiveZS++;
            }

            // Opt: Una sola búsqueda para logCdf y millsRatio
            auto [logCdf, millsRatio] = normalCDF.logCdfAndMills(zs);

            // Actualizar log-likelihood
            result.logLikelihood += logCdf;
            periodLL += logCdf;
            result.totalVotes++;
            periodVoteCount++;

            // Contar errores (prediccion incorrecta)
            if (zs < 0.0)
            {
                periodErrorCount++;
            }

            // Calcular derivadas base 
            computeBaseDerivativesOpt(
                dyesLinear.data(), dnoLinear.data(), weightsSquared.data(),
                expDC, expDB, millsRatio, voteYes, numDim, baseDerivs.data());

            // Acumular derivadas por termino temporal
            for (int k = 0; k < numDim; ++k)
            {
                result.derivatives0(k) += baseDerivs[k];
                result.derivatives1(k) += timeTrends(jj, 1) * baseDerivs[k];
                result.derivatives2(k) += timeTrends(jj, 2) * baseDerivs[k];
                result.derivatives3(k) += timeTrends(jj, 3) * baseDerivs[k];
            }

            // Expandir derivadas segun modelo temporal
            expandDerivativesForModelOpt(
                baseDerivs.data(), timeTrends, jj, model, numDim, expandedDerivs.data());

            // Siempre actualizar OUTX0 (modelo constante)
            updateInfoMatrixOpt(result.infoMatrix0, expandedDerivs.data(), numDim);

            // Actualizar segun modelo
            if (model >= TemporalModel::Linear)
            {
                updateInfoMatrixOpt(result.infoMatrix1, expandedDerivs.data(), 2 * numDim);
            }
            if (model >= TemporalModel::Quadratic)
            {
                updateInfoMatrixOpt(result.infoMatrix2, expandedDerivs.data(), 3 * numDim);
            }
            if (model >= TemporalModel::Cubic)
            {
                updateInfoMatrixOpt(result.infoMatrix3, expandedDerivs.data(), 4 * numDim);
            }
        }

        // Guardar estadisticas del periodo
        if (periodIdx < result.periodLogLikelihood.size())
        {
            result.periodLogLikelihood[periodIdx] = periodLL;
            result.periodVotes[periodIdx] = periodVoteCount;
            result.periodErrors[periodIdx] = periodErrorCount;
        }
    }

    // Calcular GMP
    if (result.totalVotes > 0)
    {
        result.geometricMeanProb = std::exp(
            result.logLikelihood / static_cast<double>(result.totalVotes));
    }

    return result;
}
