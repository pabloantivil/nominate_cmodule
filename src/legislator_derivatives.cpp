/**
 * @brief Implementacion de computeLegislatorDerivatives (PROX).
 *
 * Calcula derivadas del log-likelihood respecto a las coordenadas espaciales de un legislador especifico, 
 * considerando modelos temporales.
 */

#include "legislator_derivatives.hpp"
#include <cmath>
#include <algorithm>

namespace
{

    /**
     * @brief Reconstruye coordenadas del legislador para un periodo segun modelo temporal.
     *
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
     * @brief Calcula distancias al cuadrado y lineales hacia puntos Yes/No.
     *
     * Geometria DW-NOMINATE:
     * - Punto Yes = zmid - dyn
     * - Punto No = zmid + dyn
     *
     * @param coords Coordenadas del legislador
     * @param midpoint Punto medio del roll call
     * @param spread Spread del roll call
     * @param numDim Numero de dimensiones
     * @param dyesSquared [out] Distancias al cuadrado hacia Yes
     * @param dnoSquared [out] Distancias al cuadrado hacia No
     * @param dyesLinear [out] Distancias lineales hacia Yes (para derivadas)
     * @param dnoLinear [out] Distancias lineales hacia No (para derivadas)
     */
    void computeDistances(
        const Eigen::VectorXd &coords,
        const Eigen::VectorXd &midpoint,
        const Eigen::VectorXd &spread,
        int numDim,
        Eigen::VectorXd &dyesSquared,
        Eigen::VectorXd &dnoSquared,
        Eigen::VectorXd &dyesLinear,
        Eigen::VectorXd &dnoLinear)
    {
        for (int k = 0; k < numDim; ++k)
        {
            // Punto Yes = zmid - dyn 
            double diffYes = coords(k) - midpoint(k) + spread(k);
            // Punto No = zmid + dyn
            double diffNo = coords(k) - midpoint(k) - spread(k);

            dyesSquared(k) = diffYes * diffYes;
            dnoSquared(k) = diffNo * diffNo;
            dyesLinear(k) = diffYes;
            dnoLinear(k) = diffNo;
        }
    }

    /**
     * @brief Calcula utilidades ponderadas para ambas opciones de voto.
     *
     * @param dyesSquared Distancias al cuadrado hacia Yes
     * @param dnoSquared Distancias al cuadrado hacia No
     * @param weights Vector de pesos
     * @param numDim Numero de dimensiones
     * @return Par (utilidadOpcionElegida, utilidadOpcionContraria)
     */
    std::pair<double, double> computeUtilities(
        const Eigen::VectorXd &dyesSquared,
        const Eigen::VectorXd &dnoSquared,
        const Eigen::VectorXd &weights,
        int numDim,
        bool voteYes)
    {
        double utilityChosen = 0.0;
        double utilityOther = 0.0;

        for (int k = 0; k < numDim; ++k)
        {
            double wk2 = weights(k) * weights(k);
            if (voteYes)
            {
                utilityChosen += -wk2 * dyesSquared(k);
                utilityOther += -wk2 * dnoSquared(k);
            }
            else
            {
                utilityChosen += -wk2 * dnoSquared(k);
                utilityOther += -wk2 * dyesSquared(k);
            }
        }

        return {utilityChosen, utilityOther};
    }

    /**
     * @brief Calcula derivadas base respecto a coordenadas del legislador.
     *
     * @param dyesLinear Distancias lineales hacia Yes
     * @param dnoLinear Distancias lineales hacia No
     * @param weights Pesos dimensionales
     * @param expDC exp(utilidad opcion elegida)
     * @param expDB exp(utilidad opcion contraria)
     * @param millsRatio phi(ZS)/Phi(ZS)
     * @param voteYes true si voto fue Si
     * @param numDim Numero de dimensiones
     * @return Vector de derivadas base (NS)
     */
    Eigen::VectorXd computeBaseDerivatives(
        const Eigen::VectorXd &dyesLinear,
        const Eigen::VectorXd &dnoLinear,
        const Eigen::VectorXd &weights,
        double expDC,
        double expDB,
        double millsRatio,
        bool voteYes,
        int numDim)
    {
        Eigen::VectorXd derivs(numDim);

        for (int k = 0; k < numDim; ++k)
        {
            double wk2 = weights(k) * weights(k);
            double dcc1, dbb1;

            if (voteYes)
            {
                dcc1 = dyesLinear(k) * wk2;
                dbb1 = dnoLinear(k) * wk2;
            }
            else
            {
                dcc1 = dnoLinear(k) * wk2;
                dbb1 = dyesLinear(k) * wk2;
            }

            derivs(k) = millsRatio * (dcc1 * expDC - dbb1 * expDB);
        }

        return derivs;
    }

    /**
     * @brief Expande derivadas base a derivadas para modelo temporal.
     *
     * @param baseDerivs Derivadas base (dL/dx_k)
     * @param timeTrends Valores temporales
     * @param periodIndex Indice del periodo
     * @param model Modelo temporal
     * @param numDim Numero de dimensiones
     * @return Vector de derivadas expandido
     */
    Eigen::VectorXd expandDerivativesForModel(
        const Eigen::VectorXd &baseDerivs,
        const TimeTrends &timeTrends,
        int periodIndex,
        TemporalModel model,
        int numDim)
    {
        int size = numDim;
        if (model == TemporalModel::Linear)
            size = 2 * numDim;
        else if (model == TemporalModel::Quadratic)
            size = 3 * numDim;
        else if (model == TemporalModel::Cubic)
            size = 4 * numDim;

        Eigen::VectorXd expanded(size);

        // Derivadas respecto a beta_0 (constante)
        for (int k = 0; k < numDim; ++k)
        {
            expanded(k) = baseDerivs(k);
        }

        // Derivadas respecto a beta_1 (lineal)
        if (model >= TemporalModel::Linear)
        {
            double t = timeTrends(periodIndex, 1);
            for (int k = 0; k < numDim; ++k)
            {
                expanded(numDim + k) = t * baseDerivs(k);
            }
        }

        // Derivadas respecto a beta_2 (cuadratico)
        if (model >= TemporalModel::Quadratic)
        {
            double t2 = timeTrends(periodIndex, 2);
            for (int k = 0; k < numDim; ++k)
            {
                expanded(2 * numDim + k) = t2 * baseDerivs(k);
            }
        }

        // Derivadas respecto a beta_3 (cubico)
        if (model >= TemporalModel::Cubic)
        {
            double t3 = timeTrends(periodIndex, 3);
            for (int k = 0; k < numDim; ++k)
            {
                expanded(3 * numDim + k) = t3 * baseDerivs(k);
            }
        }

        return expanded;
    }

    /**
     * @brief Actualiza la matriz de informacion con producto externo.
     *
     * @param infoMatrix Matriz a actualizar
     * @param derivs Vector de derivadas
     * @param size Dimension de la matriz a usar
     */
    void updateInfoMatrix(
        Eigen::MatrixXd &infoMatrix,
        const Eigen::VectorXd &derivs,
        int size)
    {
        for (int j = 0; j < size; ++j)
        {
            for (int i = 0; i < size; ++i)
            {
                infoMatrix(i, j) += derivs(j) * derivs(i);
            }
        }
    }

} // namespace anonimo

// ============================================================================
// Implementacion de computeLegislatorDerivatives
// ============================================================================

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

    // Variables temporales para calculo
    Eigen::VectorXd dyesSquared(numDim);
    Eigen::VectorXd dnoSquared(numDim);
    Eigen::VectorXd dyesLinear(numDim);
    Eigen::VectorXd dnoLinear(numDim);

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

            // Verificar si el voto es missing
            if (votes.isMissing(mwhere, globalRC))
            {
                continue;
            }

            // Obtener parametros del roll call
            Eigen::VectorXd midpoint = rollCallMidpoints.row(globalRC).transpose();
            Eigen::VectorXd spread = rollCallSpreads.row(globalRC).transpose();

            // Calcular distancias
            computeDistances(coords, midpoint, spread, numDim,
                             dyesSquared, dnoSquared, dyesLinear, dnoLinear);

            // Obtener voto
            bool voteYes = votes.getVote(mwhere, globalRC);

            // Calcular utilidades
            auto [utilChosen, utilOther] = computeUtilities(
                dyesSquared, dnoSquared, weights, numDim, voteYes);

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

            // Lookup CDF
            double logCdf = normalCDF.logCdf(zs);

            // Calcular Mills ratio
            double millsRatio = normalCDF.gaussOverCdf(zs);

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
            Eigen::VectorXd baseDerivs = computeBaseDerivatives(
                dyesLinear, dnoLinear, weights, expDC, expDB, millsRatio,
                voteYes, numDim);

            // Acumular derivadas por termino temporal
            for (int k = 0; k < numDim; ++k)
            {
                result.derivatives0(k) += baseDerivs(k);
                result.derivatives1(k) += timeTrends(jj, 1) * baseDerivs(k);
                result.derivatives2(k) += timeTrends(jj, 2) * baseDerivs(k);
                result.derivatives3(k) += timeTrends(jj, 3) * baseDerivs(k);
            }

            // Expandir derivadas segun modelo temporal
            Eigen::VectorXd expandedDerivs = expandDerivativesForModel(
                baseDerivs, timeTrends, jj, model, numDim);

            // Actualizar matrices de informacion
            // Siempre actualizar OUTX0 (modelo constante)
            updateInfoMatrix(result.infoMatrix0, expandedDerivs, numDim);

            // Actualizar segun modelo
            if (model >= TemporalModel::Linear)
            {
                updateInfoMatrix(result.infoMatrix1, expandedDerivs, 2 * numDim);
            }
            if (model >= TemporalModel::Quadratic)
            {
                updateInfoMatrix(result.infoMatrix2, expandedDerivs, 3 * numDim);
            }
            if (model >= TemporalModel::Cubic)
            {
                updateInfoMatrix(result.infoMatrix3, expandedDerivs, 4 * numDim);
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
