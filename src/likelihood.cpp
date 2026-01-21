#include "likelihood.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

/**
 * @file likelihood.cpp
 * @brief Implementacion del calculo de log-likelihood para DW-NOMINATE.
 */

// ============================================================================
// Implementacion de VoteMatrix
// ============================================================================

VoteMatrix::VoteMatrix(size_t numLegislators, size_t numRollCalls)
    : numLegislators_(numLegislators),
      numRollCalls_(numRollCalls),
      votes_(numLegislators * numRollCalls, false),
      missingData_(numLegislators * numRollCalls, false)
{
}

void VoteMatrix::setVote(size_t legislator, size_t rollCall, bool vote, bool isMissing)
{
    if (legislator >= numLegislators_ || rollCall >= numRollCalls_)
    {
        throw std::out_of_range("Indices fuera de rango en VoteMatrix");
    }

    size_t idx = index(legislator, rollCall);
    votes_[idx] = vote;
    missingData_[idx] = isMissing;
}

bool VoteMatrix::getVote(size_t legislator, size_t rollCall) const
{
    if (legislator >= numLegislators_ || rollCall >= numRollCalls_)
    {
        throw std::out_of_range("Indices fuera de rango en VoteMatrix");
    }
    return votes_[index(legislator, rollCall)];
}

bool VoteMatrix::isMissing(size_t legislator, size_t rollCall) const
{
    if (legislator >= numLegislators_ || rollCall >= numRollCalls_)
    {
        throw std::out_of_range("Indices fuera de rango en VoteMatrix");
    }
    return missingData_[index(legislator, rollCall)];
}

// ============================================================================
// Funciones auxiliares para calculo de distancias
// ============================================================================

/**
 * @brief Calcula distancia cuadrada ponderada entre coordenada y midpoint+offset.
 *
 * @param coord Coordenada del legislador
 * @param midpoint Punto medio de la votacion
 * @param spread Dispersion de la votacion
 * @param usePositiveSpread true para +spread (voto Si), false para -spread (voto No)
 * @return Vector de distancias cuadradas por dimension
 */
static Eigen::VectorXd computeSquaredDistances(
    const Eigen::VectorXd &coord,
    const Eigen::VectorXd &midpoint,
    const Eigen::VectorXd &spread,
    bool usePositiveSpread)
{
    Eigen::VectorXd distances(coord.size());

    for (int k = 0; k < coord.size(); ++k)
    {
        double diff;
        if (usePositiveSpread)
        {
            // DYES: distancia al midpoint Si (midpoint + spread)
            diff = coord(k) - midpoint(k) + spread(k);
        }
        else
        {
            // DNO: distancia al midpoint No (midpoint - spread)
            diff = coord(k) - midpoint(k) - spread(k);
        }
        distances(k) = diff * diff;
    }

    return distances;
}

/**
 * @brief Calcula utilidad ponderada (negativo de distancia ponderada).
 *
 * @param squaredDistances Distancias cuadradas por dimension
 * @param weights Pesos dimensionales (solo primeras NS componentes)
 * @return Utilidad (escalar)
 */
static double computeUtility(
    const Eigen::VectorXd &squaredDistances,
    const Eigen::VectorXd &weights)
{
    double utility = 0.0;

    // Fortran: DC = DC + (-WEIGHT(K)*WEIGHT(K)*DYES(K))
    for (int k = 0; k < squaredDistances.size(); ++k)
    {
        double wk = weights(k);
        utility += -(wk * wk) * squaredDistances(k);
    }

    return utility;
}

// ============================================================================
// Implementacion de computeLogLikelihood
// ============================================================================

LikelihoodResult computeLogLikelihood(
    const Eigen::MatrixXd &legislatorCoords,
    const std::vector<RollCallParameters> &rollCallParams,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    const std::vector<bool> &validRollCalls)
{
    // Validacion de dimensiones
    const size_t numLegislators = legislatorCoords.rows();
    const size_t numDimensions = legislatorCoords.cols();
    const size_t numRollCalls = rollCallParams.size();

    if (votes.getNumLegislators() != numLegislators)
    {
        throw std::invalid_argument("Dimension de legisladores inconsistente");
    }
    if (votes.getNumRollCalls() != numRollCalls)
    {
        throw std::invalid_argument("Dimension de votaciones inconsistente");
    }
    if (validRollCalls.size() != numRollCalls)
    {
        throw std::invalid_argument("Dimension de validRollCalls inconsistente");
    }
    if (static_cast<size_t>(weights.size()) != numDimensions + 1)
    {
        throw std::invalid_argument("Dimension de weights incorrecta (debe ser NS+1)");
    }

    // Inicializacion de resultado
    LikelihoodResult result;
    result.logLikelihood = 0.0;
    result.legislatorLL.resize(numLegislators, 0.0);
    result.legislatorVotes.resize(numLegislators, 0);
    result.legislatorErrors.resize(numLegislators, 0);

    // Extrae beta (ultimo peso)
    const double beta = weights(numDimensions); // WEIGHT(NS+1)

    // Fortran: DO 1 II=NFIRST,NLAST (periodos)
    //          DO 2 I=1,NPC (legisladores)
    //          DO 33 J=1,NQC (votaciones)

    for (size_t i = 0; i < numLegislators; ++i)
    {
        double legislatorLogLikelihood = 0.0;
        int legislatorValidVotes = 0;
        int legislatorWrongPredictions = 0;

        const Eigen::VectorXd &coord = legislatorCoords.row(i);

        for (size_t j = 0; j < numRollCalls; ++j)
        {
            // Fortran: IF(RCBAD(J+KTOTQ).EQV..TRUE.)THEN
            if (!validRollCalls[j])
            {
                continue; // Skip votaciones invalidas (<2.5% en minoria)
            }

            // Fortran: IF(RCVOTE9(I+KTOTP,J).EQV..FALSE.)THEN
            if (votes.isMissing(i, j))
            {
                continue; // Skip missing data (abstencion/ausencia)
            }

            // Calcula distancias cuadradas
            // Fortran: DYES(K) = (XDATA(i,k) - ZMID(j,k) + DYN(j,k))^2
            //          DNO(K)  = (XDATA(i,k) - ZMID(j,k) - DYN(j,k))^2
            Eigen::VectorXd distYes = computeSquaredDistances(
                coord,
                rollCallParams[j].midpoint,
                rollCallParams[j].spread,
                true); // +spread

            Eigen::VectorXd distNo = computeSquaredDistances(
                coord,
                rollCallParams[j].midpoint,
                rollCallParams[j].spread,
                false); // -spread

            // Determina cual es el voto observado y cual el contrario
            bool observedVote = votes.getVote(i, j);
            double utilityChoice, utilityOpposite;
            double xcc; // +1 si voto Si, -1 si voto No

            if (observedVote) // Voto = Si
            {
                utilityChoice = computeUtility(distYes, weights.head(numDimensions));
                utilityOpposite = computeUtility(distNo, weights.head(numDimensions));
                xcc = +1.0;
            }
            else // Voto = No
            {
                utilityChoice = computeUtility(distNo, weights.head(numDimensions));
                utilityOpposite = computeUtility(distYes, weights.head(numDimensions));
                xcc = -1.0;
            }

            // Calcula Z_ij segun el modelo probit
            double zs = beta * (std::exp(utilityChoice) - std::exp(utilityOpposite));

            // Estadisticas de clasificacion deterministica
            bool correctClassification = (std::abs(utilityChoice) <= std::abs(utilityOpposite));

            result.stats.totalVotes++;
            if (correctClassification)
            {
                result.stats.correctClassified++;
            }

            // Tabla de confusion 2x2
            if (correctClassification && xcc == +1.0)
            {
                result.stats.truePositives++; // KLASSYY
            }
            else if (!correctClassification && xcc == +1.0)
            {
                result.stats.falseNegatives++; // KLASSNY
            }
            else if (!correctClassification && xcc == -1.0)
            {
                result.stats.falsePositives++; // KLASSYN
            }
            else if (correctClassification && xcc == -1.0)
            {
                result.stats.trueNegatives++; // KLASSNN
            }

            if (zs > 0.0)
            {
                result.stats.positiveUtility++; // KLASS2
            }

            // Lookup en tabla CDF precomputada
            double cdfValue, logCdfValue;

            if (zs >= 0.0)
            {
                cdfValue = normalCDF.cdf(zs);
                logCdfValue = normalCDF.logCdf(zs);
            }
            else
            {
                cdfValue = normalCDF.cdf(zs);
                logCdfValue = normalCDF.logCdf(zs);
                legislatorWrongPredictions++; // Probabilidad < 0.5
            }

            // Acumula log-likelihood
            result.logLikelihood += logCdfValue;
            legislatorLogLikelihood += logCdfValue;
            legislatorValidVotes++;
        }

        // Almacena estadisticas por legislador
        result.legislatorLL[i] = legislatorLogLikelihood;
        result.legislatorVotes[i] = legislatorValidVotes;
        result.legislatorErrors[i] = legislatorWrongPredictions;
    }

    // Fortran: XXPLOG = XXPLOG + YPLOG (ya acumulado en el loop)
    // XPLOG = XXPLOG
    // Ya esta en result.logLikelihood

    return result;
}
