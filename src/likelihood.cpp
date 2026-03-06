#include "likelihood.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <limits>

/**
 * @file likelihood.cpp
 *  Implementacion del calculo de log-likelihood para DW-NOMINATE.
 */

// Implementacion de VoteMatrix
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

// Funciones auxiliares para calculo de distancias
/**
 * Calcula distancia cuadrada ponderada entre coordenada y midpoint+offset.
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
 * Calcula utilidad ponderada (negativo de distancia ponderada).
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

    for (int k = 0; k < squaredDistances.size(); ++k)
    {
        double wk = weights(k);
        utility += -(wk * wk) * squaredDistances(k);
    }

    return utility;
}

// Implementacion de computeLogLikelihood
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

    for (size_t i = 0; i < numLegislators; ++i)
    {
        double legislatorLogLikelihood = 0.0;
        int legislatorValidVotes = 0;
        int legislatorWrongPredictions = 0;

        const Eigen::VectorXd &coord = legislatorCoords.row(i);

        for (size_t j = 0; j < numRollCalls; ++j)
        {
            if (!validRollCalls[j])
            {
                continue; // Skip votaciones invalidas (<2.5% en minoria)
            }

            // VERSION SIN BOUNDS CHECK para hot loop
            if (votes.isMissingUnsafe(i, j))
            {
                continue; // Skip missing data (abstencion/ausencia)
            }

            // Calcula distancias cuadradas
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

            // Determina cual es el voto observado y cual el contrario - VERSION SIN BOUNDS CHECK
            bool observedVote = votes.getVoteUnsafe(i, j);
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

// ===========================================================================
// IMPLEMENTACION OPTIMIZADA - CORRECCIONES A, B, C, E
// ===========================================================================
//
// CORRECCION A: Vectores de tamano fijo (elimina heap allocations)
// CORRECCION B: Buffer de trabajo reutilizable
// CORRECCION C: Pesos al cuadrado pre-cacheados
// CORRECCION E: Calculos inline de distancia y utilidad
//
// Logica matematica: INTACTA - Solo optimizacion estructural
// ===========================================================================

LikelihoodResult computeLogLikelihoodOptimized(
    const Eigen::MatrixXd &legislatorCoords,
    const std::vector<RollCallParameters> &rollCallParams,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    const std::vector<bool> &validRollCalls,
    LikelihoodWorkBuffer &buffer)
{
    // Validacion de dimensiones
    const size_t numLegislators = legislatorCoords.rows();
    const int numDimensions = static_cast<int>(legislatorCoords.cols());
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
    if (static_cast<size_t>(weights.size()) != static_cast<size_t>(numDimensions + 1))
    {
        throw std::invalid_argument("Dimension de weights incorrecta (debe ser NS+1)");
    }
    if (numDimensions > MAX_DIMENSIONS)
    {
        throw std::invalid_argument("Numero de dimensiones excede MAX_DIMENSIONS");
    }

    // CORRECCION C: Cachear pesos al cuadrado UNA vez
    buffer.cacheWeights(weights, numDimensions);

    // Inicializacion de resultado
    LikelihoodResult result;
    result.logLikelihood = 0.0;
    result.legislatorLL.resize(numLegislators, 0.0);
    result.legislatorVotes.resize(numLegislators, 0);
    result.legislatorErrors.resize(numLegislators, 0);

    // Extrae beta (ultimo peso)
    const double beta = weights(numDimensions);

    for (size_t i = 0; i < numLegislators; ++i)
    {
        double legislatorLogLikelihood = 0.0;
        int legislatorValidVotes = 0;
        int legislatorWrongPredictions = 0;

        for (size_t j = 0; j < numRollCalls; ++j)
        {
            if (!validRollCalls[j])
            {
                continue;
            }

            // VERSION SIN BOUNDS CHECK para hot loop
            if (votes.isMissingUnsafe(i, j))
            {
                continue;
            }

            // CORRECCION E: Calculos inline de distancia
            // Sin allocation de VectorXd, usando arrays del buffer
            const auto &midpoint = rollCallParams[j].midpoint;
            const auto &spread = rollCallParams[j].spread;

            // Calcular distancias cuadradas inline (CORRECCION E)
            for (int k = 0; k < numDimensions; ++k)
            {
                double coord_k = legislatorCoords(i, k);
                double diffYes = coord_k - midpoint(k) + spread(k);
                double diffNo = coord_k - midpoint(k) - spread(k);
                buffer.distYes[k] = diffYes * diffYes;
                buffer.distNo[k] = diffNo * diffNo;
            }

            // Determina cual es el voto observado y cual el contrario - VERSION SIN BOUNDS CHECK
            bool observedVote = votes.getVoteUnsafe(i, j);

            // CORRECCION E: Calcular utilidades inline usando pesos cacheados
            double utilityChoice = 0.0;
            double utilityOpposite = 0.0;
            double xcc;

            if (observedVote)
            {
                for (int k = 0; k < numDimensions; ++k)
                {
                    // CORRECCION C: Usar pesos pre-cacheados
                    utilityChoice += -buffer.weightsSquared[k] * buffer.distYes[k];
                    utilityOpposite += -buffer.weightsSquared[k] * buffer.distNo[k];
                }
                xcc = +1.0;
            }
            else
            {
                for (int k = 0; k < numDimensions; ++k)
                {
                    utilityChoice += -buffer.weightsSquared[k] * buffer.distNo[k];
                    utilityOpposite += -buffer.weightsSquared[k] * buffer.distYes[k];
                }
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
                result.stats.truePositives++;
            }
            else if (!correctClassification && xcc == +1.0)
            {
                result.stats.falseNegatives++;
            }
            else if (!correctClassification && xcc == -1.0)
            {
                result.stats.falsePositives++;
            }
            else if (correctClassification && xcc == -1.0)
            {
                result.stats.trueNegatives++;
            }

            if (zs > 0.0)
            {
                result.stats.positiveUtility++;
            }

            // Lookup en tabla CDF precomputada
            double logCdfValue = normalCDF.logCdf(zs);

            if (zs < 0.0)
            {
                legislatorWrongPredictions++;
            }

            // Acumula log-likelihood
            result.logLikelihood += logCdfValue;
            legislatorLogLikelihood += logCdfValue;
            legislatorValidVotes++;
        }

        result.legislatorLL[i] = legislatorLogLikelihood;
        result.legislatorVotes[i] = legislatorValidVotes;
        result.legislatorErrors[i] = legislatorWrongPredictions;
    }

    return result;
}
