#ifndef LIKELIHOOD_HPP
#define LIKELIHOOD_HPP

#include "normal_cdf.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <array>

/**
 * Estructuras y funciones para calcular log-likelihood del modelo DW-NOMINATE
 *
 * OPTIMIZACIONES IMPLEMENTADAS:
 * - Correccion A: Vectores de tamano fijo (stack allocation)
 * - Correccion B: Buffers de trabajo pre-allocated
 * - Correccion C: Pesos al cuadrado cacheados
 * - Correccion E: Calculos inline de distancia
 */

// ===========================================================================
// CORRECCION A: Constante maxima de dimensiones para arrays en stack
// ===========================================================================
constexpr int MAX_DIMENSIONS = 4; // DW-NOMINATE tipicamente usa 1-2 dimensiones

/**
 * Buffer de trabajo pre-allocated para evitar allocations dinamicas.
 * CORRECCION B: Se reutiliza en cada llamada a computeLogLikelihood.
 */
struct LikelihoodWorkBuffer
{
    // CORRECCION A: Arrays de tamano fijo en stack
    std::array<double, MAX_DIMENSIONS> distYes;
    std::array<double, MAX_DIMENSIONS> distNo;
    std::array<double, MAX_DIMENSIONS> weightsSquared; // CORRECCION C: Pesos cacheados
    int numDimensions;

    LikelihoodWorkBuffer() : numDimensions(0)
    {
        distYes.fill(0.0);
        distNo.fill(0.0);
        weightsSquared.fill(0.0);
    }

    // CORRECCION C: Pre-calcular pesos al cuadrado
    void cacheWeights(const Eigen::VectorXd &weights, int ns)
    {
        numDimensions = ns;
        for (int k = 0; k < ns && k < MAX_DIMENSIONS; ++k)
        {
            weightsSquared[k] = weights(k) * weights(k);
        }
    }
};

/**
 *  Estructura para almacenar parametros de una votacion (roll call)
 */
struct RollCallParameters
{
    Eigen::VectorXd midpoint; // ZMID: punto medio del plano de corte
    Eigen::VectorXd spread;   // DYN: dispersion/spread del plano de corte

    RollCallParameters(int dimensions)
        : midpoint(Eigen::VectorXd::Zero(dimensions)),
          spread(Eigen::VectorXd::Zero(dimensions))
    {
    }
};

/**
 * Matriz de votos con manejo de missing data.
 *
 * Equivalente a RCVOTE1 (voto observado) y RCVOTE9 (missing data mask) en Fortran
 */
class VoteMatrix
{
public:
    /**
     * Constructor con dimensiones
     * @param numLegislators Numero de legisladores
     * @param numRollCalls Numero de votaciones
     */
    VoteMatrix(size_t numLegislators, size_t numRollCalls);

    /**
     * Establece un voto.
     * @param legislator Indice del legislador (0-based)
     * @param rollCall Indice de la votacion (0-based)
     * @param vote true=Si, false=No
     * @param isMissing true si el voto es missing (abstencion/ausencia)
     */
    void setVote(size_t legislator, size_t rollCall, bool vote, bool isMissing = false);

    /**
     * Obtiene el voto observado.
     * @return true=Si, false=No
     */
    bool getVote(size_t legislator, size_t rollCall) const;

    /**
     * Verifica si el voto es missing data.
     */
    bool isMissing(size_t legislator, size_t rollCall) const;

    /**
     * VERSIONES SIN BOUNDS CHECKING - OPTIMIZADAS PARA HOT LOOPS
     * Solo usar cuando los índices están pre-validados.
     */
    inline bool getVoteUnsafe(size_t legislator, size_t rollCall) const
    {
        return votes_[legislator * numRollCalls_ + rollCall];
    }

    inline bool isMissingUnsafe(size_t legislator, size_t rollCall) const
    {
        return missingData_[legislator * numRollCalls_ + rollCall];
    }

    size_t getNumLegislators() const { return numLegislators_; }
    size_t getNumRollCalls() const { return numRollCalls_; }

    /**
     * Acceso directo a datos para optimización avanzada.
     */
    const std::vector<bool> &getVotesRaw() const { return votes_; }
    const std::vector<bool> &getMissingRaw() const { return missingData_; }

private:
    size_t numLegislators_;
    size_t numRollCalls_;
    std::vector<bool> votes_;       // RCVOTE1: votos observados (flat array)
    std::vector<bool> missingData_; // RCVOTE9: mascara de missing data

    size_t index(size_t leg, size_t rc) const
    {
        return leg * numRollCalls_ + rc;
    }
};

/**
 * Estadisticas de clasificacion del modelo.
 *
 * Equivalente a KLASS, KLASSYY, etc. en Fortran.
 */
struct ClassificationStats
{
    int totalVotes;        // KTOT: total de votos validos
    int correctClassified; // KLASS: votos correctamente clasificados
    int truePositives;     // KLASSYY: predijo Si, fue Si
    int falseNegatives;    // KLASSNY: predijo No, fue Si
    int falsePositives;    // KLASSYN: predijo Si, fue No
    int trueNegatives;     // KLASSNN: predijo No, fue No
    int positiveUtility;   // KLASS2: utilidad diferencial positiva

    ClassificationStats()
        : totalVotes(0), correctClassified(0), truePositives(0),
          falseNegatives(0), falsePositives(0), trueNegatives(0),
          positiveUtility(0)
    {
    }

    double getAccuracy() const
    {
        return totalVotes > 0 ? static_cast<double>(correctClassified) / totalVotes : 0.0;
    }

    double getGeometricMeanProbability(double logLikelihood) const
    {
        return totalVotes > 0 ? std::exp(logLikelihood / totalVotes) : 0.0;
    }
};

/**
 * Resultado del calculo de log-likelihood.
 */
struct LikelihoodResult
{
    double logLikelihood;              // XXPLOG: log-likelihood total
    ClassificationStats stats;         // Estadisticas de clasificacion
    std::vector<double> legislatorLL;  // XBIGLOG(i,2): LL por legislador
    std::vector<int> legislatorVotes;  // KBIGLOG(i,2): votos validos por legislador
    std::vector<int> legislatorErrors; // KBIGLOG(i,4): errores graves por legislador
};

/**
 * Calcula log-likelihood del modelo probit espacial.
 *
 * @param legislatorCoords XDATA: Coordenadas ideales de legisladores (numLeg x numDim)
 * @param rollCallParams Vector de parametros por votacion (ZMID, DYN)
 * @param votes Matriz de votos observados
 * @param weights WEIGHT: Pesos dimensionales [w1, w2, ..., wNS, beta]
 * @param normalCDF Tabla CDF precomputada
 * @param validRollCalls RCBAD: Mascara de votaciones validas (>2.5% en minoria)
 * @return Resultado con log-likelihood y estadisticas
 *
 */
LikelihoodResult computeLogLikelihood(
    const Eigen::MatrixXd &legislatorCoords,
    const std::vector<RollCallParameters> &rollCallParams,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    const std::vector<bool> &validRollCalls);

// ===========================================================================
// VERSION OPTIMIZADA - CORRECCIONES A, B, C, E
// ===========================================================================

/**
 * Calcula log-likelihood usando optimizaciones estructurales.
 *
 * OPTIMIZADO: Elimina allocations dinamicas en hot loops.
 * Usa arrays de tamano fijo y pesos pre-cacheados.
 *
 * @param legislatorCoords Coordenadas de legisladores
 * @param rollCallParams Parametros de roll calls
 * @param votes Matriz de votos
 * @param weights Pesos dimensionales [w1..wNS, beta]
 * @param normalCDF Tabla CDF
 * @param validRollCalls Mascara de roll calls validos
 * @param buffer Buffer de trabajo pre-allocated (reutilizable)
 * @return Resultado con log-likelihood y estadisticas
 */
LikelihoodResult computeLogLikelihoodOptimized(
    const Eigen::MatrixXd &legislatorCoords,
    const std::vector<RollCallParameters> &rollCallParams,
    const VoteMatrix &votes,
    const Eigen::VectorXd &weights,
    const NormalCDF &normalCDF,
    const std::vector<bool> &validRollCalls,
    LikelihoodWorkBuffer &buffer);

#endif // LIKELIHOOD_HPP
