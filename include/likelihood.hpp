#ifndef LIKELIHOOD_HPP
#define LIKELIHOOD_HPP

#include "normal_cdf.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>

/**
 * @brief Estructuras y funciones para calcular log-likelihood del modelo DW-NOMINATE.
 */

/**
 * @brief Estructura para almacenar parametros de una votacion (roll call).
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
 * @brief Matriz de votos con manejo de missing data.
 *
 * Equivalente a RCVOTE1 (voto observado) y RCVOTE9 (missing data mask) en Fortran.
 */
class VoteMatrix
{
public:
    /**
     * @brief Constructor con dimensiones.
     * @param numLegislators Numero de legisladores
     * @param numRollCalls Numero de votaciones
     */
    VoteMatrix(size_t numLegislators, size_t numRollCalls);

    /**
     * @brief Establece un voto.
     * @param legislator Indice del legislador (0-based)
     * @param rollCall Indice de la votacion (0-based)
     * @param vote true=Si, false=No
     * @param isMissing true si el voto es missing (abstencion/ausencia)
     */
    void setVote(size_t legislator, size_t rollCall, bool vote, bool isMissing = false);

    /**
     * @brief Obtiene el voto observado.
     * @return true=Si, false=No
     */
    bool getVote(size_t legislator, size_t rollCall) const;

    /**
     * @brief Verifica si el voto es missing data.
     */
    bool isMissing(size_t legislator, size_t rollCall) const;

    size_t getNumLegislators() const { return numLegislators_; }
    size_t getNumRollCalls() const { return numRollCalls_; }

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
 * @brief Estadisticas de clasificacion del modelo.
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
 * @brief Resultado del calculo de log-likelihood.
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
 * @brief Calcula log-likelihood del modelo probit espacial.
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

#endif // LIKELIHOOD_HPP
