/**
 * Calculo de derivadas y log-likelihood para parametros de legisladores (PROX)
 */

#ifndef LEGISLATOR_DERIVATIVES_HPP
#define LEGISLATOR_DERIVATIVES_HPP

#include "likelihood.hpp"
#include "normal_cdf.hpp"
#include <Eigen/Dense>
#include <vector>

/**
 * Tipo de modelo temporal para coordenadas de legisladores.
 */
enum class TemporalModel
{
    Constant = 0,  // NMODEL=0: posicion fija en el tiempo
    Linear = 1,    // NMODEL=1: evolucion lineal
    Quadratic = 2, // NMODEL=2: evolucion cuadratica
    Cubic = 3      // NMODEL=3: evolucion cubica
};

/**
 * Informacion de un periodo (congreso) en que sirvio el legislador.
 */
struct PeriodInfo
{
    int dataIndex;      // MARK(kk,1) = MWHERE: indice en XDATA
    int numRollCalls;   // MARK(kk,2) = NQ: numero de votaciones en este periodo
    int rollCallOffset; // MARK(kk,3) = KTOTQ: offset a votaciones de este periodo
    int periodIndex;    // Indice del periodo (JJ en Fortran, 0-based)

    PeriodInfo()
        : dataIndex(0), numRollCalls(0), rollCallOffset(0), periodIndex(0)
    {
    }

    PeriodInfo(int idx, int nrc, int offset, int pIdx)
        : dataIndex(idx), numRollCalls(nrc), rollCallOffset(offset), periodIndex(pIdx)
    {
    }
};

/**
 * Resultado del calculo de derivadas para un legislador.
 */
struct LegislatorDerivativesResult
{
    // Log-likelihood
    double logLikelihood;     // XPLOG: suma de log(CDF) para este legislador
    double geometricMeanProb; // GMP: exp(XPLOG/KRC)
    int totalVotes;           // KRC: numero total de votos validos

    // Derivadas (vectores de dimension NS)
    // Derivadas respecto a cada termino del modelo temporal
    Eigen::VectorXd derivatives0; // XDERV: dL/d(beta_0)  - coeficientes constantes
    Eigen::VectorXd derivatives1; // XDERV1: dL/d(beta_1) - coeficientes lineales
    Eigen::VectorXd derivatives2; // XDERV2: dL/d(beta_2) - coeficientes cuadraticos
    Eigen::VectorXd derivatives3; // XDERV3: dL/d(beta_3) - coeficientes cubicos

    // Matrices de informacion (producto externo de derivadas)
    // Aproximacion a la matriz de Fisher, su inversa da errores estandar
    Eigen::MatrixXd infoMatrix0; // OUTX0: matriz NS x NS para modelo constante
    Eigen::MatrixXd infoMatrix1; // OUTX1: matriz 2NS x 2NS para modelo lineal
    Eigen::MatrixXd infoMatrix2; // OUTX2: matriz 3NS x 3NS para modelo cuadratico
    Eigen::MatrixXd infoMatrix3; // OUTX3: matriz 4NS x 4NS para modelo cubico

    // Coordenadas reconstruidas por periodo
    // XMARK(kk,k): coordenadas del legislador en cada periodo
    Eigen::MatrixXd periodCoordinates;

    // Estadisticas de clasificacion
    int correctClassified; // KLASS: clasificaciones correctas (|DC| <= |DB|)
    int positiveZS;        // KLASS2: votos con ZS > 0

    // Log-likelihood y estadisticas por periodo
    std::vector<double> periodLogLikelihood; // XBIGLOG(MWHERE,1)
    std::vector<int> periodVotes;            // KBIGLOG(MWHERE,1)
    std::vector<int> periodErrors;           // KBIGLOG(MWHERE,3)

    /**
     * Constructor por defecto.
     */
    LegislatorDerivativesResult()
        : logLikelihood(0.0),
          geometricMeanProb(0.0),
          totalVotes(0),
          correctClassified(0),
          positiveZS(0)
    {
    }

    /**
     * Constructor con dimension especificada.
     *
     * @param numDimensions Numero de dimensiones espaciales (NS)
     * @param numPeriods Numero de periodos en que sirvio el legislador
     */
    LegislatorDerivativesResult(int numDimensions, int numPeriods)
        : logLikelihood(0.0),
          geometricMeanProb(0.0),
          totalVotes(0),
          derivatives0(Eigen::VectorXd::Zero(numDimensions)),
          derivatives1(Eigen::VectorXd::Zero(numDimensions)),
          derivatives2(Eigen::VectorXd::Zero(numDimensions)),
          derivatives3(Eigen::VectorXd::Zero(numDimensions)),
          infoMatrix0(Eigen::MatrixXd::Zero(numDimensions, numDimensions)),
          infoMatrix1(Eigen::MatrixXd::Zero(2 * numDimensions, 2 * numDimensions)),
          infoMatrix2(Eigen::MatrixXd::Zero(3 * numDimensions, 3 * numDimensions)),
          infoMatrix3(Eigen::MatrixXd::Zero(4 * numDimensions, 4 * numDimensions)),
          periodCoordinates(Eigen::MatrixXd::Zero(numPeriods, numDimensions)),
          correctClassified(0),
          positiveZS(0),
          periodLogLikelihood(numPeriods, 0.0),
          periodVotes(numPeriods, 0),
          periodErrors(numPeriods, 0)
    {
    }

    /**
     * Calcula precision de clasificacion.
     */
    double getAccuracy() const
    {
        return totalVotes > 0
                   ? static_cast<double>(correctClassified) / totalVotes
                   : 0.0;
    }

    /**
     * Obtiene el vector de derivadas concatenado segun el modelo temporal.
     *
     * @param model Tipo de modelo temporal
     * @return Vector de derivadas con dimension apropiada
     */
    Eigen::VectorXd getDerivativesForModel(TemporalModel model) const
    {
        int ns = static_cast<int>(derivatives0.size());
        int size = 0;

        switch (model)
        {
        case TemporalModel::Constant:
            return derivatives0;
        case TemporalModel::Linear:
            size = 2 * ns;
            break;
        case TemporalModel::Quadratic:
            size = 3 * ns;
            break;
        case TemporalModel::Cubic:
            size = 4 * ns;
            break;
        }

        Eigen::VectorXd result(size);
        result.head(ns) = derivatives0;
        if (model >= TemporalModel::Linear)
        {
            result.segment(ns, ns) = derivatives1;
        }
        if (model >= TemporalModel::Quadratic)
        {
            result.segment(2 * ns, ns) = derivatives2;
        }
        if (model == TemporalModel::Cubic)
        {
            result.segment(3 * ns, ns) = derivatives3;
        }
        return result;
    }

    /**
     * Obtiene la matriz de informacion segun el modelo temporal.
     *
     * @param model Tipo de modelo temporal
     * @return Referencia a la matriz de informacion apropiada
     */
    const Eigen::MatrixXd &getInfoMatrixForModel(TemporalModel model) const
    {
        switch (model)
        {
        case TemporalModel::Constant:
            return infoMatrix0;
        case TemporalModel::Linear:
            return infoMatrix1;
        case TemporalModel::Quadratic:
            return infoMatrix2;
        case TemporalModel::Cubic:
            return infoMatrix3;
        }
        return infoMatrix0;
    }
};

/**
 * Coeficientes del modelo temporal para un legislador.
 */
struct TemporalCoefficients
{
    Eigen::MatrixXd beta; // (4 x NS): beta[i][k] = coeficiente i para dimension k

    /**
     * Constructor.
     * @param numDimensions Numero de dimensiones espaciales
     */
    explicit TemporalCoefficients(int numDimensions)
        : beta(Eigen::MatrixXd::Zero(4, numDimensions))
    {
    }

    /**
     * Acceso al coeficiente.
     * @param order Orden del termino (0=constante, 1=lineal, etc.)
     * @param dimension Indice de dimension (0-based)
     */
    double &operator()(int order, int dimension)
    {
        return beta(order, dimension);
    }

    double operator()(int order, int dimension) const
    {
        return beta(order, dimension);
    }

    /**
     * Obtiene fila como vector para una dimension.
     */
    Eigen::VectorXd getForDimension(int dimension) const
    {
        return beta.col(dimension);
    }

    /**
     * Establece coeficientes para un modelo constante (solo beta_0).
     */
    void setConstant(const Eigen::VectorXd &beta0)
    {
        beta.row(0) = beta0.transpose();
    }
};

/**
 * Matriz de tendencias temporales.
 */
struct TimeTrends
{
    Eigen::MatrixXd values; // (numPeriods x 4): [1, t, t^2, t^3]

    /**
     * Constructor con periodos.
     * @param numPeriods Numero maximo de periodos
     */
    explicit TimeTrends(int numPeriods)
        : values(Eigen::MatrixXd::Zero(numPeriods, 4))
    {
        // Inicializar columna 0 con 1s (termino constante)
        values.col(0).setOnes();
    }

    /**
     * Establece valores temporales para un periodo.
     * @param period Indice del periodo (0-based)
     * @param t Valor del tiempo normalizado
     */
    void setPeriod(int period, double t)
    {
        values(period, 0) = 1.0;
        values(period, 1) = t;
        values(period, 2) = t * t;
        values(period, 3) = t * t * t;
    }

    /**
     * Acceso al valor temporal.
     * @param period Indice del periodo (0-based)
     * @param order Orden del termino (0=1, 1=t, 2=t^2, 3=t^3)
     */
    double operator()(int period, int order) const
    {
        return values(period, order);
    }
};

/**
 * Informacion sobre los periodos en que sirvio un legislador.
 */
struct LegislatorPeriodInfo
{
    std::vector<bool> served;        // LWHERE: true si sirvio en ese periodo
    std::vector<int> dataIndices;    // KWHERE: indice en matriz de datos
    std::vector<int> rollCallCounts; // MCONG(j,2): numero de votaciones por periodo

    /**
     * Constructor.
     * @param numPeriods Numero total de periodos
     */
    explicit LegislatorPeriodInfo(int numPeriods)
        : served(numPeriods, false),
          dataIndices(numPeriods, -1),
          rollCallCounts(numPeriods, 0)
    {
    }

    /**
     * Marca que el legislador sirvio en un periodo.
     * @param period Indice del periodo (0-based)
     * @param dataIndex Indice en la matriz de datos
     * @param numRollCalls Numero de votaciones en ese periodo
     */
    void markServed(int period, int dataIndex, int numRollCalls)
    {
        served[period] = true;
        dataIndices[period] = dataIndex;
        rollCallCounts[period] = numRollCalls;
    }

    /**
     * Verifica si el legislador sirvio en un periodo.
     */
    bool servedIn(int period) const
    {
        return served[period];
    }
};

/**
 * Matriz de votos extendida con soporte para multiples periodos.
 */
class LegislatorVoteAccess
{
public:
    /**
     * Constructor.
     * @param votes Matriz de votos base
     * @param rollCallsPerPeriod Numero de votaciones por periodo
     */
    LegislatorVoteAccess(
        const VoteMatrix &votes,
        const std::vector<int> &rollCallsPerPeriod)
        : votes_(votes),
          rollCallsPerPeriod_(rollCallsPerPeriod)
    {
        // Calcular offsets acumulados
        offsets_.resize(rollCallsPerPeriod.size() + 1);
        offsets_[0] = 0;
        for (size_t i = 0; i < rollCallsPerPeriod.size(); ++i)
        {
            offsets_[i + 1] = offsets_[i] + rollCallsPerPeriod[i];
        }
    }

    /**
     * Obtiene el voto de un legislador en un periodo y votacion.
     * @param legDataIndex Indice del legislador en datos
     * @param period Indice del periodo
     * @param rcInPeriod Indice de votacion dentro del periodo
     */
    bool getVote(int legDataIndex, int period, int rcInPeriod) const
    {
        int globalRC = offsets_[period] + rcInPeriod;
        return votes_.getVote(legDataIndex, globalRC);
    }

    bool isMissing(int legDataIndex, int period, int rcInPeriod) const
    {
        int globalRC = offsets_[period] + rcInPeriod;
        return votes_.isMissing(legDataIndex, globalRC);
    }

    const VoteMatrix &getVotes() const { return votes_; }

private:
    const VoteMatrix &votes_;
    std::vector<int> rollCallsPerPeriod_;
    std::vector<int> offsets_;
};

/**
 * Calcula log-likelihood y derivadas para un legislador especifico.
 * @return Resultado con log-likelihood, derivadas, matrices de informacion
 */
LegislatorDerivativesResult computeLegislatorDerivatives(
    int legislatorIndex, // Indice del legislador (NEP, 0-based)
    const LegislatorPeriodInfo &periodInfo, // Informacion de periodos del legislador
    const TimeTrends &timeTrends, // Matriz de tendencias temporales (ATIME)
    const TemporalCoefficients &coefficients, // Coeficientes del modelo temporal (XBETA)
    const Eigen::MatrixXd &rollCallMidpoints, // Puntos medios de votaciones (ZMID)
    const Eigen::MatrixXd &rollCallSpreads, // Spreads de votaciones (DYN)
    const VoteMatrix &votes, // Matriz de votos
    const std::vector<bool> &validRollCalls, // Mascara de votaciones validas (RCBAD)
    const Eigen::VectorXd &weights, // Vector de pesos [w1, ..., wNS, beta]
    const NormalCDF &normalCDF, // Tabla CDF precomputada
    TemporalModel model, // Tipo de modelo temporal
    int firstPeriod, // Primer periodo del rango (NFIRST, 0-based)
    int lastPeriod); // Ultimo periodo del rango (NLAST, 0-based)

/**
 * Version simplificada para modelo constante con un solo periodo.
 * Para testing y casos simples donde no hay evolucion temporal.
 * 
 * @return Resultado con log-likelihood y derivadas para modelo constante
 */
LegislatorDerivativesResult computeLegislatorDerivativesSimple(
    const Eigen::VectorXd &legislatorCoords, // Coordenadas actuales del legislador (1 x NS)
    const Eigen::MatrixXd &rollCallMidpoints, // Puntos medios de votaciones (numRC x NS)
    const Eigen::MatrixXd &rollCallSpreads, // Spreads de votaciones (numRC x NS)
    const std::vector<bool> &votes, // Vector de votos del legislador (numRC)
    const std::vector<bool> &voteMissing, // Vector de missing data (numRC)
    const std::vector<bool> &validRollCalls, // Mascara de votaciones validas
    const Eigen::VectorXd &weights, // Vector de pesos [w1, ..., wNS, beta]
    const NormalCDF &normalCDF); // Tabla CDF precomputada

#endif // LEGISLATOR_DERIVATIVES_HPP
