/**
 * @file dwnominate.hpp
 * @brief Clase principal DWNominate - Orquestador del algoritmo DW-NOMINATE.
 *
 * Correspondencia con Fortran:
 * - dwnom() -> DWNominate::run()
 * - Variables globales xxcom_mod -> atributos miembro
 * - variables locales -> variables en metodos
 */

#ifndef DWNOMINATE_HPP
#define DWNOMINATE_HPP

#include "normal_cdf.hpp"
#include "likelihood.hpp"
#include "grid_optimizer.hpp"
#include "rollcall_derivatives.hpp"
#include "rollcall_optimizer.hpp"
#include "legislator_derivatives.hpp"
#include "optimize_legislators.hpp"
#include "cutting_plane.hpp"
#include "cutting_point.hpp"
#include "sort_utils.hpp"

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <map>
#include <functional>

/**
 * @brief Configuracion del modelo DW-NOMINATE.
 *
 * Equivale a NOMSTARTIN(1:6) en Fortran.
 */
struct DWNominateConfig
{
    int numDimensions;      // NS: Numero de dimensiones espaciales (1 o 2)
    int temporalModel;      // NMODEL: 0=constante, 1=lineal, 2=cuadratico
    int firstCongress;      // NFIRST: Primer congreso (0-based)
    int lastCongress;       // NLAST: Ultimo congreso (0-based)
    int firstIteration;     // IHAPPY1: Primera iteracion global
    int lastIteration;      // IHAPPY2: Ultima iteracion global
    double marginThreshold; // Umbral de margen (0.025)
    bool verbose;           // Imprimir mensajes de progreso

    DWNominateConfig()
        : numDimensions(2),
          temporalModel(0),
          firstCongress(0),
          lastCongress(0),
          firstIteration(1),
          lastIteration(1),
          marginThreshold(0.025),
          verbose(false)
    {
    }
};

/**
 * @brief Metadata de un congreso.
 *
 * Equivale a MCONG(II,1:3) en Fortran.
 * Los offsets reemplazan los acumuladores KTOTP/KTOTQ.
 */
struct CongressInfo
{
    int index;            // Indice del congreso (0-based)
    int numLegislators;   // Numero de legisladores en este congreso
    int numRollCalls;     // Numero de roll calls en este congreso
    int legislatorOffset; // Offset en matriz de legisladores (KTOTP)
    int rollCallOffset;   // Offset en matriz de roll calls (KTOTQ)

    CongressInfo()
        : index(0),
          numLegislators(0),
          numRollCalls(0),
          legislatorOffset(0),
          rollCallOffset(0)
    {
    }
};

/**
 * @brief Informacion de presencia de un legislador en congresos.
 *
 * Reemplaza LWHERE(id,congreso) y KWHERE(id,congreso) del Fortran.
 */
struct LegislatorPresence
{
    int uniqueId;                           // ID1: ID unico del legislador
    std::map<int, int> congressToDataIndex; // congreso -> indice en XDATA

    LegislatorPresence() : uniqueId(-1) {}
    explicit LegislatorPresence(int id) : uniqueId(id) {}

    bool isInCongress(int congress) const
    {
        return congressToDataIndex.find(congress) != congressToDataIndex.end();
    }

    int getDataIndex(int congress) const
    {
        auto it = congressToDataIndex.find(congress);
        return (it != congressToDataIndex.end()) ? it->second : -1;
    }

    int getNumCongresses() const
    {
        return static_cast<int>(congressToDataIndex.size());
    }
};

/**
 * @brief Estadisticas de un legislador individual.
 *
 * Equivale a XBIGLOG, KBIGLOG, GMP por legislador.
 */
struct LegislatorStats
{
    double logLikelihoodBefore; // XBIGLOG(i,1)
    double logLikelihoodAfter;  // XBIGLOG(i,2)
    int voteCountBefore;        // KBIGLOG(i,1)
    int voteCountAfter;         // KBIGLOG(i,2)
    int yesCountBefore;         // KBIGLOG(i,3)
    int yesCountAfter;          // KBIGLOG(i,4)
    double gmpBefore;           // GMPA
    double gmpAfter;            // GMPB
    double varianceX1;          // VARX1
    double varianceX2;          // VARX2
    double stdDevX1;            // SDX1
    double stdDevX2;            // SDX2

    LegislatorStats()
        : logLikelihoodBefore(0.0),
          logLikelihoodAfter(0.0),
          voteCountBefore(0),
          voteCountAfter(0),
          yesCountBefore(0),
          yesCountAfter(0),
          gmpBefore(0.0),
          gmpAfter(0.0),
          varianceX1(0.0),
          varianceX2(0.0),
          stdDevX1(0.0),
          stdDevX2(0.0)
    {
    }
};

/**
 * @brief Datos de entrada para DWNominate.
 *
 * Agrupa todos los parametros de entrada de la subrutina dwnom.
 */
struct DWNominateInput
{
    // Pesos iniciales: [w1, w2, ..., wNS, beta]
    Eigen::VectorXd initialWeights;

    // Coordenadas iniciales de legisladores (XDATAIN)
    // Dimension: numLegislators x numDimensions
    Eigen::MatrixXd legislatorCoords;

    // Midpoints iniciales de roll calls (ZMIDIN)
    // Dimension: numRollCalls x numDimensions
    Eigen::MatrixXd rollCallMidpoints;

    // Spreads iniciales de roll calls (DYNIN)
    // Dimension: numRollCalls x numDimensions
    Eigen::MatrixXd rollCallSpreads;

    // Matriz de votos
    VoteMatrix votes;

    // Congreso de cada roll call (ICONGIN, 0-based)
    std::vector<int> rollCallCongress;

    // Congreso de cada legislador (NCONGIN, 0-based)
    std::vector<int> legislatorCongress;

    // ID unico de cada legislador (ID1IN)
    std::vector<int> legislatorUniqueId;

    // Metadata de congresos (MCONGIN)
    // congressMetadata[congreso] = {numLegislators, numRollCalls}
    std::vector<std::pair<int, int>> congressMetadata;

    DWNominateInput()
        : votes(0, 0)
    {
    }

    DWNominateInput(size_t numLegislators, size_t numRollCalls)
        : votes(numLegislators, numRollCalls)
    {
    }
};

/**
 * @brief Resultado completo de la ejecucion de DWNominate.
 *
 * Agrupa todos los arrays de salida de la subrutina dwnom.
 */
struct DWNominateResult
{
    // Coordenadas optimizadas de legisladores (XDATAOUT)
    Eigen::MatrixXd legislatorCoords;

    // Midpoints optimizados de roll calls (ZMIDOUT)
    Eigen::MatrixXd rollCallMidpoints;

    // Spreads optimizados de roll calls (DYNOUT)
    Eigen::MatrixXd rollCallSpreads;

    // Pesos finales (WEIGHTSOUT)
    Eigen::VectorXd weights;

    // Estadisticas por legislador
    std::vector<LegislatorStats> legislatorStats;

    // Estadisticas de clasificacion finales
    ClassificationStats finalStats;

    // Log-likelihood final
    double finalLogLikelihood;

    // Numero de iteraciones ejecutadas
    int totalIterations;

    // Clasificacion antes/despues de roll calls
    int classificationBefore;
    int classificationAfter;
    int totalValidVotes;

    DWNominateResult()
        : finalLogLikelihood(0.0),
          totalIterations(0),
          classificationBefore(0),
          classificationAfter(0),
          totalValidVotes(0)
    {
    }
};

/**
 * @brief Clase principal del algoritmo DW-NOMINATE.
 *
 * Orquesta todas las fases del algoritmo, manteniendo el estado
 * interno y coordinando las llamadas a los optimizadores.
 */
class DWNominate
{
public:
    /**
     * @brief Constructor.
     * Inicializa el modelo con configuracion y datos de entrada.
     * @param config Configuracion del modelo
     * @param input Datos de entrada
     */
    DWNominate(const DWNominateConfig &config, const DWNominateInput &input);

    /**
     * @brief Ejecuta el algoritmo completo.
     * Implementa el bucle principal IHAPPY y todas las fases.
     * @return Resultado completo de la ejecucion
     */
    DWNominateResult run();

    /**
     * @brief Obtiene los pesos actuales.
     * @return Vector de pesos [w1, ..., wNS, beta]
     */
    const Eigen::VectorXd &getWeights() const { return weights_; }

    /**
     * @brief Obtiene el log-likelihood actual.
     * @return Log-likelihood global
     */
    double getCurrentLogLikelihood() const { return currentLogLikelihood_; }

    /**
     * @brief Obtiene las coordenadas actuales de legisladores.
     * @return Matriz de coordenadas
     */
    const Eigen::MatrixXd &getLegislatorCoords() const { return legislatorCoords_; }

    /**
     * @brief Obtiene los midpoints actuales de roll calls.
     * @return Matriz de midpoints
     */
    const Eigen::MatrixXd &getRollCallMidpoints() const { return rollCallMidpoints_; }

private:
    // CONFIGURACION
    DWNominateConfig config_;

    // DATOS DE ENTRADA (inmutables)
    VoteMatrix votes_;                                   // RCVOTE1, RCVOTE9
    std::vector<CongressInfo> congressInfo_;             // MCONG
    std::vector<int> rollCallCongress_;                  // ICONG
    std::vector<int> legislatorCongress_;                // NCONG
    std::vector<int> legislatorUniqueId_;                // ID1
    std::vector<bool> validRollCalls_;                   // RCBAD
    std::vector<LegislatorPresence> legislatorPresence_; // LWHERE, KWHERE

    // ESTADO INTERNO (mutable durante ejecucion)
    NormalCDF normalCDF_;               // ZDF (tabla CDF)
    Eigen::VectorXd weights_;           // WEIGHT(1:NS+1)
    Eigen::MatrixXd legislatorCoords_;  // XDATA
    Eigen::MatrixXd rollCallMidpoints_; // ZMID
    Eigen::MatrixXd rollCallSpreads_;   // DYN

    // Estadisticas globales
    ClassificationStats globalStats_; // KLASS, KLASSYY, etc.
    double currentLogLikelihood_;     // XPLOG actual

    // Estadisticas por legislador (se actualizan en cada iteracion)
    Eigen::MatrixXd legislatorLogLikelihood_; // XBIGLOG
    Eigen::MatrixXi legislatorVoteCounts_;    // KBIGLOG

    // Varianzas por legislador unico
    Eigen::MatrixXd legislatorVariances_; // XVAR

    // Polaridad de cortes por roll call
    std::vector<CuttingPolarity> rollCallPolarity_; // MCUTS

    // METODOS DE INICIALIZACION
    /**
     * @brief Inicializa la tabla CDF.
     */
    void initializeCDF();

    /**
     * @brief Carga metadata de congresos y calcula offsets.
     */
    void loadCongressMetadata(const DWNominateInput &input);

    /**
     * @brief Carga roll calls y determina validez.
     */
    void loadRollCalls(const DWNominateInput &input);

    /**
     * @brief Carga legisladores y construye mapas de presencia.
     */
    void loadLegislators(const DWNominateInput &input);

    // METODOS DE FASES
    /**
     * @brief Ejecuta fase de optimizacion de pesos (WINT).
     * Solo si NS >= 2.
     */
    void executeWeightPhase();

    /**
     * @brief Ejecuta fase de optimizacion de beta (SIGMAS).
     */
    void executeBetaPhase();

    /**
     * @brief Ejecuta fase de roll calls.
     *
     * @param iteration Numero de iteracion global (1-based, para logica IHAPPY)
     */
    void executeRollCallPhase(int iteration);

    /**
     * @brief Ejecuta fase de legisladores.
     */
    void executeLegislatorPhase();

    // METODOS AUXILIARES DE ROLL CALLS
    /**
     * @brief Procesa un roll call individual.
     *
     * @param congressIndex Indice del congreso (0-based)
     * @param rollCallLocalIndex Indice del roll call dentro del congreso (0-based)
     * @param globalRollCallIndex Indice global del roll call
     * @param iteration Numero de iteracion global
     * @param legislatorOffset Offset de legisladores para este congreso
     * @param classificationBefore Acumulador de clasificacion antes
     * @param classificationAfter Acumulador de clasificacion despues
     * @param totalVotes Acumulador de votos totales
     */
    void processRollCall(
        int congressIndex,
        int rollCallLocalIndex,
        int globalRollCallIndex,
        int iteration,
        int legislatorOffset,
        int &classificationBefore,
        int &classificationAfter,
        int &totalVotes);

    /**
     * @brief Prepara datos para un roll call.
     *
     * Extrae coordenadas de legisladores, codigos de voto,
     * y ordena por primera dimension.
     *
     * @param congressIndex Indice del congreso
     * @param rollCallLocalIndex Indice local del roll call
     * @param legislatorOffset Offset de legisladores
     * @param coords [out] Matriz de coordenadas ordenadas
     * @param voteCodes [out] Codigos de voto ordenados
     * @param sortedIndices [out] Indices de ordenamiento
     * @param yesCount [out] Conteo de votos Si
     * @param noCount [out] Conteo de votos No
     * @return Numero de legisladores con voto valido
     */
    int prepareRollCallData(
        int congressIndex,
        int rollCallLocalIndex,
        int legislatorOffset,
        Eigen::MatrixXd &coords,
        std::vector<int> &voteCodes,
        std::vector<int> &sortedIndices,
        int &yesCount,
        int &noCount);

    /**
     * @brief Aplica JAN11PT para NS=1.
     */
    void applyJan11pt(
        int numVoters,
        const std::vector<double> &projections,
        const std::vector<int> &voteCodes,
        double &cuttingPoint,
        double &spread,
        CuttingPolarity &polarity,
        double &accuracy1,
        double &accuracy2);

    /**
     * @brief Aplica CUTPLANE para NS>1.
     */
    void applyCutplane(
        int numVoters,
        const Eigen::MatrixXd &coords,
        const std::vector<int> &voteCodes,
        Eigen::VectorXd &midpoint,
        Eigen::VectorXd &spread,
        CuttingPolarity &polarity);

    // METODOS AUXILIARES DE LEGISLADORES
    /**
     * @brief Procesa un legislador unico.
     *
     * @param uniqueId ID unico del legislador (NEP en Fortran)
     * @param presence Informacion de presencia en congresos
     */
    void processLegislator(int uniqueId, const LegislatorPresence &presence);

    // METODOS DE UTILIDAD
    /**
     * @brief Calcula log-likelihood global (PLOG).
     * @return Log-likelihood total
     */
    double computeLogLikelihood();

    /**
     * @brief Normaliza un vector a la esfera unitaria.
     * Si ||v|| > 1, lo escala a ||v|| = 1.
     */
    void normalizeToUnitSphere(Eigen::VectorXd &point);

    /**
     * @brief Verifica si un roll call es valido (margen >= umbral).
     *
     * @param yesCount Numero de votos Si
     * @param noCount Numero de votos No
     * @return true si el roll call es valido
     */
    bool isRollCallValid(int yesCount, int noCount) const;

    /**
     * @brief Log de progreso (si verbose esta activo).
     */
    void log(const std::string &message) const;

    // METODOS AUXILIARES DE INTEGRACION CON OPTIMIZADORES
    /**
     * @brief Construye vector de RollCallParameters desde estado interno.
     * @return Vector de parametros de roll calls
     */
    std::vector<RollCallParameters> buildRollCallParams() const;

    /**
     * @brief Construye LegislatorPeriodInfo para un legislador.
     *
     * Convierte LegislatorPresence al formato esperado por optimizeLegislator.
     * @param uniqueId ID unico del legislador
     * @param presence Informacion de presencia en congresos
     * @return Informacion de periodos para el optimizador
     */
    LegislatorPeriodInfo buildLegislatorPeriodInfo(
        int uniqueId,
        const LegislatorPresence &presence) const;

    /**
     * @brief Reconstruye coordenadas de legislador desde coeficientes temporales.
     *
     * Para un legislador con presencia en multiples congresos, reconstruye
     * las coordenadas XDATA usando los coeficientes XBETA optimizados.
     *
     * @param presence Informacion de presencia
     * @param coefficients Coeficientes temporales optimizados
     */
    void reconstructLegislatorCoords(
        const LegislatorPresence &presence,
        const TemporalCoefficients &coefficients);
};
#endif // DWNOMINATE_HPP
