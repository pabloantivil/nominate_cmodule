/**
 * @file cutting_point.hpp
 * @brief Algoritmos de punto de corte optimo para clasificacion 1D (JAN1PT/JAN11PT).
 *
 * Subrutinas de punto de corte del codigo Fortran.
 *
 * JAN1PT y JAN11PT encuentran el punto de corte optimo en una dimension para
 * clasificar los votos de una votacion nominal.
 *
 * Diferencias entre JAN1PT y JAN11PT:
 * - JAN1PT (NOTE=2): Prueba ambas polaridades y selecciona la mejor
 * - JAN11PT (NOTE=1): Usa polaridad fija pasada como parametro
 *
 */

#ifndef CUTTING_POINT_HPP
#define CUTTING_POINT_HPP

#include "sort_utils.hpp"
#include <Eigen/Dense>
#include <vector>

/**
 * @brief Codigos de voto en el sistema DW-NOMINATE.
 */
namespace VoteCode {
    constexpr int YES = 1;      // Voto afirmativo
    constexpr int NO = 6;       // Voto negativo
    constexpr int MISSING = 9;  // Voto ausente/abstencion
}

/**
 * @brief Modos de operacion para JAN1PT.
 *
 * IROTC en el codigo Fortran controla el comportamiento:
 * - NORMAL: Probar ambas polaridades, calcular centroides
 * - SINGLE_POLARITY: Solo probar KCUT=1/LCUT=6
 * - ROTATION_MODE: Modo especial para SEARCH/rotaciones
 */
enum class CuttingPointMode {
    NORMAL = 0,           // IROTC=0: Probar ambas polaridades, calcular centroides
    SINGLE_POLARITY = 1,  // IROTC=1: Solo probar polaridad KCUT=1/LCUT=6
    ROTATION_MODE = 2     // IROTC=2: Modo para busqueda de rotaciones
};

/**
 * @brief Polaridad del punto de corte.
 *
 * Define que tipo de voto se espera en cada lado del corte.
 */
struct CuttingPolarity {
    int lowSideVote;   // KCUT: Voto esperado en lado bajo (< corte)
    int highSideVote;  // LCUT: Voto esperado en lado alto (> corte)

    CuttingPolarity() : lowSideVote(VoteCode::YES), highSideVote(VoteCode::NO) {}
    CuttingPolarity(int kcut, int lcut) : lowSideVote(kcut), highSideVote(lcut) {}

    bool isYesLow() const { return lowSideVote == VoteCode::YES; }
};

/**
 * @brief Conteos de clasificacion para un punto de corte.
 *
 * Corresponde a la matriz de confusion 2x2:
 *              Prediccion
 *            Low   |  High
 * Real Low   JCL   |  JEL
 * Real High  JEH   |  JCH
 */
struct ClassificationCounts {
    int correctLow;   // JCL: Correctos en lado bajo
    int errorsLow;    // JEL: Errores en lado bajo
    int correctHigh;  // JCH: Correctos en lado alto
    int errorsHigh;   // JEH: Errores en lado alto

    ClassificationCounts() : correctLow(0), errorsLow(0), correctHigh(0), errorsHigh(0) {}

    int totalCorrect() const { return correctLow + correctHigh; }
    int totalErrors() const { return errorsLow + errorsHigh; }
    int total() const { return correctLow + errorsLow + correctHigh + errorsHigh; }

    double errorRate() const {
        int t = total();
        return t > 0 ? static_cast<double>(totalErrors()) / t : 1.0;
    }

    double accuracy() const {
        return 1.0 - errorRate();
    }
};

/**
 * @brief Centroides de cada grupo de clasificacion.
 *
 * Coordenadas promedio de los legisladores en cada categoria.
 * Solo se calcula cuando mode == NORMAL.
 */
struct GroupCentroids {
    Eigen::VectorXd correctLow;   // XJCL: Centroide de correctos en lado bajo
    Eigen::VectorXd errorsLow;    // XJEL: Centroide de errores en lado bajo
    Eigen::VectorXd correctHigh;  // XJCH: Centroide de correctos en lado alto
    Eigen::VectorXd errorsHigh;   // XJEH: Centroide de errores en lado alto

    explicit GroupCentroids(int numDim = 1)
        : correctLow(Eigen::VectorXd::Zero(numDim)),
          errorsLow(Eigen::VectorXd::Zero(numDim)),
          correctHigh(Eigen::VectorXd::Zero(numDim)),
          errorsHigh(Eigen::VectorXd::Zero(numDim)) {}
};

/**
 * @brief Informacion de cortes candidatos para un modo intermedio (JROTC=0).
 *
 * Almacena informacion de todos los puntos de corte evaluados,
 * util para analisis posterior o refinamiento.
 */
struct CuttingPointCandidates {
    std::vector<double> positions;   // ZS: Posiciones de corte
    std::vector<int> correctLow;     // LLV: Correctos lado bajo
    std::vector<int> errorsLow;      // LLE: Errores lado bajo
    std::vector<int> correctHigh;    // LLVB: Correctos lado alto
    std::vector<int> errorsHigh;     // LLEB: Errores lado alto

    void resize(size_t n) {
        positions.resize(n);
        correctLow.resize(n);
        errorsLow.resize(n);
        correctHigh.resize(n);
        errorsHigh.resize(n);
    }

    size_t size() const { return positions.size(); }
};

/**
 * @brief Resultado del algoritmo de punto de corte optimo.
 */
struct CuttingPointResult {
    // Punto de corte optimo
    double cuttingPoint;          // WS(IVOT): Posicion del corte optimo
    CuttingPolarity polarity;     // KCCUT/LCCUT: Polaridad optima
    double errorRate;             // Tasa de error del corte optimo

    // Conteos de clasificacion
    ClassificationCounts counts;  // JCH, JEH, JCL, JEL

    // Centroides (solo si mode == NORMAL)
    GroupCentroids centroids;     // XJCH, XJEH, XJCL, XJEL

    // Matriz de errores por legislador (opcional)
    std::vector<int> legislatorErrors;  // LERROR: 0=correcto, 1=error

    // Informacion de candidatos intermedios (solo si mode == ROTATION_MODE)
    CuttingPointCandidates candidates;

    // Metadata
    int numLegislators;           // NP
    int numDimensions;            // NS

    CuttingPointResult()
        : cuttingPoint(0.0),
          errorRate(1.0),
          numLegislators(0),
          numDimensions(1) {}

    explicit CuttingPointResult(int numDim)
        : cuttingPoint(0.0),
          errorRate(1.0),
          centroids(numDim),
          numLegislators(0),
          numDimensions(numDim) {}
};

/**
 * @brief Encuentra el punto de corte optimo en 1D (JAN1PT).
 *
 * @param projections Proyecciones ordenadas de legisladores (YSS)
 * @param votes Votos reordenados segun projections (KA): 1=Si, 6=No, 9=Ausente
 * @param originalIndices Mapeo de indices ordenados a originales (LLL), 0-based
 * @param legislatorCoords Coordenadas originales de legisladores (XMAT)
 * @param numDimensions Numero de dimensiones espaciales (NS)
 * @param rollCallIndex Indice del roll call actual (IVOT), 0-based
 * @param mode Modo de operacion (IROTC)
 * @return Resultado con punto de corte optimo y clasificacion
 *
 * @note Las proyecciones deben estar ordenadas en orden ascendente.
 * @note Los votos deben estar reordenados para corresponder con las proyecciones.
 */
CuttingPointResult findCuttingPoint1D(
    const std::vector<double>& projections,
    const std::vector<int>& votes,
    const std::vector<int>& originalIndices,
    const Eigen::MatrixXd& legislatorCoords,
    int numDimensions,
    int rollCallIndex,
    CuttingPointMode mode = CuttingPointMode::NORMAL);

/**
 * @brief Version simplificada sin calculo de centroides.
 *
 * Util para busquedas rapidas donde solo se necesita el punto de corte
 * y la clasificacion, sin las coordenadas de centroides.
 *
 * @param projections Proyecciones ordenadas
 * @param votes Votos reordenados
 * @return Resultado con punto de corte optimo (sin centroides)
 */
CuttingPointResult findCuttingPoint1DSimple(
    const std::vector<double>& projections,
    const std::vector<int>& votes);

/**
 * @brief Encuentra el punto de corte optimo con polaridad fija (JAN11PT).
 *
 * Traduccion de SUBROUTINE JAN11PT (lineas 781-963).
 *
 * A diferencia de findCuttingPoint1D (JAN1PT), esta funcion:
 * - NO prueba ambas polaridades: usa la polaridad pasada como parametro
 * - Esta disenada para datos ya proyectados y ordenados en 1D
 * - Es mas eficiente cuando la polaridad ya se conoce externamente
 *
 * @param projections Proyecciones ordenadas de legisladores (YSS)
 * @param votes Votos ordenados correspondientes (KA): 1=Si, 6=No, 9=Ausente
 * @param polarity Polaridad fija para la clasificacion (KCCUT/LCCUT)
 * @return Resultado con punto de corte optimo y clasificacion
 *
 * @note Las proyecciones deben estar ordenadas en orden ascendente.
 * @note Los votos deben estar reordenados para corresponder con las proyecciones.
 */
CuttingPointResult findCuttingPoint1DFixedPolarity(
    const std::vector<double>& projections,
    const std::vector<int>& votes,
    const CuttingPolarity& polarity);

#endif // CUTTING_POINT_HPP
