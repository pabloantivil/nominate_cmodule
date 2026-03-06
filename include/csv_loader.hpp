/**
 * @file csv_loader.hpp
 * @brief Cargador de datos CSV para DW-NOMINATE.
 *
 * Este módulo permite cargar datos reales de votaciones desde archivos CSV
 * en el formato utilizado por el paquete dwnominate en R.
 *
 * Estructura esperada de archivos:
 * - votes_matrix_pN.csv: Matriz de votos (legisladores x votaciones)
 *   - Primera columna: ID legislador
 *   - Columnas restantes: votos (1=Sí, 0=No, NA=Missing)
 *
 * - legislator_metadata.csv: Metadata de legisladores
 *   - Columnas: legislator_id, id, nombres, partido, region, distrito
 *
 * - vote_metadata.csv: Metadata de votaciones
 *   - Columnas específicas del dataset
 */

#ifndef CSV_LOADER_HPP
#define CSV_LOADER_HPP

#include "dwnominate.hpp"
#include <string>
#include <vector>
#include <map>
#include <set>

/**
 * @brief Información de un legislador cargado desde CSV.
 */
struct LegislatorInfo
{
    int id;
    std::string name;
    std::string party;
    std::string region;
    std::string district;

    LegislatorInfo() : id(-1) {}
};

/**
 * @brief Información de un período (congreso) cargado.
 */
struct PeriodData
{
    int periodIndex;                     // Índice del período (0-based)
    std::vector<int> legislatorIds;      // IDs de legisladores en orden
    int numRollCalls;                    // Número de votaciones
    std::vector<std::vector<int>> votes; // votes[legislator][rollcall]: 1=Sí, 0=No, -1=Missing

    PeriodData() : periodIndex(0), numRollCalls(0) {}
};

/**
 * @brief Resultado de cargar coordenadas de referencia (salida de R).
 */
struct ReferenceCoordinates
{
    int period;
    int legislatorId;
    std::string party;
    double coord1D;
    double coord2D;
    double se1D;
    double se2D;
    double logLikelihood;
    int numVotes;
    int numErrors;
    double gmp;
};

/**
 * @brief Parámetros de votación de referencia (salida de R).
 */
struct ReferenceBillParams
{
    int session;
    int billId;
    double midpoint1D;
    double midpoint2D;
    double spread1D;
    double spread2D;
    bool isValid; // false si tiene NA

    ReferenceBillParams() : session(0), billId(0), isValid(false) {}
};

/**
 * @brief Coordenadas iniciales de W-NOMINATE (para inicialización de DW-NOMINATE).
 *
 * Estas coordenadas provienen de una ejecución independiente de W-NOMINATE
 * y se usan como punto de partida para DW-NOMINATE, replicando el pipeline
 * metodológico del paquete dwnominate en R.
 */
struct WNominateCoords
{
    int legislatorId;
    double coord1D;
    double coord2D;
    std::string party;
    std::string name;

    WNominateCoords() : legislatorId(-1), coord1D(0.0), coord2D(0.0) {}
};

/**
 * @brief Configuración de inicialización para DW-NOMINATE.
 */
struct InitializationConfig
{
    double beta = 4.925;            // Parámetro de error espacial (WEIGHT[NS+1])
    double w2 = 0.5;                // Peso de segunda dimensión (WEIGHT[2])
    bool useWNominateStart = false; // Si true, usa coordenadas de W-NOMINATE
    std::string wnominatePath;      // Path al CSV de coordenadas W-NOMINATE
};

/**
 * @brief Clase para cargar datos de DW-NOMINATE desde CSV.
 */
class CSVLoader
{
public:
    /**
     * @brief Constructor.
     * @param inputDir Directorio con archivos de entrada (votes_matrix_pN.csv, etc.)
     * @param outputDir Directorio con archivos de salida de R (opcional, para validación)
     */
    CSVLoader(const std::string &inputDir, const std::string &outputDir = "");

    /**
     * @brief Carga todos los datos y construye DWNominateInput.
     *
     * @param numPeriods Número de períodos a cargar (1-5)
     * @return DWNominateInput poblado con los datos
     */
    DWNominateInput loadInput(int numPeriods = 5);

    /**
     * @brief Carga datos con configuración de inicialización específica.
     *
     * Esta versión permite especificar:
     * - Coordenadas iniciales desde W-NOMINATE
     * - Parámetros beta y w exactos
     *
     * @param numPeriods Número de períodos a cargar
     * @param initConfig Configuración de inicialización
     * @return DWNominateInput poblado con coordenadas y parámetros especificados
     */
    DWNominateInput loadInput(int numPeriods, const InitializationConfig &initConfig);

    /**
     * @brief Carga coordenadas de W-NOMINATE desde CSV.
     *
     * El archivo debe tener formato:
     * coord1D,coord2D,legislator_id,legislator_name,party
     *
     * @param path Ruta al archivo CSV de coordenadas W-NOMINATE
     * @return Mapa de legislator_id -> WNominateCoords
     */
    std::map<int, WNominateCoords> loadWNominateCoordinates(const std::string &path);

    /**
     * @brief Carga coordenadas de referencia desde output de R.
     *
     * @return Vector de coordenadas de referencia por período y legislador
     */
    std::vector<ReferenceCoordinates> loadReferenceCoordinates();

    /**
     * @brief Carga parámetros de votaciones de referencia desde output de R.
     *
     * @return Vector de parámetros de votaciones
     */
    std::vector<ReferenceBillParams> loadReferenceBillParams();

    /**
     * @brief Obtiene el mapeo de ID legislador a índice en la matriz.
     */
    const std::map<int, int> &getLegislatorIdToIndex() const { return legislatorIdToIndex_; }

    /**
     * @brief Obtiene la lista ordenada de IDs de legisladores.
     */
    const std::vector<int> &getLegislatorIds() const { return legislatorIds_; }

    /**
     * @brief Obtiene información de legisladores.
     */
    const std::map<int, LegislatorInfo> &getLegislatorInfo() const { return legislatorInfo_; }

    /**
     * @brief Obtiene el número de votaciones por período.
     */
    const std::vector<int> &getRollCallsPerPeriod() const { return rollCallsPerPeriod_; }

    /**
     * @brief Obtiene el offset de votaciones para cada período.
     */
    int getRollCallOffset(int period) const;

private:
    std::string inputDir_;
    std::string outputDir_;

    // Datos cargados
    std::map<int, LegislatorInfo> legislatorInfo_;
    std::vector<int> legislatorIds_;
    std::map<int, int> legislatorIdToIndex_;
    std::vector<PeriodData> periodData_;
    std::vector<int> rollCallsPerPeriod_;

    // Método interno para construir input con opciones
    DWNominateInput buildDWNominateInput(
        int numPeriods,
        const InitializationConfig *initConfig = nullptr);

    // Métodos auxiliares
    void loadLegislatorMetadata();
    PeriodData loadPeriodVotes(int periodNum);
    void buildUnifiedLegislatorList();

    // Utilidades de parsing CSV
    static std::vector<std::string> splitCSVLine(const std::string &line);
    static double parseDouble(const std::string &str, double defaultVal = 0.0);
    static int parseInt(const std::string &str, int defaultVal = -1);
    static bool isNA(const std::string &str);
};

#endif // CSV_LOADER_HPP
