/**
 * @file csv_loader.cpp
 * @brief Implementación del cargador de datos CSV para DW-NOMINATE.
 */

#include "csv_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <cmath>

// Constructor
CSVLoader::CSVLoader(const std::string &inputDir, const std::string &outputDir)
    : inputDir_(inputDir), outputDir_(outputDir)
{
}

// Utilidades de parsing CSV
std::vector<std::string> CSVLoader::splitCSVLine(const std::string &line)
{
    std::vector<std::string> result;
    std::string current;
    bool inQuotes = false;

    for (size_t i = 0; i < line.size(); ++i)
    {
        char c = line[i];
        if (c == '"')
        {
            inQuotes = !inQuotes;
        }
        else if (c == ',' && !inQuotes)
        {
            result.push_back(current);
            current.clear();
        }
        else
        {
            current += c;
        }
    }
    result.push_back(current);
    return result;
}

double CSVLoader::parseDouble(const std::string &str, double defaultVal)
{
    if (isNA(str) || str.empty())
    {
        return defaultVal;
    }
    try
    {
        return std::stod(str);
    }
    catch (...)
    {
        return defaultVal;
    }
}

int CSVLoader::parseInt(const std::string &str, int defaultVal)
{
    if (isNA(str) || str.empty())
    {
        return defaultVal;
    }
    try
    {
        return std::stoi(str);
    }
    catch (...)
    {
        return defaultVal;
    }
}

bool CSVLoader::isNA(const std::string &str)
{
    std::string s = str;
    // Trim whitespace
    s.erase(0, s.find_first_not_of(" \t\r\n"));
    s.erase(s.find_last_not_of(" \t\r\n") + 1);

    return s == "NA" || s == "na" || s == "N/A" || s == "NaN" || s == "nan" || s.empty();
}

// Carga de metadata de legisladores
void CSVLoader::loadLegislatorMetadata()
{
    std::string path = inputDir_ + "/legislator_metadata.csv";
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("No se puede abrir: " + path);
    }

    std::string line;
    // Leer header
    std::getline(file, line);
    // Esperado: legislator_id,id,nombres,partido,region,distrito

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        auto fields = splitCSVLine(line);
        if (fields.size() < 5)
            continue;

        LegislatorInfo info;
        info.id = parseInt(fields[0]);
        info.name = fields.size() > 2 ? fields[2] : "";
        info.party = fields.size() > 3 ? fields[3] : "";
        info.region = fields.size() > 4 ? fields[4] : "";
        info.district = fields.size() > 5 ? fields[5] : "";

        if (info.id > 0)
        {
            legislatorInfo_[info.id] = info;
        }
    }
}

// Carga de votos de un período
PeriodData CSVLoader::loadPeriodVotes(int periodNum)
{
    PeriodData data;
    data.periodIndex = periodNum - 1; // 0-based

    std::string path = inputDir_ + "/votes_matrix_p" + std::to_string(periodNum) + ".csv";
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("No se puede abrir: " + path);
    }

    std::string line;
    bool isHeader = true;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        auto fields = splitCSVLine(line);

        if (isHeader)
        {
            // Primera fila es header: primera columna es ID legislador,
            // resto son IDs de votaciones
            data.numRollCalls = static_cast<int>(fields.size()) - 1;
            isHeader = false;
            continue;
        }

        // Primera columna es ID del legislador
        int legId = parseInt(fields[0]);
        if (legId < 0)
            continue;

        data.legislatorIds.push_back(legId);

        // Resto son votos
        std::vector<int> legVotes;
        for (size_t j = 1; j < fields.size(); ++j)
        {
            if (isNA(fields[j]))
            {
                legVotes.push_back(-1); // Missing
            }
            else
            {
                int vote = parseInt(fields[j], -1);
                legVotes.push_back(vote);
            }
        }
        data.votes.push_back(legVotes);
    }

    return data;
}

// Construir lista unificada de legisladores
void CSVLoader::buildUnifiedLegislatorList()
{
    // Obtener todos los IDs únicos de legisladores de todos los períodos
    std::set<int> allIds;
    for (const auto &period : periodData_)
    {
        for (int id : period.legislatorIds)
        {
            allIds.insert(id);
        }
    }

    // Convertir a vector ordenado
    legislatorIds_.clear();
    legislatorIds_.assign(allIds.begin(), allIds.end());
    std::sort(legislatorIds_.begin(), legislatorIds_.end());

    // Construir mapeo de ID a índice
    legislatorIdToIndex_.clear();
    for (size_t i = 0; i < legislatorIds_.size(); ++i)
    {
        legislatorIdToIndex_[legislatorIds_[i]] = static_cast<int>(i);
    }
}

// Obtener offset de roll calls para un período
int CSVLoader::getRollCallOffset(int period) const
{
    int offset = 0;
    for (int p = 0; p < period && p < static_cast<int>(rollCallsPerPeriod_.size()); ++p)
    {
        offset += rollCallsPerPeriod_[p];
    }
    return offset;
}

// Carga de coordenadas W-NOMINATE
std::map<int, WNominateCoords> CSVLoader::loadWNominateCoordinates(const std::string &path)
{
    std::map<int, WNominateCoords> result;

    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("No se puede abrir archivo W-NOMINATE: " + path);
    }

    std::string line;
    bool isHeader = true;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        auto fields = splitCSVLine(line);

        if (isHeader)
        {
            // Esperado: "coord1D","coord2D","legislator_id","legislator_name","party"
            isHeader = false;
            continue;
        }

        if (fields.size() < 3)
            continue;

        WNominateCoords coords;
        coords.coord1D = parseDouble(fields[0]);
        coords.coord2D = parseDouble(fields[1]);
        coords.legislatorId = parseInt(fields[2]);
        if (fields.size() > 3)
            coords.name = fields[3];
        if (fields.size() > 4)
            coords.party = fields[4];

        if (coords.legislatorId > 0)
        {
            result[coords.legislatorId] = coords;
        }
    }

    std::cout << "  W-NOMINATE: Cargadas " << result.size()
              << " coordenadas iniciales desde " << path << std::endl;

    return result;
}

// Carga principal: construir DWNominateInput (wrapper simple)
DWNominateInput CSVLoader::loadInput(int numPeriods)
{
    return buildDWNominateInput(numPeriods, nullptr);
}

// Carga con configuración de inicialización específica
DWNominateInput CSVLoader::loadInput(int numPeriods, const InitializationConfig &initConfig)
{
    return buildDWNominateInput(numPeriods, &initConfig);
}

// Implementación interna: construir DWNominateInput
DWNominateInput CSVLoader::buildDWNominateInput(int numPeriods, const InitializationConfig *initConfig)
{
    // 1. Cargar metadata de legisladores
    loadLegislatorMetadata();

    // 2. Cargar datos de votaciones de cada período
    periodData_.clear();
    rollCallsPerPeriod_.clear();
    for (int p = 1; p <= numPeriods; ++p)
    {
        PeriodData pd = loadPeriodVotes(p);
        periodData_.push_back(pd);
        rollCallsPerPeriod_.push_back(pd.numRollCalls);
    }

    // 3. Construir lista unificada de legisladores
    buildUnifiedLegislatorList();

    // 4. Calcular dimensiones totales
    int totalLegislators = static_cast<int>(legislatorIds_.size());
    int totalRollCalls = 0;
    for (int rc : rollCallsPerPeriod_)
    {
        totalRollCalls += rc;
    }

    // 5. Construir DWNominateInput
    DWNominateInput input(totalLegislators, totalRollCalls);

    // 5.1 Pesos iniciales: [W1=1.0, W2, Beta]
    // Si hay configuración, usar valores especificados; sino, usar defaults
    input.initialWeights.resize(3); // NS=2 + 1
    input.initialWeights(0) = 1.0;  // W1 siempre es 1.0
    if (initConfig)
    {
        input.initialWeights(1) = initConfig->w2;   // Peso dimension 2
        input.initialWeights(2) = initConfig->beta; // Beta (sigma^2)
        std::cout << "  Pesos iniciales: W1=1.0, W2=" << initConfig->w2
                  << ", Beta=" << initConfig->beta << std::endl;
    }
    else
    {
        input.initialWeights(1) = 0.5;
        input.initialWeights(2) = 4.925;
    }

    // 5.2 Coordenadas iniciales de legisladores
    input.legislatorCoords = Eigen::MatrixXd::Zero(totalLegislators, 2);

    // Si hay configuración con coordenadas externas, cargarlas
    std::map<int, WNominateCoords> wnomCoords;
    bool useWNominate = initConfig && initConfig->useWNominateStart &&
                        !initConfig->wnominatePath.empty();

    if (useWNominate)
    {
        wnomCoords = loadWNominateCoordinates(initConfig->wnominatePath);
    }

    // Asignar coordenadas: usar externas si disponibles, sino valores arbitrarios
    int coordsFromWNom = 0;
    int coordsFallback = 0;
    for (int i = 0; i < totalLegislators; ++i)
    {
        int legId = legislatorIds_[i];
        auto it = wnomCoords.find(legId);

        if (useWNominate && it != wnomCoords.end())
        {
            // Usar coordenadas de archivo externo (W-NOMINATE o R)
            input.legislatorCoords(i, 0) = it->second.coord1D;
            input.legislatorCoords(i, 1) = it->second.coord2D;
            coordsFromWNom++;
        }
        else
        {
            // Fallback: distribución uniforme simple basada en índice
            double frac = static_cast<double>(i) / static_cast<double>(totalLegislators);
            input.legislatorCoords(i, 0) = frac - 0.5;
            input.legislatorCoords(i, 1) = (i % 2 == 0 ? 0.1 : -0.1) * frac;
            coordsFallback++;
        }
    }

    if (useWNominate)
    {
        std::cout << "  Coordenadas aplicadas: " << coordsFromWNom
                  << " desde archivo, " << coordsFallback << " fallback\n";
    }

    // 5.3 Midpoints y spreads iniciales de roll calls
    input.rollCallMidpoints = Eigen::MatrixXd::Zero(totalRollCalls, 2);
    input.rollCallSpreads = Eigen::MatrixXd::Constant(totalRollCalls, 2, 0.3);

    // 5.3.1 NUEVO: Si tenemos parámetros de referencia de R, usarlos como inicialización
    // Esto replica el flujo del Fortran donde ZMID viene pre-calculado
    auto rBillParams = loadReferenceBillParams();
    if (!rBillParams.empty())
    {
        int paramsLoaded = 0;
        for (const auto &bp : rBillParams)
        {
            int period = bp.session - 1;
            if (period < 0 || period >= numPeriods)
                continue;

            int rcOffset = getRollCallOffset(period);
            int globalIdx = rcOffset + (bp.billId - 1);

            if (globalIdx >= 0 && globalIdx < totalRollCalls && bp.isValid)
            {
                input.rollCallMidpoints(globalIdx, 0) = bp.midpoint1D;
                input.rollCallMidpoints(globalIdx, 1) = bp.midpoint2D;
                input.rollCallSpreads(globalIdx, 0) = bp.spread1D;
                input.rollCallSpreads(globalIdx, 1) = bp.spread2D;
                paramsLoaded++;
            }
        }

        if (paramsLoaded > 0)
        {
            std::cout << "  Bill params inicializados desde R: " << paramsLoaded << "/" << totalRollCalls << "\n";
        }
    }

    // 5.4 Votos y congresos
    input.rollCallCongress.resize(totalRollCalls);
    input.legislatorCongress.resize(totalLegislators);
    input.legislatorUniqueId.resize(totalLegislators);

    // Asignar IDs únicos de legisladores
    for (int i = 0; i < totalLegislators; ++i)
    {
        input.legislatorUniqueId[i] = legislatorIds_[i];
        // Determinar en qué período aparece el legislador
        // Para simplificar, asumimos que aparecen en todos los períodos
        input.legislatorCongress[i] = 0; // Se ajustará según participación
    }

    // 5.5 Construir matriz de votos
    int rollCallOffset = 0;
    for (int period = 0; period < numPeriods; ++period)
    {
        const auto &pd = periodData_[period];

        // Mapear legisladores de este período
        std::map<int, int> periodLegIdxMap;
        for (size_t i = 0; i < pd.legislatorIds.size(); ++i)
        {
            periodLegIdxMap[pd.legislatorIds[i]] = static_cast<int>(i);
        }

        // Asignar congresos de roll calls
        for (int j = 0; j < pd.numRollCalls; ++j)
        {
            input.rollCallCongress[rollCallOffset + j] = period;
        }

        // Asignar votos
        for (int globalLegIdx = 0; globalLegIdx < totalLegislators; ++globalLegIdx)
        {
            int legId = legislatorIds_[globalLegIdx];
            auto it = periodLegIdxMap.find(legId);

            for (int j = 0; j < pd.numRollCalls; ++j)
            {
                int globalRcIdx = rollCallOffset + j;

                if (it != periodLegIdxMap.end())
                {
                    int periodLegIdx = it->second;
                    int vote = pd.votes[periodLegIdx][j];
                    // Codificación: 1=Sí, 0/6=No, 9=Missing/Abstención, -1=NA
                    if (vote == -1 || vote == 9)
                    {
                        // Missing data: NA en CSV o abstención/ausencia (código 9)
                        input.votes.setVote(globalLegIdx, globalRcIdx, false, true);
                    }
                    else
                    {
                        // Vote 1 = Sí, Vote 0 o 6 = No
                        input.votes.setVote(globalLegIdx, globalRcIdx, vote == 1, false);
                    }
                }
                else
                {
                    // Legislador no presente en este período
                    input.votes.setVote(globalLegIdx, globalRcIdx, false, true);
                }
            }
        }

        rollCallOffset += pd.numRollCalls;
    }

    // 5.6 Metadata de congresos
    input.congressMetadata.clear();
    for (int period = 0; period < numPeriods; ++period)
    {
        // Contar legisladores activos en este período
        int activeLegislators = static_cast<int>(periodData_[period].legislatorIds.size());
        int numRC = periodData_[period].numRollCalls;
        input.congressMetadata.push_back({activeLegislators, numRC});
    }

    return input;
}

// Carga de coordenadas de referencia (output de R)
std::vector<ReferenceCoordinates> CSVLoader::loadReferenceCoordinates()
{
    std::vector<ReferenceCoordinates> result;

    if (outputDir_.empty())
    {
        return result;
    }

    // Usar versión corregida con polaridad alineada a convención estándar
    // La versión sin corrección tiene signos arbitrarios que dependen de la inicialización
    std::string path = outputDir_ + "/dwnominate_coordinates_all_periods_corrected.csv";
    std::ifstream file(path);

    // Si no existe la versión corregida, intentar con la original
    if (!file.is_open())
    {
        path = outputDir_ + "/dwnominate_coordinates_all_periods.csv";
        file.open(path);
    }
    if (!file.is_open())
    {
        std::cerr << "Advertencia: No se puede abrir " << path << std::endl;
        return result;
    }

    std::string line;
    bool isHeader = true;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        auto fields = splitCSVLine(line);

        if (isHeader)
        {
            isHeader = false;
            continue;
        }

        // Esperado: period,legislator,party,name,coord1D,coord2D,se1D,se2D,var1D,var2D,
        //           loglikelihood,numVotes,numErrors,GMP,...
        if (fields.size() < 14)
            continue;

        ReferenceCoordinates rc;
        rc.period = parseInt(fields[0]);
        rc.legislatorId = parseInt(fields[1]);
        rc.party = fields[2];
        rc.coord1D = parseDouble(fields[4]);
        rc.coord2D = parseDouble(fields[5]);
        rc.se1D = parseDouble(fields[6]);
        rc.se2D = parseDouble(fields[7]);
        rc.logLikelihood = parseDouble(fields[10]);
        rc.numVotes = parseInt(fields[11]);
        rc.numErrors = parseInt(fields[12]);
        rc.gmp = parseDouble(fields[13]);

        result.push_back(rc);
    }

    return result;
}

// Carga de parámetros de votaciones de referencia (output de R)
std::vector<ReferenceBillParams> CSVLoader::loadReferenceBillParams()
{
    std::vector<ReferenceBillParams> result;

    if (outputDir_.empty())
    {
        return result;
    }

    std::string path = outputDir_ + "/dwnominate_bill_parameters.csv";
    std::ifstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Advertencia: No se puede abrir " << path << std::endl;
        return result;
    }

    std::string line;
    bool isHeader = true;

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        auto fields = splitCSVLine(line);

        if (isHeader)
        {
            isHeader = false;
            continue;
        }

        // Esperado: session,ID,midpoint1D,midpoint2D,spread1D,spread2D
        if (fields.size() < 6)
            continue;

        ReferenceBillParams bp;
        bp.session = parseInt(fields[0]);
        bp.billId = parseInt(fields[1]);
        bp.isValid = !isNA(fields[2]) && !isNA(fields[3]) && !isNA(fields[4]) && !isNA(fields[5]);

        if (bp.isValid)
        {
            bp.midpoint1D = parseDouble(fields[2]);
            bp.midpoint2D = parseDouble(fields[3]);
            bp.spread1D = parseDouble(fields[4]);
            bp.spread2D = parseDouble(fields[5]);
        }

        result.push_back(bp);
    }

    return result;
}
