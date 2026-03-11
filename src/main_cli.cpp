/**
 * @file main_cli.cpp
 * @brief Ejecutable CLI para DW-NOMINATE C++.
 *
 * Uso:
 *   dwnominate [opciones]
 *
 * Opciones:
 *   --input-dir=<path>     Directorio de votaciones (default: input_R)
 *   --output-dir=<path>    Directorio de salida CSV (default: output_cpp)
 *   --wnominate=<path>     Archivo de coordenadas iniciales WNOMINATE
 *   --bill-params=<path>   Archivo de parámetros de bill iniciales
 *   --model=<0|1|2|3>      Modelo temporal: 0=const, 1=linear, 2=quad, 3=cubic (default: 1)
 *   --iterations=<n>       Número de iteraciones (default: 4)
 *   --periods=<n>          Número de períodos (default: auto-detectar)
 *   --dimensions=<n>       Número de dimensiones espaciales (default: 2)
 *   --beta=<value>         Parámetro beta inicial (default: 5.9539)
 *   --w2=<value>           Peso de dimensión 2 inicial (default: 0.3463)
 *   --verbose              Mostrar progreso detallado
 *   --help                 Mostrar ayuda
 *
 * Ejemplo:
 *   dwnominate --model=1 --iterations=10 --verbose
 *   dwnominate --input-dir=datos --output-dir=resultados --periods=5
 */

#include "dwnominate.hpp"
#include "csv_loader.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

// Configuración CLI
struct CLIConfig
{
    std::string inputDir = "input_R";
    std::string outputDir = "output_cpp";
    std::string wnominatePath = "output_wnominate/wnominate_coordinates.csv";
    std::string billParamsPath = ""; // Si vacío, no cargar
    int temporalModel = 1;           // 0=const, 1=linear, 2=quad, 3=cubic
    int iterations = 4;
    int periods = 0; // 0 = auto-detectar
    int dimensions = 2;
    double beta = 5.9539;
    double w2 = 0.3463;
    bool verbose = false;
    bool showHelp = false;
    bool exportCorrected = true; // Exportar con corrección de polaridad
};

// Parsing de argumentos CLI
void printHelp(const char *programName)
{
    std::cout << "DW-NOMINATE C++ - Implementación optimizada\n\n";
    std::cout << "Uso: " << programName << " [opciones]\n\n";
    std::cout << "Opciones:\n";
    std::cout << "  --input-dir=<path>     Directorio de votaciones (default: input_R)\n";
    std::cout << "  --output-dir=<path>    Directorio de salida CSV (default: output_cpp)\n";
    std::cout << "  --wnominate=<path>     Coordenadas iniciales WNOMINATE\n";
    std::cout << "                         (default: output_wnominate/wnominate_coordinates.csv)\n";
    std::cout << "  --bill-params=<path>   Parámetros de bill iniciales (opcional)\n";
    std::cout << "  --model=<0|1|2|3>      Modelo temporal (default: 1)\n";
    std::cout << "                         0=constante, 1=lineal, 2=cuadrático, 3=cúbico\n";
    std::cout << "  --iterations=<n>       Número de iteraciones (default: 4)\n";
    std::cout << "  --periods=<n>          Número de períodos (default: auto-detectar)\n";
    std::cout << "  --dimensions=<n>       Dimensiones espaciales (default: 2)\n";
    std::cout << "  --beta=<value>         Parámetro beta inicial (default: 5.9539)\n";
    std::cout << "  --w2=<value>           Peso dimensión 2 inicial (default: 0.3463)\n";
    std::cout << "  --verbose              Mostrar progreso detallado\n";
    std::cout << "  --no-corrected         No exportar archivos con polaridad corregida\n";
    std::cout << "  --help                 Mostrar esta ayuda\n\n";
    std::cout << "Ejemplos:\n";
    std::cout << "  " << programName << " --model=1 --iterations=10 --verbose\n";
    std::cout << "  " << programName << " --input-dir=datos --periods=5\n";
    std::cout << "  " << programName << " --model=0 --iterations=4\n";
}

std::string getArgValue(const std::string &arg, const std::string &prefix)
{
    if (arg.find(prefix) == 0)
    {
        return arg.substr(prefix.length());
    }
    return "";
}

CLIConfig parseArguments(int argc, char *argv[])
{
    CLIConfig config;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            config.showHelp = true;
        }
        else if (arg == "--verbose" || arg == "-v")
        {
            config.verbose = true;
        }
        else if (arg == "--no-corrected")
        {
            config.exportCorrected = false;
        }
        else if (arg.find("--input-dir=") == 0)
        {
            config.inputDir = getArgValue(arg, "--input-dir=");
        }
        else if (arg.find("--output-dir=") == 0)
        {
            config.outputDir = getArgValue(arg, "--output-dir=");
        }
        else if (arg.find("--wnominate=") == 0)
        {
            config.wnominatePath = getArgValue(arg, "--wnominate=");
        }
        else if (arg.find("--bill-params=") == 0)
        {
            config.billParamsPath = getArgValue(arg, "--bill-params=");
        }
        else if (arg.find("--model=") == 0)
        {
            config.temporalModel = std::stoi(getArgValue(arg, "--model="));
        }
        else if (arg.find("--iterations=") == 0)
        {
            config.iterations = std::stoi(getArgValue(arg, "--iterations="));
        }
        else if (arg.find("--periods=") == 0)
        {
            config.periods = std::stoi(getArgValue(arg, "--periods="));
        }
        else if (arg.find("--dimensions=") == 0)
        {
            config.dimensions = std::stoi(getArgValue(arg, "--dimensions="));
        }
        else if (arg.find("--beta=") == 0)
        {
            config.beta = std::stod(getArgValue(arg, "--beta="));
        }
        else if (arg.find("--w2=") == 0)
        {
            config.w2 = std::stod(getArgValue(arg, "--w2="));
        }
        else
        {
            std::cerr << "Advertencia: argumento desconocido: " << arg << "\n";
        }
    }

    return config;
}

int detectNumPeriods(const std::string &inputDir)
{
    int maxPeriod = 0;

    try
    {
        for (const auto &entry : fs::directory_iterator(inputDir))
        {
            std::string filename = entry.path().filename().string();

            // Buscar archivos votes_matrix_p<N>.csv
            if (filename.find("votes_matrix_p") == 0 && filename.find(".csv") != std::string::npos)
            {
                // Extraer número del nombre
                size_t start = std::string("votes_matrix_p").length();
                size_t end = filename.find(".csv");
                if (end > start)
                {
                    std::string numStr = filename.substr(start, end - start);
                    try
                    {
                        int period = std::stoi(numStr);
                        maxPeriod = std::max(maxPeriod, period);
                    }
                    catch (...)
                    {
                        // Ignorar archivos con nombres inválidos
                    }
                }
            }
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Error al leer directorio " << inputDir << ": " << e.what() << "\n";
        return 0;
    }

    return maxPeriod;
}

void exportCoordinatesAllPeriods(const std::string &path,
                                 const DWNominateResult &result,
                                 int numPeriods)
{
    std::ofstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Error: No se puede crear " << path << std::endl;
        return;
    }

    if (!result.hasTemporalCoefficients())
    {
        std::cerr << "Error: No hay coeficientes temporales disponibles.\n";
        return;
    }

    file << std::fixed << std::setprecision(15);
    file << "legislator_id,period,coord1D,coord2D\n";

    int exported = 0;
    for (int period = 1; period <= numPeriods; ++period)
    {
        for (int legId : result.legislatorUniqueIds)
        {
            Eigen::VectorXd coords = result.getCoordinatesAtPeriod(legId, period);
            if (coords.size() >= 2)
            {
                file << legId << "," << period << ","
                     << coords(0) << "," << coords(1) << "\n";
                exported++;
            }
        }
    }

    std::cout << "Exportado: " << path << " (" << exported << " registros)\n";
}

void exportCoordinatesAllPeriodsCorrected(const std::string &path,
                                          const DWNominateResult &result,
                                          int numPeriods)
{
    std::ofstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Error: No se puede crear " << path << std::endl;
        return;
    }

    if (!result.hasTemporalCoefficients())
    {
        std::cerr << "Error: No hay coeficientes temporales disponibles.\n";
        return;
    }

    file << std::fixed << std::setprecision(15);
    file << "legislator_id,period,coord1D,coord2D\n";

    int exported = 0;
    for (int period = 1; period <= numPeriods; ++period)
    {
        for (int legId : result.legislatorUniqueIds)
        {
            Eigen::VectorXd coords = result.getCoordinatesAtPeriod(legId, period);
            if (coords.size() >= 2)
            {
                // Corrección de polaridad: multiplicar por -1
                file << legId << "," << period << ","
                     << (-coords(0)) << "," << (-coords(1)) << "\n";
                exported++;
            }
        }
    }

    std::cout << "Exportado (polaridad corregida): " << path << " (" << exported << " registros)\n";
}

void exportBillParameters(const std::string &path,
                          const DWNominateResult &result,
                          int numRollCalls)
{
    std::ofstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Error: No se puede crear " << path << std::endl;
        return;
    }

    file << std::fixed << std::setprecision(15);
    file << "rollcall_id,midpoint1D,midpoint2D,spread1D,spread2D\n";

    for (int i = 0; i < numRollCalls; ++i)
    {
        file << i << ","
             << result.rollCallMidpoints(i, 0) << ","
             << (result.rollCallMidpoints.cols() > 1 ? result.rollCallMidpoints(i, 1) : 0.0) << ","
             << result.rollCallSpreads(i, 0) << ","
             << (result.rollCallSpreads.cols() > 1 ? result.rollCallSpreads(i, 1) : 0.0) << "\n";
    }

    std::cout << "Exportado: " << path << " (" << numRollCalls << " roll calls)\n";
}

void exportSummary(const std::string &path,
                   const DWNominateResult &result,
                   const CLIConfig &config,
                   double elapsedSeconds)
{
    std::ofstream file(path);
    if (!file.is_open())
    {
        std::cerr << "Error: No se puede crear " << path << std::endl;
        return;
    }

    file << "parameter,value\n";
    file << "log_likelihood," << std::fixed << std::setprecision(6) << result.finalLogLikelihood << "\n";
    file << "iterations," << result.totalIterations << "\n";
    file << "valid_votes," << result.totalValidVotes << "\n";
    file << "correct_classifications," << result.classificationAfter << "\n";
    double classPct = result.totalValidVotes > 0 ? (100.0 * result.classificationAfter / result.totalValidVotes) : 0.0;
    file << "classification_pct," << std::setprecision(4) << classPct << "\n";
    file << "w1," << std::setprecision(6) << result.weights(0) << "\n";
    file << "w2," << result.weights(1) << "\n";
    file << "beta," << result.weights(2) << "\n";
    file << "temporal_model," << config.temporalModel << "\n";
    file << "dimensions," << config.dimensions << "\n";
    file << "periods," << config.periods << "\n";
    file << "elapsed_seconds," << std::setprecision(2) << elapsedSeconds << "\n";

    std::cout << "Exportado: " << path << "\n";
}

// Función principal
int main(int argc, char *argv[])
{
    // Parsear argumentos
    CLIConfig config = parseArguments(argc, argv);

    if (config.showHelp)
    {
        printHelp(argv[0]);
        return 0;
    }

    // Banner inicial
    std::cout << "============================================================\n";
    std::cout << "  DW-NOMINATE C++ CLI\n";
    std::cout << "============================================================\n\n";

    // Auto-detectar períodos si no se especificó
    if (config.periods == 0)
    {
        config.periods = detectNumPeriods(config.inputDir);
        if (config.periods == 0)
        {
            std::cerr << "ERROR: No se pudieron detectar archivos de votación en "
                      << config.inputDir << "\n";
            std::cerr << "       Especifique --periods=<n> o verifique el directorio.\n";
            return 1;
        }
        if (config.verbose)
        {
            std::cout << "Períodos auto-detectados: " << config.periods << "\n";
        }
    }

    // Crear directorio de salida si no existe
    try
    {
        fs::create_directories(config.outputDir);
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "ERROR: No se puede crear directorio de salida: " << e.what() << "\n";
        return 1;
    }

    // Mostrar configuración
    std::cout << "Configuracion:\n";
    std::cout << "  Input:      " << config.inputDir << "\n";
    std::cout << "  Output:     " << config.outputDir << "\n";
    std::cout << "  WNOMINATE:  " << config.wnominatePath << "\n";
    std::cout << "  Modelo:     " << config.temporalModel
              << (config.temporalModel == 0   ? " (constante)"
                  : config.temporalModel == 1 ? " (lineal)"
                  : config.temporalModel == 2 ? " (cuadratico)"
                                              : " (cubico)")
              << "\n";
    std::cout << "  Iteraciones:" << config.iterations << "\n";
    std::cout << "  Periodos:   " << config.periods << "\n";
    std::cout << "  Dimensiones:" << config.dimensions << "\n";
    std::cout << "  Beta:       " << config.beta << "\n";
    std::cout << "  W2:         " << config.w2 << "\n\n";

    // Iniciar cronómetro
    auto startTime = std::chrono::high_resolution_clock::now();

    // Cargar datos
    std::cout << "Cargando datos...\n";

    // Determinar directorio de referencia R basado en modelo temporal
    std::string refDir = "output_R_dwnominate_model" + std::to_string(config.temporalModel);
    if (!config.billParamsPath.empty())
    {
        fs::path billPath(config.billParamsPath);
        refDir = billPath.parent_path().string();
    }

    std::cout << "  Ref. R:     " << refDir << "\n";

    CSVLoader loader(config.inputDir, refDir);
    DWNominateInput input;

    // Configurar inicialización
    InitializationConfig initConfig;
    initConfig.beta = config.beta;
    initConfig.w2 = config.w2;
    initConfig.useWNominateStart = true;
    initConfig.wnominatePath = config.wnominatePath;

    try
    {
        input = loader.loadInput(config.periods, initConfig);
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR cargando datos: " << e.what() << "\n";
        return 1;
    }

    std::cout << "  Legisladores: " << input.votes.getNumLegislators() << "\n";
    std::cout << "  Roll calls:   " << input.votes.getNumRollCalls() << "\n\n";

    // Configurar DW-NOMINATE
    DWNominateConfig dwConfig;
    dwConfig.numDimensions = config.dimensions;
    dwConfig.temporalModel = config.temporalModel;
    dwConfig.firstCongress = 0;
    dwConfig.lastCongress = config.periods - 1;
    dwConfig.firstIteration = 1;
    dwConfig.lastIteration = config.iterations;
    dwConfig.marginThreshold = 0.025;
    dwConfig.verbose = config.verbose;
    dwConfig.fixGlobalParams = false;
    dwConfig.fixRollCalls = false;
    dwConfig.fixLegislators = false;

    // Ejecutar algoritmo
    std::cout << "Ejecutando DW-NOMINATE...\n";
    if (config.verbose)
    {
        std::cout << "============================================================\n";
    }

    DWNominate nominate(dwConfig, input);
    DWNominateResult result;

    try
    {
        result = nominate.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR en ejecucion: " << e.what() << "\n";
        return 1;
    }

    // Calcular tiempo transcurrido
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    double elapsedSeconds = elapsed.count() / 1000.0;

    // Resumen
    if (config.verbose)
    {
        std::cout << "============================================================\n";
    }
    double classPctMain = result.totalValidVotes > 0 ? (100.0 * result.classificationAfter / result.totalValidVotes) : 0.0;
    std::cout << "\nResultados:\n";
    std::cout << "  Log-likelihood: " << std::fixed << std::setprecision(4) << result.finalLogLikelihood << "\n";
    std::cout << "  Iteraciones:    " << result.totalIterations << "\n";
    std::cout << "  Clasificacion:  " << result.classificationAfter << "/" << result.totalValidVotes
              << " (" << std::setprecision(2) << classPctMain << "%)\n";
    std::cout << "  W1=" << std::setprecision(4) << result.weights(0)
              << ", W2=" << result.weights(1)
              << ", Beta=" << result.weights(2) << "\n";
    std::cout << "  Tiempo: " << std::setprecision(1) << elapsedSeconds << "s\n\n";

    // Exportar resultados
    std::cout << "Exportando resultados...\n";

    exportCoordinatesAllPeriods(
        config.outputDir + "/cpp_coordinates_all_periods.csv",
        result, config.periods);

    if (config.exportCorrected)
    {
        exportCoordinatesAllPeriodsCorrected(
            config.outputDir + "/cpp_coordinates_all_periods_corrected.csv",
            result, config.periods);
    }

    exportBillParameters(
        config.outputDir + "/cpp_bill_parameters.csv",
        result, input.votes.getNumRollCalls());

    exportSummary(
        config.outputDir + "/cpp_summary.csv",
        result, config, elapsedSeconds);

    std::cout << "\nCompletado exitosamente.\n";

    return 0;
}
