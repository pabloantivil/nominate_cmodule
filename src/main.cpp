/**
 * Punto de entrada principal para DW-NOMINATE C++
 *
 * Uso: nominate_cmodule [input_dir] [output_dir] [num_iterations]
 */

#include "dwnominate.hpp"
#include "csv_loader.hpp"
#include <iostream>
#include <chrono>
#include <string>

int main(int argc, char *argv[])
{
    std::cout << "============================================================\n";
    std::cout << "  DW-NOMINATE C++ Implementation                            \n";
    std::cout << "  Optimized Version with Performance Fixes                  \n";
    std::cout << "============================================================\n\n";

    // Configuración por defecto
    std::string inputDir = "input_R";
    std::string outputDir = "output_R";
    int numIterations = 25;
    int numPeriods = 5;
    int numDimensions = 2;

    // Parsear argumentos
    if (argc >= 2)
        inputDir = argv[1];
    if (argc >= 3)
        outputDir = argv[2];
    if (argc >= 4)
        numIterations = std::stoi(argv[3]);
    if (argc >= 5)
        numPeriods = std::stoi(argv[4]);
    if (argc >= 6)
        numDimensions = std::stoi(argv[5]);

    std::cout << "Configuración:\n";
    std::cout << "  Directorio entrada:  " << inputDir << "\n";
    std::cout << "  Directorio salida:   " << outputDir << "\n";
    std::cout << "  Iteraciones:         " << numIterations << "\n";
    std::cout << "  Períodos:            " << numPeriods << "\n";
    std::cout << "  Dimensiones:         " << numDimensions << "\n\n";

    // Tiempo total
    auto startTotal = std::chrono::high_resolution_clock::now();

    // Cargar datos
    std::cout << "Cargando datos...\n";
    auto startLoad = std::chrono::high_resolution_clock::now();

    CSVLoader loader(inputDir, outputDir);
    DWNominateInput input;

    try
    {
        input = loader.loadInput(numPeriods);
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR cargando datos: " << e.what() << std::endl;
        return 1;
    }

    auto endLoad = std::chrono::high_resolution_clock::now();
    auto loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(endLoad - startLoad).count();
    std::cout << "Datos cargados en " << loadTime << " ms\n";
    std::cout << "  Legisladores: " << input.votes.getNumLegislators() << "\n";
    std::cout << "  Roll calls:   " << input.votes.getNumRollCalls() << "\n\n";

    // Configurar DW-NOMINATE
    DWNominateConfig config;
    config.numDimensions = numDimensions;
    config.temporalModel = 1; // 0=constante, 1=lineal, 2=cuadratico
    config.firstCongress = 0;
    config.lastCongress = numPeriods - 1;
    config.firstIteration = 1;
    config.lastIteration = numIterations;
    config.marginThreshold = 0.025;
    config.verbose = true;

    // Ejecutar algoritmo
    std::cout << "Ejecutando DW-NOMINATE...\n";
    std::cout << "============================================================\n";

    auto startRun = std::chrono::high_resolution_clock::now();

    DWNominate nominate(config, input);
    DWNominateResult result;

    try
    {
        result = nominate.run();
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR en ejecución: " << e.what() << std::endl;
        return 1;
    }

    auto endRun = std::chrono::high_resolution_clock::now();
    auto runTime = std::chrono::duration_cast<std::chrono::seconds>(endRun - startRun).count();

    std::cout << "============================================================\n";
    std::cout << "\nEjecución completada en " << runTime << " segundos\n";

    // Tiempo total
    auto endTotal = std::chrono::high_resolution_clock::now();
    auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(endTotal - startTotal).count();
    std::cout << "Tiempo total (incluyendo carga): " << totalTime << " segundos\n";

    // Resumen de resultados
    std::cout << "\nResumen:\n";
    std::cout << "  Log-likelihood final: " << result.finalLogLikelihood << "\n";
    std::cout << "  Iteraciones completadas: " << result.totalIterations << "\n";

    return 0;
}
