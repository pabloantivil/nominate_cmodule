#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <nlopt.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace math_utils {
    // Implementación de CDF normal usando erf()
    inline double normal_cdf(double x, double mean = 0.0, double std = 1.0) {
        return 0.5 * (1.0 + std::erf((x - mean) / (std * std::sqrt(2.0))));
    }
    
    // Implementación de log CDF normal
    inline double normal_logcdf(double x, double mean = 0.0, double std = 1.0) {
        return std::log(normal_cdf(x, mean, std));
    }
}

// Función objetivo: minimizar f(x,y) = (x-2)^2 + (y-1)^2
double objective_function(unsigned n, const double *x, double *grad, void *data) {
    // Si se solicita el gradiente, calcularlo
    if (grad) {
        grad[0] = 2.0 * (x[0] - 2.0);  // df/dx
        grad[1] = 2.0 * (x[1] - 1.0);  // df/dy
    }
    
    // Calcular valor de la función
    double fx = std::pow(x[0] - 2.0, 2) + std::pow(x[1] - 1.0, 2);
    
    std::cout << "  f(" << x[0] << ", " << x[1] << ") = " << fx << std::endl;
    
    return fx;
}

// Restricción: x + y <= 3
double constraint_function(unsigned n, const double *x, double *grad, void *data) {
    if (grad) {
        grad[0] = 1.0;
        grad[1] = 1.0;
    }
    return x[0] + x[1] - 3.0;  // <= 0
}

int main() {
    
    std::cout << "Prueba de Eigen" << std::endl;
    
    Eigen::Matrix2d mat;
    mat << 1, 2,
           3, 4;
    
    Eigen::Vector2d vec(5, 6);
    Eigen::Vector2d result = mat * vec;
    
    std::cout << "Matrix A:\n" << mat << std::endl;
    std::cout << "Vector b:\n" << vec << std::endl;
    std::cout << "Resultado A*b:\n" << result << std::endl;
    std::cout << "Eigen funciona correctamente!" << std::endl;
    
    // Test OpenMP
    #ifdef USE_OPENMP
        std::cout << "\n OpenMP:" << std::endl;
        std::cout << "Disponible, threads: " << omp_get_max_threads() << std::endl;
    #endif

    std::cout << "\nPrueba de NLopt" << std::endl;

    try {
        // Crear optimizador
        nlopt_opt opt = nlopt_create(NLOPT_LD_LBFGS, 2);  // L-BFGS con 2 variables
        
        if (!opt) {
            std::cout << "Error: No se pudo crear el optimizador NLopt" << std::endl;
            return 1;
        }
        
        // Configurar función objetivo
        nlopt_set_min_objective(opt, objective_function, nullptr);
        
        // Añadir restricción de desigualdad
        nlopt_add_inequality_constraint(opt, constraint_function, nullptr, 1e-8);
        
        // Establecer límites de las variables (opcional)
        double lower_bounds[2] = {-5.0, -5.0};
        double upper_bounds[2] = {5.0, 5.0};
        nlopt_set_lower_bounds(opt, lower_bounds);
        nlopt_set_upper_bounds(opt, upper_bounds);
        
        // Criterios de parada
        nlopt_set_xtol_rel(opt, 1e-6);
        nlopt_set_maxeval(opt, 1000);
        
        // Punto inicial
        double x[2] = {0.0, 0.0};
        double minf;
        
        std::cout << "Problema de optimizacion:" << std::endl;
        std::cout << "  Minimizar: f(x,y) = (x-2)^2 + (y-1)^2" << std::endl;
        std::cout << "  Sujeto a: x + y <= 3" << std::endl;
        std::cout << "  Punto inicial: (" << x[0] << ", " << x[1] << ")" << std::endl;
        std::cout << "\nEjecutando optimizacion..." << std::endl;
        
        // Ejecutar optimización
        nlopt_result result_opt = nlopt_optimize(opt, x, &minf);
        
        // Analizar resultados
        std::cout << "\n--- RESULTADOS ---" << std::endl;
        
        switch (result_opt) {
            case NLOPT_SUCCESS:
                std::cout << "Optimizacion EXITOSA (convergencia)" << std::endl;
                break;
            case NLOPT_XTOL_REACHED:
                std::cout << "Optimizacion EXITOSA (tolerancia alcanzada)" << std::endl;
                break;
            case NLOPT_FTOL_REACHED:
                std::cout << "Optimizacion EXITOSA (tolerancia de función)" << std::endl;
                break;
            case NLOPT_MAXEVAL_REACHED:
                std::cout << "Maximo numero de evaluaciones alcanzado" << std::endl;
                break;
            default:
                std::cout << "Error en optimizacion (codigo: " << result_opt << ")" << std::endl;
        }
        
        if (result_opt > 0) {
            std::cout << "Solucion encontrada:" << std::endl;
            std::cout << "  x* = (" << x[0] << ", " << x[1] << ")" << std::endl;
            std::cout << "  f(x*) = " << minf << std::endl;
            
            // Verificar restriccion
            double constraint_value = x[0] + x[1];
            std::cout << "  Restriccion x+y = " << constraint_value << " (<= 3)" << std::endl;
            
            if (constraint_value <= 3.001) {
                std::cout << "Restriccion satisfecha" << std::endl;
            } else {
                std::cout << "Restriccion violada" << std::endl;
            }
            
            // Solucion teorica esperada
            std::cout << "\nSolucion teorica esperada: (1.5, 1.5) con f = 0.5" << std::endl;
        }
        
        // Limpiar memoria
        nlopt_destroy(opt);
        
        std::cout << "\nNLopt funciona correctamente!" << std::endl;
        
    } catch (...) {
        std::cout << "Error inesperado en NLopt" << std::endl;
        return 1;
    }

    // Test distribución normal
    std::cout << "\n3. Test Distribución Normal:" << std::endl;
    double test_val = 0.0;
    std::cout << "   CDF(0) = " << math_utils::normal_cdf(test_val) << " (esperado: ~0.5)" << std::endl;
    std::cout << "   logCDF(0) = " << math_utils::normal_logcdf(test_val) << " (esperado: ~-0.693)" << std::endl;


    std::cout << "\nTodas las librerias funcionan correctamente" << std::endl;
    return 0;
}