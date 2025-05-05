using CSV, DataFrames, Random, Statistics
include("P1_soluciones.jl") # Asegúrate de que aquí esté definida `crossvalidation` y `normalizeMinMax`
using LIBSVM

# ------------------------------------------
# Normalización Min-Max
# ------------------------------------------
function normalizacionANN(inputs::Matrix{<:Real})
    return normalizeMinMax(inputs)  # Debe estar en P1_soluciones.jl
end

# ------------------------------------------
# Ejecutar SVM con validación cruzada k=10
# ------------------------------------------
function ejecutarSVM()

    # 1. Leer dataset limpio
    df = CSV.read("alzheimers_limpio_balanced.csv", DataFrame)
    inputs = Matrix(select(df, Not(:Diagnosis)))
    targets = Vector(df.Diagnosis)

    # 2. Normalizar entradas
    X_norm = normalizacionANN(inputs)

    # 3. Generar índices de validación cruzada estratificada
    k = 10
    cv_indices = crossvalidation(targets, k)

    # 4. Configuraciones a evaluar
    svm_configs = [
        Dict("kernel" => "linear", "C" => 0.1),
        Dict("kernel" => "linear", "C" => 1.0),
        Dict("kernel" => "rbf", "C" => 0.5, "gamma" => 0.01),
        Dict("kernel" => "rbf", "C" => 2.0, "gamma" => 0.1),
        Dict("kernel" => "rbf", "C" => 1.0, "gamma" => 0.2),
        Dict("kernel" => "poly", "C" => 1.0, "gamma" => 0.3, "degree" => 5),
        Dict("kernel" => "poly", "C" => 0.5, "gamma" => 0.2, "degree" => 4),
        Dict("kernel" => "sigmoid", "C" => 1.5, "gamma" => 0.3)
    ]

    # 5. Crear DataFrame de resultados
    resultados = DataFrame(Configuración=String[], Kernel=String[], C=Float64[], Gamma=Any[], Degree=Any[], Accuracy=Float64[], F1_Score=Float64[], Tiempo=Float64[])

    for config in svm_configs
        println("\nEvaluando SVM: ", config)
        kernel = config["kernel"]
        C = config["C"]
        gamma = get(config, "gamma", "auto")
        degree = get(config, "degree", "auto")

        modelo = SVMClassifier(
            kernel = kernel == "linear" ? LIBSVM.Kernel.Linear :
                     kernel == "rbf"    ? LIBSVM.Kernel.RadialBasis :
                     kernel == "poly"   ? LIBSVM.Kernel.Polynomial :
                                          LIBSVM.Kernel.Sigmoid,
            cost = Float64(C),
            gamma = gamma == "auto" ? -1.0 : Float64(gamma),
            degree = degree == "auto" ? Int32(3) : Int32(degree)
        )

        tiempo = @elapsed begin
            (acc, _), (_, _), (_, _), (_, _), (_, _), (_, _), (f1, _), _ =
                modelCrossValidation(:SVC, config, (X_norm, targets), cv_indices)
        end

        nombre = "SVM_$(kernel)_C$(C)" *
                 (gamma != "auto" ? "_gamma$(gamma)" : "") *
                 (degree != "auto" ? "_degree$(degree)" : "")

        push!(resultados, (
            nombre,
            kernel,
            C,
            gamma,
            degree,
            round(acc, digits=4),
            round(f1, digits=4),
            round(tiempo, digits=2)
        ))
    end

    println("\nResumen resultados:")
    show(resultados, allcols=true)

    CSV.write("resultados_crossval_svm.csv", resultados)

end

# Ejecutar
ejecutarSVM()
