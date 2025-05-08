using CSV, DataFrames, Random, Statistics
include("P2/src/P1_soluciones.jl")  # Aquí están las funciones como crossvalidation y normalizeMinMax
using DecisionTree

# ------------------------------------------
# Normalización Min-Max
# ------------------------------------------
function normalizacionDecisionTree(inputs::Matrix{<:Real})
    return normalizeMinMax(inputs)
end

# ------------------------------------------
# Ejecutar Árbol de Decisión con validación cruzada k=10
# ------------------------------------------
function ejecutarDecisionTree()

    # 1. Leer dataset limpio
    df = CSV.read("P2/OTROS ARCHIVOS/alzheimers_limpio_balanced.csv", DataFrame)
    inputs = Matrix(select(df, Not(:Diagnosis)))
    targets = Vector(df.Diagnosis)

    # 2. Normalizar entradas
    X_norm = normalizacionDecisionTree(inputs)

    # 3. Generar índices de validación cruzada estratificada
    k = 10
    cv_indices = crossvalidation(targets, k)

    # 4. Profundidades a evaluar
    profundidades = [2, 4, 6, 8, 10, 12]

    resultados = DataFrame(
        Profundidad = Int[],
        Accuracy = String[],
        F1_Score = String[],
        Precision = String[],
        Recall = String[],
        Specificity = String[],
        NPV = String[],
        Tiempo = Float64[]
    )

    for profundidad in profundidades
        println("\nEvaluando árbol con profundidad = ", profundidad)

        tiempo = @elapsed begin
            accs, _, recalls, specificities, precisions, npvs, f1s, _ =
                modelCrossValidation(:DecisionTreeClassifier,
                    Dict("max_depth" => profundidad),
                    (X_norm, targets), cv_indices)
        end

        push!(resultados, (
            profundidad,
            string(accs),
            string(f1s),
            string(precisions),
            string(recalls),
            string(specificities),
            string(npvs),
            round(tiempo, digits=2)
        ))
    end

    println("\nResumen resultados Árbol de Decisión:")
    show(resultados, allcols=true)

    CSV.write("P2/RESULTADOS/resultados_crossval_dt.csv", resultados)

end

# Ejecutar
ejecutarDecisionTree()
