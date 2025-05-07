module experimentoDT

using CSV, DataFrames, DecisionTree, Statistics
include("P1_soluciones.jl") # Asegúrate de tener crossvalidation y normalizeMinMax disponibles

function normalizacionDecisionTree(inputs::Matrix{<:Real})
    return normalizeMinMax(inputs)
end

function ejecutarDecisionTree()
    df = CSV.read("alzheimers_limpio_balanced.csv", DataFrame)
    inputs = Matrix(select(df, Not(:Diagnosis)))
    targets = Vector(df.Diagnosis)
    X_norm = normalizacionDecisionTree(inputs)
    k = 10
    cv_indices = crossvalidation(targets, k)
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
        println("Profundidad: $profundidad")
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

    CSV.write("resultados_crossval_dt.csv", resultados)
end

end # module
