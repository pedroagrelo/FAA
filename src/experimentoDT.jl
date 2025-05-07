module experimentoDT

using CSV, DataFrames, DecisionTree, Statistics, HypothesisTests
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

    # Llamar al test ANOVA solo para Accuracy
    realizar_anova(resultados)

end

# Función para realizar el test ANOVA sobre Accuracy
function realizar_anova(df::DataFrame, colname::Symbol = :Accuracy)
    # Expandir los datos por fold (solo para Accuracy)
    expanded = expand_metric_column(df, colname)

    # Agrupar los valores por profundidad
    grupos = [expanded[expanded.Profundidad .== profundidad, :Valor] for profundidad in unique(expanded.Profundidad)]

    # Ejecutar el test ANOVA
    anova_result = OneWayANOVATest(grupos...)
    
    println("Resultados del test ANOVA para Accuracy: ")
    println(anova_result)
end

# Función para expandir los valores de la columna de métricas
function expand_metric_column(df::DataFrame, colname::Symbol)
    expanded_data = DataFrame(Profundidad = Int[], Fold = Int[], Valor = Float64[])
    
    for (i, row) in enumerate(eachrow(df))
        metric_values = row[colname]
        for (j, val) in enumerate(metric_values)
            push!(expanded_data, (df.Profundidad[i], j, val))
        end
    end
    return expanded_data
end

end # module
