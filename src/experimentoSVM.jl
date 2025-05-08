module experimentoSVM

using CSV, DataFrames, Random, Statistics, StatsBase, HypothesisTests, JSON
include("P1_soluciones.jl")
using LIBSVM

export ejecutarSVM, realizar_anova

function normalizacionANN(inputs::Matrix{<:Real})
    min_vals = minimum(inputs, dims=1)
    max_vals = maximum(inputs, dims=1)
    return 2 * ((inputs .- min_vals) ./ (max_vals .- min_vals)) .- 1
end

function ejecutarSVM()
    df = CSV.read("P2/OTROS ARCHIVOS/alzheimers_limpio_balanced.csv", DataFrame)
    inputs = Matrix(select(df, Not(:Diagnosis)))
    targets = Vector(df.Diagnosis)
    X_norm = normalizacionANN(inputs)
    k = 10
    cv_indices = CSV.read("P2/OTROS ARCHIVOS/indices_crossval.csv", DataFrame).Fold 

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

    # DataFrame con tipos explícitos y flexibles
    resultados = DataFrame(
        Configuracion = String[],
        Kernel = String[],
        C = Float64[],
        Gamma = Union{Missing, Float64}[],
        Degree = Union{Missing, Int}[],
        Accuracy = Vector{Float64}[],
        Precision = Vector{Float64}[],
        Recall = Vector{Float64}[],
        Specificity = Vector{Float64}[],
        VPN = Vector{Float64}[],
        F1_Score = Vector{Float64}[],
        Tiempo = Float64[]
    )

    for config in svm_configs
        println("\nEvaluando SVM: ", config)
        kernel = config["kernel"]
        C = config["C"]
        gamma = haskey(config, "gamma") ? config["gamma"] : missing
        degree = get(config, "degree", 3)

        # Obtener resultados con manejo de errores
        try
            accs, precisions, recalls, specificities, vpns, _, f1s, _ = modelCrossValidation(:SVC, config, (X_norm, targets), cv_indices)

            # Conversión garantizada a Vector{Float64}
            acc_vec = accs isa Number ? [Float64(accs) for _ in 1:k] : Float64.(accs)
            f1_vec = f1s isa Number ? [Float64(f1s) for _ in 1:k] : Float64.(f1s)
            prec_vec= precisions isa Number ? [Float64(precisions) for _ in 1:k] : Float64.(precisions)
            recall_vec = recalls isa Number ? [Float64(recalls) for _ in 1:k] : Float64.(recalls)
            spec_vec = specificities isa Number ? [Float64(specificities) for _ in 1:k] : Float64.(specificities)
            vpn_vec = vpns isa Number ? [Float64(vpns) for _ in 1:k] : Float64.(vpns)

            nombre = "SVM_$(kernel)_C$(C)" *
                     (gamma !== missing ? "_gamma$(gamma)" : "") *
                     (degree !== missing ? "_degree$(degree)" : "")

            tiempo = @elapsed begin
                accs, precisions, recalls, specificities, vpns, _, f1s, _ = modelCrossValidation(:SVC, config, (X_norm, targets), cv_indices)
            end

            push!(resultados, (
                nombre,
                kernel,
                C,
                gamma,
                degree,
                acc_vec,
                prec_vec,
                recall_vec,
                spec_vec,
                vpn_vec,
                f1_vec,
                round(tiempo, digits=2)
            ))

        catch e
            @error "Error procesando configuración $config" exception=(e, catch_backtrace())
            continue
        end
    end
    temp = deepcopy(resultados)
    for col in [:Accuracy, :F1_Score, :Precision, :Recall, :Specificity, :VPN]
        temp[!, col] = [JSON.json(v) for v in temp[!, col]]
    end
     # Mostrar los resultados en la terminal
     println("\nResultados de cross-validation SVM:")
     println("Configuración | Kernel | C | Gamma | Degree | Accuracy | Precision | Recall | Specificity | VPN | F1-Score | Tiempo")
     for row in eachrow(resultados)
         println("$(row.Configuracion) | $(row.Kernel) | $(row.C) | $(row.Gamma) | $(row.Degree) | $(mean(row.Accuracy)) | $(mean(row.Precision)) | $(mean(row.Recall)) | $(mean(row.Specificity)) | $(mean(row.VPN)) | $(mean(row.F1_Score)) | $(row.Tiempo)")
     end

    CSV.write("P2/RESULTADOS/resultados_crossval_svm.csv", resultados)
    return resultados
end

function expand_metric_column(df::DataFrame, colname::Symbol)
    expanded_data = DataFrame(Configuracion = String[], Fold = Int[], Valor = Float64[])
    for row in eachrow(df)
        metric_values = row[colname]
        for (fold, val) in enumerate(metric_values)
            push!(expanded_data, (row.Configuracion, fold, val))
        end
    end
    return expanded_data
end

function realizar_anova(df::DataFrame, colname::Symbol = :Accuracy)
    expanded = expand_metric_column(df, colname)
    grupos = [expanded[expanded.Configuracion .== arch, :Valor] for arch in unique(expanded.Configuracion)]
    
    # Realizar la prueba ANOVA
    test_result = OneWayANOVATest(grupos...)
    
    # Imprimir los resultados del ANOVA en la terminal
    println("\nResultados del Test ANOVA para $colname:")
    println(test_result)
    
    return test_result
end

end