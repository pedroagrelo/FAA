module experimentoSVM

using CSV, DataFrames, Random, Statistics, StatsBase, HypothesisTests, JSON, Distributions
include("P1_soluciones.jl")
using LIBSVM

export ejecutarSVM, realizar_anova


function ejecutarSVM()
    df = CSV.read("P2/OTROS_ARCHIVOS/alzheimers_limpio_balanced.csv", DataFrame)
    inputs = Matrix(select(df, Not(:Diagnosis)))
    targets = Vector(df.Diagnosis)
    X_norm = normalizeZeroMean(inputs) #Normalizacion z-score

    k = 10
    cv_indices = CSV.read("P2/OTROS_ARCHIVOS/indices_crossval.csv", DataFrame).Fold 

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

#Test ANOVA + Tukey
function realizar_anova(df::DataFrame, colname::Symbol = :Accuracy)
    expanded = expand_metric_column(df, colname)
    grupos = [expanded[expanded.Configuracion .== arch, :Valor] for arch in unique(expanded.Configuracion)]
    # Realizar la prueba ANOVA
    anova_result = OneWayANOVATest(grupos...)
    
    # Imprimir los resultados del ANOVA en la terminal
    println("\nResultados del Test ANOVA para $colname:")
    println(anova_result)

    if pvalue(anova_result) < 0.05
        println("-> Se rechaza H₀ con un p-valor de $(round(pvalue(anova_result), digits=4)). Se realizarán comparaciones múltiples.")
        realizar_tukey_hsd(df, colname)
    else
        println("→ No se rechaza H₀. No se hacen comparaciones múltiples.")
    end
end

function realizar_tukey_hsd(df::DataFrame, colname::Symbol = :Accuracy)
    println("\nComparaciones múltiples tipo Tukey HSD:")
    expanded = expand_metric_column(df, colname)
    grupos = unique(expanded.Configuracion)
    α = 0.05

    for i = 1:length(grupos)-1
        for j = i+1:length(grupos)
            g1, g2 = grupos[i], grupos[j]
            vals1 = expanded[expanded.Configuracion .== g1, :Valor]
            vals2 = expanded[expanded.Configuracion .== g2, :Valor]

            diff = mean(vals1) - mean(vals2)
            pooled_var = (var(vals1) + var(vals2)) / 2
            se = sqrt(pooled_var * (1/length(vals1) + 1/length(vals2)))
            t_stat = abs(diff) / se

            dfree = length(vals1) + length(vals2) - 2
            critical_t = quantile(TDist(dfree), 1 - α/2)
            significant = t_stat > critical_t
            resultado = significant ? "DIFERENCIA SIGNIFICATIVA" : "sin diferencia"

            println("Comparación $g1 vs $g2: t = $(round(t_stat, digits=3)) (umbral = $(round(critical_t, digits=3))) → $resultado")
        end
    end
end

function seleccionar_mejor_configuracion_svm(df::DataFrame)
    # Encuentra el índice de la mejor accuracy promedio
    promedios = [mean(a) for a in df.Accuracy]
    desvios = [std(a) for a in df.Accuracy]
    mejor_indice = argmax(promedios)
    mejor_accuracy = promedios[mejor_indice]
    mejor_std = desvios[mejor_indice]

    println("\nLa mejor configuración de SVM es: ", df.Configuracion[mejor_indice])
    println("Con un Accuracy promedio de: ", round(mejor_accuracy, digits=4), " ± ", round(mejor_std, digits=4))
end
end