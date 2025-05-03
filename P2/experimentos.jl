using CSV, DataFrames, Random, Statistics, StatsBase, HypothesisTests
include("P1_soluciones.jl")

# ------------------------------------------
# Normalización Min-Max para RNA
# ------------------------------------------
function normalizacionANN(inputs::Matrix{<:Real})
    return normalizeMinMax(inputs)
end



# ------------------------------------------
# Ejecutar RNA con validación cruzada k=10
# ------------------------------------------
function ejecutarRNA()

    # 1. Leer dataset limpio
    df = CSV.read("alzheimers_limpio_balanced.csv", DataFrame)
    inputs = Matrix(select(df, Not(:Diagnosis)))
    targets = Vector(df.Diagnosis)

    # 2. Normalizar entradas
    X_norm = normalizacionANN(inputs)

    # 3. Generar índices de validación cruzada estratificada
    k = 10
    cv_indices = crossvalidation(targets, k)
    df_indices = DataFrame(Fold = cv_indices)
    CSV.write("indices_validacion_cruzada.csv", df_indices)


    resultados = DataFrame(
    Arquitectura = String[],
    AccuracyMean = Vector{Vector{Float64}}(),
    AccuracyStd  = Vector{Vector{Float64}}(),
    F1Mean       = Vector{Vector{Float64}}(),
    F1Std        = Vector{Vector{Float64}}(),
    PrecisionMean = Vector{Vector{Float64}}(),
    PrecisionStd  = Vector{Vector{Float64}}(),
    RecallMean    = Vector{Vector{Float64}}(),
    RecallStd     = Vector{Vector{Float64}}(),
    SpecificityMean = Vector{Vector{Float64}}(),
    SpecificityStd = Vector{Vector{Float64}}(),
    NPVMean = Vector{Vector{Float64}}(),
    NPVStd =Vector{Vector{Float64}}()

    )


    # 4. Probar diferentes arquitecturas
    architectures = [
        [5],
        [8],
        [12],
        [15],
        [8, 5],
        [12, 8],
        [15, 12],
        [18, 15]
    ]

    #ojo que quede o 10 porcento de test por que fago un k fold 

    # 5x2 crossvalidatrion 

    #2 3 4 e 5 elementos para favvorecer eses hiperpplanos de 2 en 2 basicamente 10, 6, 16  

    conf_matrix_final = zeros(Int64, 2, 2)  # Inicializa la matriz global

    for arch in architectures
        println("\n Evaluando arquitectura: ", arch ) 
        (acc_mean, acc_std),
        (_, _),  # Error rate  no se usa
        (rec_mean, rec_std),
        (spec_mean, spec_std),  
        (prec_mean, prec_std),
        (npv_mean, npv_std), 
        (f1_mean, f1_std),
        conf_matrix =   ANNCrossValidation(
            arch, (X_norm, targets), cv_indices;
                numExecutions=5,
                maxEpochs=100,
                learningRate=0.01,
                validationRatio=0.1,
                maxEpochsVal=20
            )
    

        push!(resultados, (
        Arquitectura = string(arch),
        AccuracyMean = acc_mean,
        AccuracyStd  = acc_std,
        F1Mean       = f1_mean,
        F1Std        = f1_std,
        PrecisionMean = prec_mean,
        PrecisionStd  = prec_std,
        RecallMean    = rec_mean,
        RecallStd     = rec_std,
        SpecificityMean = spec_mean,
        SpecificityStd = spec_std,
        NPVMean = npv_mean,
        NPVStd = npv_std
        ))


        # # Acumula la matriz de confusión de esta arquitectura
        conf_matrix_final = conf_matrix  # Aquí tomamos la matriz de la última arquitectura evaluada
    end


    println("\nResultados resumen:")
    show(resultados, allcols=true)

    # Guardar si quieres
    CSV.write("resultados_crossval_rna.csv", resultados)

    return resultados, conf_matrix_final

end;


function expand_metric_column(df::DataFrame, colname::Symbol)
    expanded_data = DataFrame(Arquitectura = String[], Fold = Int[], Valor = Float64[])
    
    for (i, row) in enumerate(eachrow(df))
        metric_values = row[colname]
        for (j, val) in enumerate(metric_values)
            push!(expanded_data, (string(df.Arquitectura[i]), j, val))
        end
    end
    return expanded_data
end

function realizar_anova(df::DataFrame, colname::Symbol = :AccuracyMean)
    # Expandir los datos por fold
    expanded = expand_metric_column(df, colname)

    # Agrupar los valores por arquitectura
    grupos = [expanded[expanded.Arquitectura .== arch, :Valor] for arch in unique(expanded.Arquitectura)]

    # Ejecutar el test ANOVA
    anova_result = OneWayANOVATest(grupos...)
    
    return anova_result
end

