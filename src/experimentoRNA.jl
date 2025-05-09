module experimentoRNA

using CSV, DataFrames, Random, Statistics, StatsBase, HypothesisTests
include("P1_soluciones.jl")


export ejecutarRNA, realizar_anova
# ------------------------------------------
# Normalización Min-Max para RNA
# ------------------------------------------
function normalizacionANN(inputs::Matrix{<:Real})
    return normalizeMinMax(inputs)
end

#maybe non fai falta, evitar data leakage normalizando antes de particionar o dataset en entrenamiento e test, e eso e antes de crossvaliation 
# PROBEI TAMEN a estratificada vs a normal, pretty much the same no meu caso polo menos 


# ------------------------------------------
# Ejecutar RNA con validación cruzada k=10
# ------------------------------------------
function ejecutarRNA()

    # 1. Leer dataset limpio
    df = CSV.read("P2/OTROS_ARCHIVOS/alzheimers_limpio_balanced.csv", DataFrame)
    inputs = Matrix(select(df, Not(:Diagnosis)))
    targets = Vector(df.Diagnosis)

    # 2. Normalizar entradas
    #X_norm = normalizacionANN(inputs)

    # 3. Generar índices de validación cruzada estratificada
    k = 10
    cv_indices = CSV.read("P2/OTROS ARCHIVOS/indices_crossval2.csv", DataFrame).Fold  #Cargar indices de cv comunes
    df_indices = DataFrame(Fold = cv_indices)
    #~CSV.write("P2/OTROS ARCHIVOS/indices_validacion_cruzada.csv", df_indices)

    # Si hiciera repeticiones 3 veces vector para que cubra las repeticiones, la arquitectura y los folds
    # AccuracyMean[i][j] → fold k del experimento j para la arquitectura i.
    # Como solo hacemos una repeticion me vale con el 2 vector, arquitecra y folds 
    # AccuracyMean[i] → 10 valores de accuracy para la arquitectura i.
    resultados = DataFrame(
    Arquitectura = String[],
    AccuracyMean = Vector{Vector{Float64}}(),
    F1Mean       = Vector{Vector{Float64}}(),
    PrecisionMean = Vector{Vector{Float64}}(),
    RecallMean    = Vector{Vector{Float64}}(),
    SpecificityMean = Vector{Vector{Float64}}(),
    NPVMean = Vector{Vector{Float64}}(),
    )


    # 4. Probar diferentes arquitecturas
    architectures = [
        [12],
        [16],
        [24],
        [32],
        [48],
        [16, 12],
        [24, 16],
        [32, 24],
        [48, 32]
    ]

    #ojo que quede o 10 porcento de test por que fago un k fold 

    # 5x2 crossvalidatrion 

    #2 3 4 e 5 elementos para favvorecer eses hiperpplanos de 2 en 2 basicamente 10, 6, 16  

    for arch in architectures
        println("\n Evaluando arquitectura: ", arch ) 
        acc_mean,
        _,
        rec_mean,
        spec_mean,
        prec_mean,
        npv_mean,
        f1_mean,
        _ = ANNCrossValidation(
            arch, (inputs, targets), cv_indices;
            numExecutions=5,
            maxEpochs=100,
            learningRate=0.01,
            validationRatio=0.1,
            maxEpochsVal=20
        )


        push!(resultados, (
        Arquitectura = string(arch),
        AccuracyMean = acc_mean,
        F1Mean = f1_mean,
        PrecisionMean = prec_mean,
        RecallMean = rec_mean,
        SpecificityMean = spec_mean,
        NPVMean = npv_mean
    ))


    end


    println("\nResultados crossvalidation de la RNA:")
    show(resultados, allcols=true)

    # Guardar en CSV
    CSV.write("P2/RESULTADOS/resultados_crossval_rna.csv", resultados)

    return resultados

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

end