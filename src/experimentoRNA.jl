module experimentoRNA

using CSV, DataFrames, Random, Statistics, StatsBase, HypothesisTests
include("P1_soluciones.jl")


export ejecutarRNA, realizar_anova

function ejecutarRNA()

    # Leer dataset limpio
    df = CSV.read("P2/OTROS_ARCHIVOS/alzheimers_limpio_balanced.csv", DataFrame)
    inputs = Matrix(select(df, Not(:Diagnosis)))
    targets = Vector(df.Diagnosis)



    # Generar índices de validación cruzada estratificada

    cv_indices = CSV.read("P2/OTROS_ARCHIVOS/indices_crossval2.csv", DataFrame).Fold  #Cargar indices de cv comunes

    # Creamos
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


    # Probar diferentes arquitecturas
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