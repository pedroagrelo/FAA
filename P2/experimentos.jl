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

    # resultados = DataFrame(
    #     Arquitectura = String[],
    #     Accuracy_mean = Array{Float64,1}, Accuracy_std = Array{Float64,1},
    #     F1_mean = Array{Float64,1}, F1_std = Array{Float64,1},
    #     Precision_mean = Array{Float64,1}, Precision_std = Array{Float64,1},
    #     Recall_mean = Array{Float64,1}, Recall_std = Array{Float64,1}
    # )

#     resultados = DataFrame(
#     Arquitectura = String[],
#     Accuracy_mean = Float64[],
#     # Accuracy_std = Float64[],
#     F1_mean = Float64[],
#     # F1_std = Float64[],
#     Precision_mean = Float64[],
#     # Precision_std = Float64[],
#     Recall_mean = Float64[],
#     # Recall_std = Float64[]
# )

    # resultados = DataFrame(
    # Arquitectura = String[],
    # Accuracy = Vector{Vector{Float64}}(),
    # F1 = Vector{Vector{Float64}}(),
    # Precision = Vector{Vector{Float64}}(),
    # Recall = Vector{Vector{Float64}}()
    # )



    resultados = DataFrame(
    Arquitectura = String[],
    AccuracyMean = Vector{Vector{Float64}}(),
    AccuracyStd  = Vector{Vector{Float64}}(),
    F1Mean       = Vector{Vector{Float64}}(),
    F1Std        = Vector{Vector{Float64}}(),
    PrecisionMean = Vector{Vector{Float64}}(),
    PrecisionStd  = Vector{Vector{Float64}}(),
    RecallMean    = Vector{Vector{Float64}}(),
    RecallStd     = Vector{Vector{Float64}}()
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

    # for arch in architectures
    #     println("\n Evaluando arquitectura: ", arch)
    #     (acc_mean, acc_std), (err_mean, err_std), 
    #     (rec_mean, rec_std), (spec_mean, spec_std), 
    #     (prec_mean, prec_std), (npv_mean, npv_std), 
    #     (f1_mean, f1_std), conf_matrix = 
    #         ANNCrossValidation(arch, (X_norm, targets), cv_indices;
    #     numExecutions=5,
    #     maxEpochs=100,
    #     learningRate=0.01,
    #     validationRatio=0.1,
    #     maxEpochsVal=20)

        # push!(resultados, (
        # string(arch),
        # [acc_mean], [acc_std],  # Coloca los resultados en vectores
        # [f1_mean], [f1_std],
        # [prec_mean], [prec_std],
        # [rec_mean], [rec_std]
        # ))
    # end
    # for arch in architectures
    #     println("\n Evaluando arquitectura: ", arch )    
    #     acc, err, recall, spec, prec, npv, f1, conf_matrix = ANNCrossValidation(
    #     arch, (X_norm, targets), cv_indices;
    #         numExecutions=5,
    #         maxEpochs=100,
    #         learningRate=0.01,
    #         validationRatio=0.1,
    #         maxEpochsVal=20
    #     )

        # (acc_mean, acc_std),
        # (err_mean, err_std),
        # (rec_mean, rec_std),
        # (spec_mean, spec_std),
        # (prec_mean, prec_std),
        # (npv_mean, npv_std),
        # (f1_mean, f1_std),
        # conf_matrix = ANNCrossValidation(
        #     arch, (X_norm, targets), cv_indices;
        #         numExecutions=5,
        #         maxEpochs=100,
        #         learningRate=0.01,
        #         validationRatio=0.1,
        #         maxEpochsVal=20
        #     )

        for arch in architectures
        println("\n Evaluando arquitectura: ", arch ) 
        (acc_mean, acc_std),
        (_, _),  # Error rate si no lo usas
        (rec_mean, rec_std),
        (_, _),  # Specificity si no lo usas
        (prec_mean, prec_std),
        (_, _),  # NPV si no lo usas
        (f1_mean, f1_std),
        conf_matrix =   ANNCrossValidation(
            arch, (X_norm, targets), cv_indices;
                numExecutions=5,
                maxEpochs=100,
                learningRate=0.01,
                validationRatio=0.1,
                maxEpochsVal=20
            )
        
    

        # push!(resultados, (
        #     Arquitectura = string(arch),
        #     Accuracy     = acc,
        #     F1           = f1,
        #     Precision    = prec,
        #     Recall       = recall
        # ))

        push!(resultados, (
        Arquitectura = string(arch),
        AccuracyMean = acc_mean,
        AccuracyStd  = acc_std,
        F1Mean       = f1_mean,
        F1Std        = f1_std,
        PrecisionMean = prec_mean,
        PrecisionStd  = prec_std,
        RecallMean    = rec_mean,
        RecallStd     = rec_std
        ))


        
        # Agregar los resultados como una fila con los valores correctos

        # println(typeof(acc_mean))  # Debería ser Vector{Float64}
        # println(typeof(acc_std))   # Debería ser Vector{Float64}

        # push!(resultados, (
        #     string(arch),  # Arquitectura como string
        #     acc_mean, acc_std,  # Resultados como Float64
        #     f1_mean, f1_std,
        #     prec_mean, prec_std,
        #     rec_mean, rec_std
        # ))

        # push!(resultados, (
        # string(arch),          # Arquitectura como string
        # acc_mean,              # Vector de accuracy
        # f1_mean,               # Vector de F1
        # prec_mean,             # Vector de precisión
        # rec_mean               # Vector de recall
        # ))



        # # Acumula la matriz de confusión de esta arquitectura
        conf_matrix_final = conf_matrix  # Aquí tomamos la matriz de la última arquitectura evaluada
    end


    println("\nResultados resumen:")
    show(resultados, allcols=true)

    # Guardar si quieres
    CSV.write("resultados_crossval_rna.csv", resultados)

    return resultados, conf_matrix_final

end;

# Función para realizar el test ANOVA
# function realizar_anova(df::DataFrame)
    # Obtenemos las métricas de todas las arquitecturas
    # n_values = nrow(df)
    # println("Número de valores: $n_values")

    # Crear los grupos de accuracy para cada arquitectura (ajustar según el número de valores)
    # num_architectures = 6  # Suponiendo que tienes 6 arquitecturas
    # accuracy_groups = [
    #     [df.AccuracyMean[i] for i in ((j-1)*10+1):(j*10)] for j in 1:num_architectures
    # ]
    

#     accuracy_values = [df.AccuracyMean[i] for i in 1:nrow(df)] 

#     # Para ANOVA, tenemos que hacer una lista de listas (una por arquitectura)
#     accuracy_groups = [
#     [df.AccuracyMean[i] for i in 1:10],  # Resultados para Arquitectura 1
#     [df.AccuracyMean[i] for i in 11:20],  # Resultados para Arquitectura 2
#     [df.AccuracyMean[i] for i in 21:30],  # Resultados para Arquitectura 3
#     [df.AccuracyMean[i] for i in 31:40],  # Resultados para Arquitectura 4
#     [df.AccuracyMean[i] for i in 41:50],  # Resultados para Arquitectura 5
#     [df.AccuracyMean[i] for i in 51:60],  # Resultados para Arquitectura 6
#     [df.AccuracyMean[i] for i in 61:70],  # Resultados para Arquitectura 7
#     [df.AccuracyMean[i] for i in 71:80]   # Resultados para Arquitectura 8
# ]

    # Realizamos el ANOVA para la variable Accuracy
    # anova_result = OneWayANOVATest(accuracy_groups...)
    
    # return anova_result
# end

# """
# realizar_anova(df; metric_symbols=nothing, n_architectures=8, n_folds=10)

# Realiza pruebas One-Way ANOVA para cada métrica indicada en `metric_symbols`.

# # Argumentos
# - `df::DataFrame`: DataFrame con n_architectures * n_folds filas y columnas de métricas.
# - `metric_symbols::Vector{Symbol}`: nombres de columnas métricas a testear. Por defecto, todas las columnas numéricas.
# - `n_architectures::Int`: número de grupos (arquitecturas).
# - `n_folds::Int`: número de valores por grupo (folds).

# # Retorna
# - Un diccionario con clave = métrica,  valor = resultado de OneWayANOVATest.
# """
# function realizar_anova(
#     df::DataFrame;
#     metric_symbols::Union{Nothing,Vector{Symbol}}=nothing,
#     n_architectures::Int=8,
#     n_folds::Int=10
# )
#     # Selección de métricas
#     if metric_symbols === nothing
#         # todas las columnas numéricas
#         metric_symbols = Symbol[ c for c in names(df) if eltype(df[!, c]) <: Real ]
#     end
#     results = Dict{Symbol,Any}()
#     total_rows = n_architectures * n_folds
#     for metric in metric_symbols
#         vals = df[!, metric]
#         if length(vals) != total_rows
#             @warn "La métrica $metric no tiene el número esperado de filas ($total_rows). Se omite."
#             continue
#         end
#         # Formar grupos por arquitectura
#         groups = Vector{Vector{eltype(vals)}}(undef, n_architectures)
#         for i in 1:n_architectures
#             start_idx = (i-1)*n_folds + 1
#             end_idx   = i*n_folds
#             groups[i] = vals[start_idx:end_idx]
#         end
#         # Prueba One-Way ANOVA
#         test = OneWayANOVATest(groups...)
#         results[metric] = test
#     end
#     return results
# end


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



# function obtener_matriz_confusion_global(topology, X, y, cv_indices)
#     k = maximum(cv_indices)
#     num_classes = length(unique(y))
#     conf_total = zeros(Int, num_classes, num_classes)

#     for fold in 1:k
#         train_idx = findall(cv_indices .!= fold)
#         test_idx  = findall(cv_indices .== fold)

#         X_train = X[train_idx, :]
#         X_test  = X[test_idx, :]
#         y_train = y[train_idx]
#         y_test  = y[test_idx]

#         y_train_enc = oneHotEncoding(y_train)
#         y_test_enc = oneHotEncoding(y_test)

#         ann, _ = trainClassANN(
#             topology,
#             (X_train, y_train_enc),
#             maxEpochs = 100,
#             learningRate = 0.01,
#             minLoss = 0.0
#         )

#         y_pred = ann(Float32.(X_test'))'
#         conf = confusionMatrix(y_pred, y_test_enc)
#         conf_total .+= conf
#     end

#     return conf_total
# end


# Ejecutar
#ejecutarRNA()
