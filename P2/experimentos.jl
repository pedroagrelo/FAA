using CSV, DataFrames, Random, Statistics
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
        Accuracy_mean = Float64[], Accuracy_std = Float64[],
        F1_mean = Float64[], F1_std = Float64[],
        Precision_mean = Float64[], Precision_std = Float64[],
        Recall_mean = Float64[], Recall_std = Float64[]
    )

    # 4. Probar diferentes arquitecturas
    architectures = [
        [5],
        [8],
        [12],
        [15],
        [8, 5],
        [12, 8],
        [15,12],
        [18, 15]
    ]

    conf_matrix_final = []

    for arch in architectures
        println("\n Evaluando arquitectura: ", arch)
        (acc_mean, acc_std), (err_mean, err_std), 
        (rec_mean, rec_std), (spec_mean, spec_std), 
        (prec_mean, prec_std), (npv_mean, npv_std), 
        (f1_mean, f1_std), conf_matrix = 
            ANNCrossValidation(arch, (X_norm, targets), cv_indices;
        numExecutions=5,
        maxEpochs=100,
        learningRate=0.01,
        validationRatio=0.1,
        maxEpochsVal=20)

        push!(resultados, (
        string(arch),
        round(acc_mean, digits=4), round(acc_std, digits=4),
        round(f1_mean, digits=4), round(f1_std, digits=4),
        round(prec_mean, digits=4), round(prec_std, digits=4),
        round(rec_mean, digits=4), round(rec_std, digits=4)
        ))
         
        # Guardar la matriz de confusión de la última arquitectura
        conf_matrix_final = conf_matrix  # Aquí tomamos la matriz de la última arquitectura evaluada
    end


println("\nResultados resumen:")
    show(resultados, allcols=true)

    # Guardar si quieres
    CSV.write("resultados_crossval_rna.csv", resultados)

    return resultados, conf_matrix_final
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
