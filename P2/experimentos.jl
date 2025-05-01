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

    resultados = DataFrame(Arquitectura=String[], Accuracy=Float64[], F1=Float64[], Error=Float64[], Precision=Float64[], Recall=Float64[])

    for arch in architectures
        println("\n Evaluando arquitectura: ", arch)
        (acc, _), (_, _), (rec, _), (_, _), (prec, _), (_, _), (f1, _), _ =
            ANNCrossValidation(arch, (X_norm, targets), cv_indices;
                numExecutions=5,
                maxEpochs=100,
                learningRate=0.01,
                validationRatio=0.1,
                maxEpochsVal=20)

        push!(resultados, (
            string(arch),
            round(acc, digits=4),
            round(f1, digits=4),
            round(1 - acc, digits=4),
            round(prec, digits=4),
            round(rec, digits=4)
        ))
    end

    println("\nResultados resumen:")
    show(resultados, allcols=true)

    # Guardar si quieres
    CSV.write("resultados_crossval_rna.csv", resultados)

    return resultados
end

# Ejecutar
ejecutarRNA()
