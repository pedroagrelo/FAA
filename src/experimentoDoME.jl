module experimentoDoME
using CSV, DataFrames, Combinatorics
include("P1_soluciones.jl")
export realizar_test_tukey_dome
export ejecutar_dome
export realizar_test_anova_DoME

function ejecutar_dome()
    data = CSV.read("P2/OTROS_ARCHIVOS/alzheimers_limpio_balanced.csv", DataFrame)

    # 2. Extraer entradas (X) y salida (y), y normalizar las entradas 
    inputs = Matrix(data[:, Not("Diagnosis")])
    targets = data[:, "Diagnosis"]
    inputs = normalizeMinMax(inputs)

    # 4. Convertir las etiquetas a strings para DoME
    targets = string.(targets)

    
    # Cargar índices de validación cruzada
    indicesCV = CSV.read("P2/OTROS_ARCHIVOS/indices_crossval.csv", DataFrame).Fold

    # -------------------------------
    # 6. Valores de nodos a probar
    # -------------------------------
    nodos_a_probar = [5, 10, 15, 20, 25, 30, 35, 40]

    #Inicializar tabla de resultados resumen 
    resultados_dome = DataFrame(
        MaxNodes = Int[],
        AccuracyMean = Float64[], AccuracyStd = Float64[],
        F1Mean = Float64[], F1Std = Float64[],
        RecallMean = Float64[], RecallStd = Float64[],
        PrecisionMean = Float64[], PrecisionStd = Float64[],
        SpecificityMean = Float64[], SpecificityStd = Float64[],
        NPVMean = Float64[], NPVStd = Float64[]
    )

    #Resultados por fold
    resultados_fold = DataFrame(
        MaxNodes = Int[],
        Fold = Int[],
        Accuracy = Float64[],
        F1 = Float64[],
        Recall = Float64[],
        Precision = Float64[],
        Specificity = Float64[],
        NPV = Float64[]
    )

    # 7. Ejecutar experimentos
    println("=== EXPERIMENTO DoME - Alzheimer ===")
    for max_nodes in nodos_a_probar
        println("\n[DoME] Máximo nodos = ", max_nodes)
        
        acc,
        _, 
        recall,
        specificity,
        precision,
        npv,
        f1,
        _ = modelCrossValidation(:DoME,

            Dict("maximumNodes" => max_nodes),
            (inputs, targets),
            indicesCV)

        # Guardar resultados en DataFrame
        push!(resultados_dome, (
            max_nodes,
            mean(acc), std(acc),
            mean(f1), std(f1),
            mean(recall), std(recall),
            mean(precision), std(precision),
            mean(specificity), std(specificity),
            mean(npv), std(npv)
        ))
        # Guardar todos los folds (fila por fold)
        for i in 1:length(acc)
            push!(resultados_fold, (
                max_nodes,
                i,  # número de fold
                acc[i],
                f1[i],
                recall[i],
                precision[i],
                specificity[i],
                npv[i]
            ))
        end

        println("Resultados para maximumNodes = ", max_nodes)
        println("→ Accuracy     : ", round(mean(acc), digits=4), " ± ", round(std(acc), digits=4))
        println("→ F1 Score     : ", round(mean(f1), digits=4), " ± ", round(std(f1), digits=4))
        println("→ Recall       : ", round(mean(recall), digits=4), " ± ", round(std(recall), digits=4))
        println("→ Precision    : ", round(mean(precision), digits=4), " ± ", round(std(precision), digits=4))
        println("→ Specificity  : ", round(mean(specificity), digits=4), " ± ", round(std(specificity), digits=4))
        println("→ NPV          : ", round(mean(npv), digits=4), " ± ", round(std(npv), digits=4))

    end

    #Guardar CSVs
    CSV.write("P2/RESULTADOS/resultados_crossval_dome.csv", resultados_dome)
    CSV.write("P2/RESULTADOS/resultados_folds_dome.csv", resultados_fold)
    println("Archivos guardados: resumen y fold a fold.")
return resultados_dome
end

using HypothesisTests, Statistics

function realizar_test_anova_DoME()
    
    println("\nTest ANOVA sobre Accuracy (por MaxNodes):")

    # Agrupar los valores de accuracy por número de nodos
    # Vamos a leer el CSV recién guardado (por si se usa de forma modular)
    df = CSV.read("P2/RESULTADOS/resultados_folds_dome.csv", DataFrame)

    # Crear listas de grupos (una lista por cada valor de MaxNodes)
    grupos_accuracy = [df[df.MaxNodes .== n, :Accuracy] for n in unique(df.MaxNodes)]

    # Aplicar test ANOVA con splatting (...) para pasar los grupos como argumentos
    anova_test = OneWayANOVATest(grupos_accuracy...)
    p_valor = pvalue(anova_test)

    println("p-value: ", p_valor)
    println("Grados de libertad (entre grupos): ", anova_test.DFt)
    println("Grados de libertad (dentro de grupos): ", anova_test.DFe)

    if p_valor < 0.05
        println("Rechazamos la hipótesis nula: hay diferencias significativas entre las precisiones.")
    else
        println("No se rechaza la hipótesis nula: no hay diferencias significativas.")
    end
end

function realizar_test_tukey_dome()
    println("\nComparación múltiple tipo Tukey (DoME):")

    # 1. Leer archivo con datos por fold
    df = CSV.read("P2/RESULTADOS/resultados_folds_dome.csv", DataFrame)

    # 2. Agrupar por configuración (MaxNodes)
    grouped = combine(groupby(df, :MaxNodes),
        :Accuracy => mean => :mean,
        :Accuracy => std => :std,
        :Accuracy => length => :n
    )

    # 3. Función de diferencia significativa
    function diferencia_significativa(x, y, alpha = 0.05)
        d = abs(x.mean - y.mean)
        se = sqrt(x.std^2 / x.n + y.std^2 / y.n)
        t = 2.0  # Aprox. para 95% de confianza
        return d > t * se
    end

    # 4. Comparaciones entre todos los pares de configuraciones
    significativas = Set{Tuple{Int, Int}}()
    for (a, b) in combinations(eachrow(grouped), 2)
        if diferencia_significativa(a, b)
            println("MaxNodes = $(a.MaxNodes) vs $(b.MaxNodes): diferencia significativa")
            push!(significativas, (a.MaxNodes, b.MaxNodes))
            push!(significativas, (b.MaxNodes, a.MaxNodes))
        else
            println("MaxNodes = $(a.MaxNodes) vs $(b.MaxNodes): NO significativa")
        end
    end
        # 5. Identificar las mejores configuraciones
    # Ordenar por media descendente
    sorted = sort(grouped, :mean, rev = true)
    mejor = sorted[1, :MaxNodes]

    # Seleccionar aquellas que no tienen diferencias significativas con ninguna mejor
    mejores = [mejor]
    for i in 2:size(sorted, 1)
        nodo = sorted[i, :MaxNodes]
        if !((nodo, mejor) in significativas)
            push!(mejores, nodo)
        end
    end

    println("\nConfiguraciones recomendadas (sin diferencias significativas con las mejores):")
    println("→ MaxNodes = ", sort(mejores))
end
end
