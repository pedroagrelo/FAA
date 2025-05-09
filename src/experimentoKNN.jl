module experimentoKNN
using CSV, DataFrames, Statistics, Combinatorics
include("P1_soluciones.jl")

export ejecutar_knn
export realizar_test_anova_knn
export realizar_test_tukey_knn

function ejecutar_knn()
        
    # 1. Cargar datos
    data = CSV.read("P2/OTROS_ARCHIVOS/alzheimers_limpio_balanced.csv", DataFrame)
    inputs = Matrix(data[:, Not("Diagnosis")])
    targets = string.(data[:, "Diagnosis"])
    inputs = normalizeMinMax(inputs)

    # 2. Cargar índices de validación cruzada comunes
    cv_indices = CSV.read("P2/OTROS_ARCHIVOS/indices_crossval.csv", DataFrame).Fold  #Cargar indices de cv comunes

    # 3. Valores de k a probar
    k_valores = [1, 3, 5, 7, 9, 11, 13, 15]

    # 4. Inicializar DataFrame de resultados
    resultados_knn = DataFrame(
        K = Int[],
        AccuracyMean = Float64[], AccuracyStd = Float64[],
        F1Mean = Float64[], F1Std = Float64[],
        RecallMean = Float64[], RecallStd = Float64[],
        PrecisionMean = Float64[], PrecisionStd = Float64[],
        SpecificityMean = Float64[], SpecificityStd = Float64[],
        NPVMean = Float64[], NPVStd = Float64[]
    )
    
    resultados_fold_knn = DataFrame(
        K = Int[],
        Fold = Int[],
        Accuracy = Float64[],
        F1 = Float64[],
        Recall = Float64[],
        Precision = Float64[],
        Specificity = Float64[],
        NPV = Float64[]
    )
    
    # 5. Ejecutar experimentos
    for k in k_valores
        println("\n[KNN] Vecinos = $k")

        acc, _, recall, specificity, precision, npv, f1, _ =
            modelCrossValidation(
                :KNeighborsClassifier,
                Dict("n_neighbors" => k),
                (inputs, targets),
                cv_indices
            )

        # Guardar en DataFrame
        push!(resultados_knn, (
            k,
            mean(acc), std(acc),
            mean(f1), std(f1),
            mean(recall), std(recall),
            mean(precision), std(precision),
            mean(specificity), std(specificity),
            mean(npv), std(npv)
        ))

        for (i, _) in enumerate(acc)
            push!(resultados_fold_knn, (
                k, i,
                acc[i], f1[i], recall[i],
                precision[i], specificity[i], npv[i]
            ))
        end
        

        # Mostrar por pantalla
        println("→ Accuracy     : ", round(mean(acc), digits=4), " ± ", round(std(acc), digits=4))
        println("→ F1 Score     : ", round(mean(f1), digits=4), " ± ", round(std(f1), digits=4))
        println("→ Recall       : ", round(mean(recall), digits=4), " ± ", round(std(recall), digits=4))
        println("→ Precision    : ", round(mean(precision), digits=4), " ± ", round(std(precision), digits=4))
        println("→ Specificity  : ", round(mean(specificity), digits=4), " ± ", round(std(specificity), digits=4))
        println("→ NPV          : ", round(mean(npv), digits=4), " ± ", round(std(npv), digits=4))
    end

    # 6. Guardar resultados en CSV
    CSV.write("resultados_crossval_knn.csv", resultados_knn)
    println("Resultados guardados en 'resultados_crossval_knn.csv'")
    CSV.write("resultados_folds_knn.csv", resultados_fold_knn)
    println("También guardado 'resultados_folds_knn.csv' con métricas por fold.")
return resultados_knn
end


using HypothesisTests, Statistics
function realizar_test_anova_knn()
    println("\nTest ANOVA sobre Accuracy:")

    # Agrupar los valores de accuracy por número de nodos
    # Vamos a leer el CSV recién guardado (por si se usa de forma modular)
    df = CSV.read("resultados_folds_knn.csv", DataFrame)

    # Agrupar accuracies por valor de K
    grouped = groupby(df, :K)
    grupos_accuracy = [group.Accuracy for group in grouped]

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

function realizar_test_tukey_knn()
    println("\nComparación múltiple tipo Tukey (KNN):")

    # 1. Leer resultados por fold
    df = CSV.read("resultados_folds_knn.csv", DataFrame)

    # 2. Agrupar por K
    grouped = combine(groupby(df, :K),
        :Accuracy => mean => :mean,
        :Accuracy => std => :std,
        :Accuracy => length => :n
    )

    # 3. Función para diferencia significativa
    function diferencia_significativa(x, y, alpha = 0.05)
        d = abs(x.mean - y.mean)
        se = sqrt(x.std^2 / x.n + y.std^2 / y.n)
        t = 2.0  # Aprox. 95% confianza
        return d > t * se
    end

    # 4. Comparar todos los pares de configuraciones
    for (a, b) in combinations(eachrow(grouped), 2)
        if diferencia_significativa(a, b)
            println("K = $(a.K) vs $(b.K): diferencia significativa")
        else
            println("K = $(a.K) vs $(b.K): NO significativa")
        end
    end
end
end #module