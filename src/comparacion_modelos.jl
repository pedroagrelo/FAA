module comparacion_modelos
using HypothesisTests
using MultivariateStats, Distributions, DataFrames

# Función para realizar ANOVA
function realizar_anova(acc_rna, acc_dt, acc_svm, acc_dome, acc_knn)
    anova_test = OneWayANOVATest(acc_rna, acc_dt, acc_svm, acc_dome, acc_knn)

    p_value = pvalue(anova_test)

    println("=== ANOVA ===")
    # Estadístico F
    println("p-valor: ", p_value)

    # Obtener los grados de libertad
    degrees_of_freedom_between = anova_test.DFt
    degrees_of_freedom_within = anova_test.DFe
    println("Grados de libertad entre grupos: ", degrees_of_freedom_between)
    println("Grados de libertad dentro de los grupos: ", degrees_of_freedom_within)

    # Decisión basada en el valor p
    if p_value < 0.05
        println("→ Rechazamos la hipótesis nula: Hay una diferencia significativa entre los modelos.")
    else
        println("→ No rechazamos la hipótesis nula: No hay una diferencia significativa entre los modelos.")
    end
end

# Función de comparaciones múltiples tipo Tukey HSD
function realizar_tukey_hsd(df::DataFrame, colname::Symbol = :Valor)
    println("\nComparaciones múltiples tipo Tukey HSD:")

    expanded = df
    grupos = unique(expanded.Profundidad)
    α = 0.05

    for i = 1:length(grupos)-1
        for j = i+1:length(grupos)
            g1, g2 = grupos[i], grupos[j]
            vals1 = expanded[expanded.Profundidad .== g1, :Valor]
            vals2 = expanded[expanded.Profundidad .== g2, :Valor]

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

# Función para encontrar el mejor modelo basado en el promedio de precisión
function mejor_modelo(df::DataFrame, colname::Symbol = :Valor)
    # Agrupar los datos por 'Profundidad' (modelo) y calcular la media de precisión para cada grupo
    grouped = groupby(df, :Profundidad)
    medias = combine(grouped, :Valor => mean)

    # Encontrar el modelo con la mejor precisión (media más alta)
    mejor_modelo = argmax(medias[!, :Valor_mean])  # Obtiene el índice del modelo con la media más alta
    modelo_mejor = medias[mejor_modelo, :Profundidad]
    precision_mejor = medias[mejor_modelo, :Valor_mean]
    
    println("\nEl mejor modelo es: $modelo_mejor con una precisión promedio de: $precision_mejor")
end

# Función principal que ejecuta todo el flujo
function ejecutar_experimentos()
    # Datos de precisión de los modelos
    acc_rna = [0.5355263157894737, 0.5381578947368422, 0.5328947368421053, 0.5355263157894737, 0.5460526315789473, 0.5236842105263159, 0.5578947368421053, 0.5236842105263159, 0.5315789473684209, 0.5315789473684212]
    acc_dt = [0.9276315789473685, 0.9407894736842105, 0.9276315789473685, 0.9539473684210527, 0.9605263157894737, 0.9342105263157895, 0.9210526315789473, 0.881578947368421, 0.9144736842105263, 0.9276315789473685]
    acc_svm = [0.8486842105263158, 0.8552631578947368, 0.8289473684210527, 0.8157894736842105, 0.868421052631579, 0.8618421052631579, 0.8355263157894737, 0.8289473684210527, 0.8223684210526315, 0.868421052631579]
    acc_dome = [0.8223684210526315, 0.8289473684210527, 0.7960526315789473, 0.8355263157894737, 0.8881578947368421, 0.8421052631578947, 0.8223684210526315, 0.7894736842105263, 0.8289473684210527, 0.8355263157894737]
    acc_knn = [0.9144736842105263, 0.9144736842105263, 0.907894736842105, 0.9276315789473685, 0.9144736842105263, 0.9144736842105263, 0.8947368421052632, 0.8552631578947368, 0.868421052631579, 0.9276315789473685]

    # Realizar ANOVA
    realizar_anova(acc_rna, acc_dt, acc_svm, acc_dome, acc_knn)

    # Crear un DataFrame con los resultados
    models = [:rna, :dt, :svm, :dome, :knn]
    data = hcat(acc_rna, acc_dt, acc_svm, acc_dome, acc_knn)
    df = DataFrame(Profundidad = repeat(models, inner=length(acc_rna)), Valor = vec(data))

    # Llamada a la función para determinar el mejor modelo
    mejor_modelo(df)

    # Llamada a la función para realizar Tukey HSD
    realizar_tukey_hsd(df)
end

end