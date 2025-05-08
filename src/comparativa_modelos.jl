using CSV, DataFrames, Plots

# Cargar los CSV individuales
df_knn = CSV.read("resultados_crossval_knn.csv", DataFrame)
df_dome = CSV.read("resultados_crossval_dome.csv", DataFrame)
df_rna = CSV.read("D:/CopiaPedro/CLASE/2º/2ºcuatri/Fundamentos de Aprendizaje Automático/Práctica2aParte/FAA/resumen_resultados_crossval_rna.csv", DataFrame)  # generado por plotsRNA
df_dt = CSV.read("D:/CopiaPedro/CLASE/2º/2ºcuatri/Fundamentos de Aprendizaje Automático/Práctica2aParte/FAA/resultados_crossval_dt.csv", DataFrame)    # generado por plotsDT

# Tomar la mejor fila de cada modelo (puedes cambiar criterio: por max Accuracy)
mejor_knn = df_knn[argmax(df_knn.AccuracyMean), :]
mejor_dome = df_dome[argmax(df_dome.AccuracyMean), :]
mejor_rna = df_rna[argmax(df_rna.Accuracy_mean), :]
mejor_dt = df_dt[argmax(df_dt.Accuracy_vals), :]

# Unificar en un DataFrame comparativo
comparacion = DataFrame(
    Modelo = ["KNN", "DoME", "RNA", "Árbol"],
    Accuracy = [mejor_knn.AccuracyMean, mejor_dome.AccuracyMean, mejor_rna.Accuracy_mean, mejor_dt.Accuracy_vals],
    Desviacion = [mejor_knn.AccuracyStd, mejor_dome.AccuracyStd, mejor_rna.Accuracy_std, mejor_dt.Accuracy_std]
)

# Graficar
bar(
    comparacion.Modelo,
    comparacion.Accuracy,
    yerror = comparacion.Desviacion,
    title = "Comparativa de Accuracy entre modelos",
    ylabel = "Accuracy",
    xlabel = "Modelo",
    legend = false,
    color = :skyblue,
    size = (800, 500)
)
savefig("comparativa_accuracy_modelos.png")
println("Gráfica comparativa guardada como 'comparativa_accuracy_modelos.png'")
