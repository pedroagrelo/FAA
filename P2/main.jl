# main.jl

# 1. Cargar módulos y paquetes
push!(LOAD_PATH, "src")  # Asegura que Julia pueda encontrar el módulo

include("../src/preprocesamiento.jl")
include("../src/experimentoRNA.jl")
include("../src/plotsRNA.jl")

using CSV, DataFrames
using Revise #eliminar antes de entregar proyecto, esto es solo para desarrollo
using .preprocesamiento  # Tu módulo con funciones de preprocesado
using .experimentoRNA #módulo con funcion de experimento y test de  hipótesis RNA
using .plotsRNA #módulo con funciones de resumen y graficando

println("Iniciando el preprocesamiento...")

# 2. Paso 1: Calcular correlaciones (para selección de variables visual)
preprocesamiento.calcular_correlaciones("P2/OTROS_ARCHIVOS/alzheimers_disease_data.csv", "P2/IMÁGENES/correlacion_con_diagnosis.png")

# 3. Paso 2: Filtrar el dataset con columnas seleccionadas
preprocesamiento.filtrar_dataset(
    "P2/OTROS_ARCHIVOS/alzheimers_disease_data.csv",
    "P2/OTROS_ARCHIVOS/alzheimers_limpio.csv"
)

# 4. Paso 3: Aplicar balanceo por subsampling
preprocesamiento.balancear_dataset(
    "P2/OTROS_ARCHIVOS/alzheimers_limpio.csv",
    "P2/OTROS_ARCHIVOS/alzheimers_limpio_balanced.csv"
)

# 5. Paso 4: Revisar si hay outliers (mensaje informativo)
preprocesamiento.analizar_outliers("P2/OTROS_ARCHIVOS/alzheimers_limpio_balanced.csv")

println("Preprocesamiento finalizado.")



# 6. Ejecutar experimento de RNA con validación cruzada
println("Ejecutando experimentos con RNA...")
resultados = ejecutarRNA()

# 7. Guardar resumen de métricas y graficar
println("Resumiendo y graficando métricas...")
resumen = resumir_metricas_RNA(resultados)
CSV.write("P2/RESULTADOS/resumen_resultados_crossval_rna.csv", resumen)
print(resumen)
graficar_metricas_RNA("P2/RESULTADOS/resumen_resultados_crossval_rna.csv")

# 8. Realizar test ANOVA
println("\n Resultados del test ANOVA:")
anova_results = experimentoRNA.realizar_anova(resultados, :AccuracyMean)
println(anova_results)

println("\n Flujo experimento RNA finalizado.")

# ------------------------------------------------------
# Ejecutar experimento con Árboles de Decisión
# ------------------------------------------------------

# Incluir e importar los módulos (debes tener definidos los módulos con `module ... end`)
include("../src/experimentoDT.jl")
include("../src/plotsDT.jl")
using .experimentoDT
using .plotsDT: graficar_metricas_barras_dt, resumir_metricas_dt_detalle


println("Ejecutando experimentos con Árboles de Decisión...")
experimentoDT.ejecutarDecisionTree()

println("Resumiendo y graficando métricas del Árbol de Decisión...")
resumen_dt = resumir_metricas_dt_detalle("P2/RESULTADOS/resultados_crossval_dt.csv")
CSV.write("P2/RESULTADOS/resumen_resultados_crossval_dt.csv", resumen_dt)
graficar_metricas_barras_dt("P2/RESULTADOS/resumen_resultados_crossval_dt.csv")

println("\n Flujo experimento Árbol de Decisión finalizado.")


# ------------------------------------------------------
# Ejecutar experimento SVM
# ------------------------------------------------------
include("../src/experimentoSVM.jl")
include("../src/plotsSVM.jl")
using .experimentoSVM: ejecutarSVM
using .plotsSVM: graficar_metricas_svm, resumir_metricas_svm
println("Ejecutando experimentos con SVM...")

resultados = ejecutarSVM()  # <- genera un DataFrame con las métricas por fold
CSV.write("P2/RESULTADOS/resultados_svm_raw.csv", resultados)

# -------------------------------------
# ANOVA sobre Accuracy
# -------------------------------------
println("\nResultados del test ANOVA SVM:")
experimentoSVM.realizar_anova(resultados)
experimentoSVM.seleccionar_mejor_configuracion_svm(resultados)

# -------------------------------------
# Resumen y gráficas
# -------------------------------------
println("Resumiendo métricas por configuración...")
resumen = resumir_metricas_svm(resultados)
CSV.write("P2/RESULTADOS/resultados_resumen_svm.csv", resumen)

println("Generando gráficos de resultados...")
graficar_metricas_svm("P2/RESULTADOS/resultados_resumen_svm.csv")

# ------------------------------------------------------
# Ejecutar experimento DoME
# ------------------------------------------------------
include("../src/experimentoDoME.jl")
include("../src/plotsDoME.jl")
using .plotsDoME
using .experimentoDoME

println("Ejecutando experimento DoME y generando gráficas...")
resultados_dome = ejecutar_dome()
CSV.write("P2/RESULTADOS/resultados_crossval_dome.csv", resultados_dome)
plotsDoME.graficar_metricas_dome("resultados_crossval_dome.csv")

println("\nResultados del test ANOVA DoME:")
experimentoDoME.realizar_test_anova_DoME()

println("\nResultados del test de Tukey DoME:")
experimentoDoME.realizar_test_tukey_dome()


# ------------------------------------------------------
# Ejecutar experimento KNN
# ------------------------------------------------------
include("../src/experimentoKNN.jl")
include("../src/plotsKNN.jl")
using .plotsKNN

println("Ejecutando experimento KNN y generando gráficas...")
resultados_knn = experimentoKNN.ejecutar_knn()
CSV.write("P2/RESULTADOS/resultados_crossval_knn.csv", resultados_knn)
plotsKNN.graficar_metricas_knn("resultados_crossval_knn.csv")

println("\nResultados del test ANOVA KNN:")
experimentoKNN.realizar_test_anova_knn()

println("\n Ejecutando test de Tukey KNN...")   
experimentoKNN.realizar_test_tukey_knn()


# ------------------------------------------------------
# Comparación final de modelos (ANOVA y Tukey)
# ------------------------------------------------------
include("../src/comparacion_modelos.jl")


println("\nResultados de la comparación entre modelos:")
# 2. Ejecutar la comparación de los modelos (ANOVA, Tukey y mejor modelo)
comparacion_modelos.ejecutar_experimentos()