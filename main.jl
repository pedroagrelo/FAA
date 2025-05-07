# main.jl

# 1. Cargar módulos y paquetes
push!(LOAD_PATH, "src")  # Asegura que Julia pueda encontrar el módulo

# include("src/preprocesamiento.jl")
# include("src/experimento_RNA.jl")
# include("src/plots_RNA.jl")

using CSV, DataFrames
using Revise #eliminar antes de entregar proyecto, esto es solo para desarrollo
using preprocesamiento  # Tu módulo con funciones de preprocesado
using experimentoRNA #módulo con funcion de experimento y test de  hipótesis RNA
using plotsRNA #módulo con funciones de resumen y graficando

println("Iniciando el preprocesamiento...")

# 2. Paso 1: Calcular correlaciones (para selección de variables visual)
calcular_correlaciones("alzheimers_disease_data.csv", "correlacion_con_diagnosis.png")

# 3. Paso 2: Filtrar el dataset con columnas seleccionadas
filtrar_dataset(
    "alzheimers_disease_data.csv",
    "alzheimers_limpio.csv"
)

# 4. Paso 3: Aplicar balanceo por subsampling
balancear_dataset(
    "alzheimers_limpio.csv",
    "alzheimers_limpio_balanced.csv"
)

# 5. Paso 4: Revisar si hay outliers (mensaje informativo)
analizar_outliers("alzheimers_limpio_balanced.csv")

println("Preprocesamiento finalizado.")

# 6. Ejecutar experimento de RNA con validación cruzada
println("Ejecutando experimentos con RNA...")
resultados = ejecutarRNA()

# 7. Guardar resumen de métricas y graficar
println("Resumiendo y graficando métricas...")
resumen = resumir_metricas_RNA(resultados)
CSV.write("resumen_resultados_crossval_rna.csv", resumen)
graficar_metricas_RNA("resumen_resultados_crossval_rna.csv")

# 8. Realizar test ANOVA
println("\n Resultados del test ANOVA:")
anova_results = realizar_anova(resultados, :AccuracyMean)
println(anova_results)

println("\n Flujo experimento RNA finalizado.")

# ------------------------------------------------------
# Ejecutar experimento con Árboles de Decisión
# ------------------------------------------------------

# Incluir e importar los módulos (debes tener definidos los módulos con `module ... end`)
include("src/experimentoDT.jl")
include("src/plotsDT.jl")
using .experimentoDT
using .plotsDT

println("Ejecutando experimentos con Árboles de Decisión...")
experimentoDT.ejecutarDecisionTree()

println("Resumiendo y graficando métricas del Árbol de Decisión...")
resumen_dt = plotsDT.resumir_metricas_dt_detalle("resultados_crossval_dt.csv")
CSV.write("resumen_resultados_crossval_dt.csv", resumen_dt)
plotsDT.graficar_metricas_barras_dt("resumen_resultados_crossval_dt.csv")

println("\n Flujo experimento Árbol de Decisión finalizado.")


