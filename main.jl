# main.jl

# 1. Cargar módulos y paquetes
push!(LOAD_PATH, "src")  # Asegura que Julia pueda encontrar el módulo
using CSV, DataFrames
using Preprocesamiento  # Tu módulo con funciones de preprocesado

println("🚀 Iniciando el preprocesamiento...")

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

