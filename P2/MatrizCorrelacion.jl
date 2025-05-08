using CSV
using DataFrames
using Statistics
using Plots
using StatsPlots  # Para heatmap con nombres

# 1. Leer el archivo CSV
df = CSV.read("P2/OTROS ARCHIVOS/alzheimers_disease_data.csv", DataFrame)

# 2. Seleccionar solo columnas numéricas
numeric_df = select(df, names(df, eltype.(eachcol(df)) .<: Number))

# 3. Seleccionar solo columnas numéricas
numeric_df = select(df, names(df, eltype.(eachcol(df)) .<: Number))

# 4. Calcular la matriz de correlación Pearson
cor_matrix = cor(Matrix(numeric_df))

# 5. Obtener nombres de columnas
colnames = names(numeric_df)

# 6. Dibujar heatmap
heatmap(
    cor_matrix;
    xticks = (1:length(colnames), colnames),
    yticks = (1:length(colnames), colnames),
    title = "Matriz de correlación completa",
    xlabel = "", ylabel = "",
    color = :coolwarm,
    clim = (-1, 1),
    size = (800, 800),
    bottom_margin = 10Plots.mm,  # margen en mm con Plots.jl
    left_margin   = 10Plots.mm
)

# 7. Guardar la figura
savefig("P2/IMÁGENES/correlacion_alzheimer.png")

println("¡Hecho! Heatmap guardado en 'correlacion_alzheimer.png'.")