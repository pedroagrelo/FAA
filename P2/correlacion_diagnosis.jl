using CSV
using DataFrames
using Statistics
using Plots


# 1. Leer el archivo CSV
df = CSV.read("P2/OTROS ARCHIVOS/alzheimers_disease_data.csv", DataFrame)

# 2. Convertir Diagnosis a número si no lo es (opcional)
if !(eltype(df.Diagnosis) <: Number)
    df.Diagnosis = map(x -> x == "Demented" ? 1 : x == "Converted" ? 0.5 : 0, df.Diagnosis)
end

# 3. Seleccionar solo columnas numéricas (incluyendo Diagnosis)
numeric_df = select(df, names(df, eltype.(eachcol(df)) .<: Number))

# 4. Separar Diagnosis como vector
target = numeric_df[:, :Diagnosis]

# 5. Calcular correlación de Pearson entre Diagnosis y cada otra variable
cor_vals = [cor(target, numeric_df[:, col]) for col in names(numeric_df) if col != :Diagnosis]
colnames = [col for col in names(numeric_df) if col != :Diagnosis]

# 6. Graficar como heatmap 1D (barra vertical de correlaciones)
bar(
    reverse(cor_vals);
    orientation = :horizontal,
    yticks = (1:length(colnames), reverse(colnames)),
    title = "Correlación con Diagnosis (Pearson)",
    xlabel = "Correlación",
    color = :coolwarm,
    xlim = (-1, 1),
    legend = false,
    size = (800, 600)
)

# 7. Guardar figura
savefig("IMÁGENES/correlacion_con_diagnosis.png")
println("¡Hecho! Correlación con Diagnosis guardada como imagen.")
