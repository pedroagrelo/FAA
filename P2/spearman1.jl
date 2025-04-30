using CSV
using StatsBase
using DataFrames

# Leer el CSV en un DataFrame
df = CSV.File("P2/alzheimers_disease_data.csv") |> DataFrame

# Calcular correlación de Spearman entre 'Diagnosis' y las demás columnas
for col in names(df)
    if col != :Diagnosis && eltype(df[!, col]) <: Number
        ρ = corspearman(df[!, :Diagnosis], df[!, col])
        println("$col: ρ = $ρ")
    end
end
