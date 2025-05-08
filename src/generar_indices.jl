module indicesCrossval

using CSV, DataFrames
include("P1_soluciones.jl")  # para usar crossvalidation

export generar_indices

function generar_indices()
    # 1. Cargar datos
    data = CSV.read("P2/OTROS_ARCHIVOS/alzheimers_limpio_balanced.csv", DataFrame)
    targets = data.Diagnosis

    # 2. Generar índices estratificados
    k = 10
    indicesCV = crossvalidation(targets, k)

    # 3. Guardar en archivo CSV
    df_indices = DataFrame(Fold = indicesCV)
    CSV.write("P2/OTROS_ARCHIVOS/indices_crossval.csv", df_indices)

    println("Índices guardados en 'P2/OTROS_ARCHIVOS/indices_crossval.csv'")
end

end
