using CSV, DataFrames

# Cargar el dataset original
df = CSV.read("P2/alzheimers_disease_data.csv", DataFrame)

# Lista de columnas seleccionadas por Lasso (sin duplicados y sin PatientID ni DoctorInCharge)
selected_features = [ "MMSE",
    "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL",
     "Diagnosis" # Target variable
]

# Filtrar el DataFrame
filtered_df = select(df, selected_features)

# Verifica las primeras filas
println(first(filtered_df, 5))

# (Opcional) Guardar el nuevo dataset limpio
CSV.write("P2/OTROS ARCHIVOS/alzheimers_limpio.csv", filtered_df)
