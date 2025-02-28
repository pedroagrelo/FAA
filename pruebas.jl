include("firmas.jl");

# Datos de ejemplo
X = rand(10, 100)  # 10 características, 100 patrones
Y = rand(Bool, 3, 100)  # 100 salidas binarias (0 o 1)

# Convertir los datos a un formato adecuado
dataset = (X, Y)

# Definir la topología (2 capas ocultas con 5 y 3 neuronas)
topology = [5, 3]

# Entrenar la red
rna, losses = trainClassANN(topology, dataset)  
# Imprimir las pérdidas de cada época
println(losses)

#println(size(Y), " ", typeof(Y))  
#println("dataset: ", dataset)  # Ver qué tiene `dataset`
#println(typeof(dataset))  # Ver qué tiene `dataset`