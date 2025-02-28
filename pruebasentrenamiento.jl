using DelimitedFiles

    # Cargamos el dataset
    data = readdlm("iris.data",',');

    # Preparamos las entradas
    inputs = convert(Array{Float32,2}, data[:,1:4]);
    # Con cualquiera de estas 3 maneras podemos convertir la matriz de entradas de tipo Array{Any,2} en Array{Float32,2}, si los valores son numéricos:
    targets = oneHotEncoding(data[:,5]);

    # Cantidad total de patrones (filas)
    N = size(inputs, 1)
    
    #Separar los conjuntos de entrenamiento, validacion y test

    # Definir los porcentajes para validación y test
    Pval = 0.2  # 20% de validación
    Ptest = 0.2 # 20% de test

    # Obtener índices de cada conjunto
    train_idx, val_idx, test_idx = holdOut(N, Pval, Ptest)

    # Crear subconjuntos de datos
    train_inputs, train_targets = inputs[train_idx, :], targets[train_idx, :]
    val_inputs, val_targets = inputs[val_idx, :], targets[val_idx, :]
    test_inputs, test_targets = inputs[test_idx, :], targets[test_idx, :]

    #Calcular los valor de normalizacion de los datos de entrenamiento para normalizar todos los conjuntos

    

    #Proceso de entreno 
    trainingDataset = (train_inputs, train_targets)
    validationDataset = (val_inputs, val_targets)
    testDataset = (test_inputs, test_targets)

