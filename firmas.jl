
# Tened en cuenta que en este archivo todas las funciones tienen puesta la palabra reservada 'function' y 'end' al final
# Según cómo las defináis, podrían tener que llevarlas o no

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Statistics
using Flux
using Flux.Losses

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    num_classes = length(classes)
    num_samples = length(feature)
    
    if num_classes <= 2
        # Caso binario: Generar un vector booleano y convertirlo en una matriz columna
        return reshape(feature .== classes[1], num_samples, 1) #NUM SAMPLE FILAS 1 columna
    else
        # Caso multiclase: Crear una matriz de valores booleanos one-hot# Comparar con cada clase y asignar
        return convert(BitArray{2}, hcat([feature.== cl for cl in classes]...))   
    end
end


# Sobrecarga que extrae automáticamente las categorías y llama a la función principal
oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature))

# Sobrecarga para el caso booleano
oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, length(feature), 1)

    



# Función para calcular los parámetros de normalización MinMax
function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real, 2})
  
    
    min_vals = minimum(dataset, dims=1)
    max_vals = maximum(dataset, dims=1)


    min_vals = reshape(min_vals, 1, length(min_vals))
    max_vals = reshape(max_vals, 1, length(max_vals))
  
    # Devolvemos como una tupla
    return (min_vals, max_vals)
end



# Función para calcular los parámetros de normalización Zero-Mean
function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real, 2})
    if isempty(dataset)
        error("El dataset está vacío.")
    end
    if any(isnan, dataset) || any(isinf, dataset)
        error("El dataset contiene valores NaN o Inf.")
    end

    means = mean(dataset, dims=1)
    std_devs = std(dataset, dims=1)

    # Evita divisiones por 0 reemplazando std_devs == 0 con 1
    std_devs = std_devs .+ (std_devs .== 0)

    means = reshape(means, 1, length(means))
    std_devs = reshape(std_devs, 1, length(std_devs))

    return (means, std_devs)
end


# Función para normalizar entre Min-Max, modificando la matriz original
function normalizeMinMax!(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
    min_vals, max_vals = normalizationParameters
    min_vals=Matrix(min_vals)
    max_vals=Matrix(max_vals)
    
    # Evitamos la división por cero en el caso de que el valor máximo sea igual al valor mínimo
    dataset .-= min_vals
    dataset .*= 1 ./ (max_vals .- min_vals) .^ (max_vals .!= min_vals)

end

# Función para normalizar entre Min-Max, calculando los parámetros antes de normalizar
function normalizeMinMax!(dataset::AbstractArray{<:Real, 2})
    # Calculamos los parámetros de normalización
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)

    # Normalizamos la matriz
    normalizeMinMax!(dataset, normalizationParameters)
end

# Función para normalizar entre Min-Max sin modificar la matriz original, usando parámetros previos
function normalizeMinMax(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
    
    # Hacemos una copia de la matriz para no modificar la original
    dataset_copy = copy(dataset)
    # Normalizamos la copia de la matriz
    normalizeMinMax!(dataset_copy, normalizationParameters)
    return dataset_copy
end

# Función para normalizar entre Min-Max sin modificar la matriz original, calculando parámetros antes de normalizar
function normalizeMinMax(dataset::AbstractArray{<:Real, 2})
    # Calculamos los parámetros de normalización
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    # Normalizamos y devolvemos la nueva matriz
    return normalizeMinMax(dataset, normalizationParameters)
end



function normalizeZeroMean!(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
    means, std_devs = normalizationParameters
    means=Matrix(means)
    std_devs=Matrix(std_devs)
    dataset .-= means
    dataset .*= 1 ./ std_devs
end


function normalizeZeroMean!(dataset::AbstractArray{<:Real, 2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean!(dataset, normalizationParameters)
end



function normalizeZeroMean(dataset::AbstractArray{<:Real, 2}, normalizationParameters::NTuple{2, AbstractArray{<:Real, 2}})
    dataset_copy = copy(dataset)
    normalizeZeroMean!(dataset_copy, normalizationParameters)
    return dataset_copy
end


function normalizeZeroMean(dataset::AbstractArray{<:Real, 2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean(dataset, normalizationParameters)
end

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    return outputs.>=threshold
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs,2) == 1 # dimensiones de la matriz solo una columna
        return reshape(classifyOutputs(outputs[:], threshold=threshold),num_rows,1 )
    else 
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2);
        classified = falses(size(outputs));
        classified[indicesMaxEachInstance] .= true
        return classified
    end
end;

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    return mean((outputs .== targets)) 
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    if size(targets, 2) == 1 # columnas = 2?  || size(outputs, 2) == 2
        return accuracy(vec(outputs), vec(targets')) #array multidimensional a columna 
    else
        classComparison = targets' .== outputs
        correctClassifications = all(classComparison, dims=2)
        accuracy = mean(correctClassifications) 
        return accuracy 
    end
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
        return accuracy(outputs, targets' .>= threshold)
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    if size(targets', 2) == 1 
        return accuracy(vec(outputs), vec(targets'))
    else
        classifiedOutputs = classifyOutputs(outputs)
        return accuracy(targets', classifiedOutputs)
    end
end;

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1}, (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    #
    # Codigo a desarrollar
    #
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random

function holdOut(N::Int, P::Real)
    @assert 0 ≤ P ≤ 1 "P debe estar entre 0 y 1"
    indices = randperm(N)  # Permutación aleatoria de los índices
    n_test = round(Int, P * N)  # Cantidad de patrones para test
    train_idx, test_idx = indices[1:end-n_test], indices[end-n_test+1:end]  # División en dos subconjuntos
    return train_idx, test_idx
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    @assert Pval + Ptest <= 1 "La suma de Pval y Ptest debe ser menor o igual a 1"
    #Conjunto de test
    train_val_idx, test_idx = holdOut(N, Ptest)

    # Calcular la nueva tasa de validación relativa al conjunto de entrenamiento+validación
    new_Pval = Pval / (1 - Ptest)

    # Segunda separación: Conjunto de validación
    train_idx, val_idx = holdOut(length(train_val_idx), new_Pval)

    # Ajustar índices para que coincidan con el conjunto original
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    return train_idx, val_idx, test_idx

end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0,size(trainingDataset[2],2))),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::  Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    testDataset::      Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=(Array{eltype(trainingDataset[1]),2}(undef,0,size(trainingDataset[1],2)), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5, weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

using SymDoME


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;

function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;


function trainClassDoME(trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, testInputs::AbstractArray{<:Real,2}, maximumNodes::Int)
    #
    # Codigo a desarrollar
    #
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using Random
using Random:seed!

function crossvalidation(N::Int64, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, validationRatio::Real=0, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end;


# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using MLJ
using LIBSVM, MLJLIBSVMInterface
using NearestNeighborModels, MLJDecisionTreeInterface

SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
kNNClassifier = MLJ.@load KNNClassifier pkg=NearestNeighborModels verbosity=0
DTClassifier  = MLJ.@load DecisionTreeClassifier pkg=DecisionTree verbosity=0


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}, crossValidationIndices::Array{Int64,1})
    #
    # Codigo a desarrollar
    #
end
