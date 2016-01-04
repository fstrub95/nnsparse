# nnsparse
Tools for sparse networks with the Torch Framework

To install the plugin:
```
luarocks install nnsparse
```

Feel free to contact me for furhter information.

You may also find additional information on my webpage http://www.lighting-torch.com/
A concrete network using sparse input for collaborative filtering can be found here: https://github.com/fstrub95/Autoencoders_cf


<a name="nn.Containers"></a>
## Sparse Tensor ##

Additional methods are added to Tensor to handle sparse inputs

  * [sparsify](#torch.Tensor.sparsify) : turn a dense matrix/vector into a sparse vector/matrix
  * [densify](#torch.Tensor.densify) : turn a sparse matrix/vector into a dense vector/matrix
  * [ssort](#torch.Tensor.ssort) : sort a sparse vector according its values
  * [ssortByIndex](#torch.Tensor.ssortByIndex) : sort the index of a sparseVector ;
  * [DynamicSparseTensor](#DynamicSparseTensor) : builder to efficiently create Tensor by preallocating memory (up to 100 time faster than classic methods)

New methods will be progressively added. To come, addSparse(), mulSparse().  

Warning : if the tensor type is modified after loading the nnsparse package. The previous method will be erasesd
```lua
require("nnsparse")
torch.setdefaulttensortype('torch.FloatTensor') 
x = torch.zeros(10)
x[4] = 1
x:sparsify() --error
```
```lua
torch.setdefaulttensortype('torch.FloatTensor') 
require("nnsparse")
x = torch.zeros(10)
x[4] = 1
x:sparsify() --ok
# Layers #
```
<a name="torch.Tensor.densify"></a>

### sparsify([elem]) ###

Turn a dense matrix/vector into a sparse vector/matrix. Sparsification can be done by defining one element `elem ` default(0). The sparse matix is returned as a table of sparse vector.
```lua
x = torch.zeros(6)
x[2] = 1
x[6] = 8
x[3] = 4

th> x:sparsify()
 2  1
 3  4
 6  8

x = torch.ones(6,6):apply(function(x) if torch.uniform() < 0.6 then return 0 else return x end end)
th> x:sparsify()
{
  1 : DoubleTensor - size: 1x2
  2 : DoubleTensor - size: 2x2
  3 : DoubleTensor - size: 2x2
  4 : DoubleTensor - size: 2x2
  5 : DoubleTensor - size: 1x2
  6 : DoubleTensor - size: 1x2
}

th> x:sparsify()[2]
 1  1
 6  1
```

<a name="torch.Tensor.densify"></a>
### densify([elem], [dim]) ###
Turn a sparse vector/matrix into a dense vector/matrix. The sparse element can be choosen. The final dimension can be provide to speed up the method. Otherwise, the method will find itself the final size.

```lua
th> x = torch.Tensor{{1,1},{3,4},{6,2}}
th> x:densify()
 1
 0
 4
 0
 0
 2

th> x:densify(0/0)
1.000000
nan
4.000000
nan
nan
2.000000


th> x:densify(0, 8)
 1
 0
 4
 0
 0
 2
 0
 0

th> y = { torch.Tensor{{1,1},{3,4},{6,2}}, torch.Tensor{{4,8}} }
{
  1 : DoubleTensor - size: 3x2
  2 : DoubleTensor - size: 1x2
}

th>  torch.Tensor.densify(y)
 1  0  4  0  0  2
 0  0  0  8  0  0
 
 torch.Tensor.densify(y, 0, torch.Tensor{2, 8})
 1  0  4  0  0  2  0  0
 0  0  0  8  0  0  0  0
```

<a name="torch.Tensor.ssort"></a>
### ssort([ascend], [inplace]) ###
Sort a sparse vector. By default, it is a descent sort, the sort can also be inplaced.

```lua
x = torch.Tensor{{1,1},{3,4},{6,2}}
th> x:ssort()
 1  1
 6  2
 3  4
 
 th> x:ssort(true)
 3  4
 6  2
 1  1
```

### ssortByIndex([ascend], [inplace]) ###
Sort a sparse vector by using its index. By default, it is a descent sort, the sort can also be inplaced.
This feature is very important while using SparseLinear, or SparseLinearBatch.

```lua
x = torch.Tensor{{3,1},{1,4},{6,2}}
th> x:ssortByIndex()
 1  4
 3  1
 6  2

th> x:ssortByIndex(true)
 6  2
 3  1
 1  4
```


<a name="DynamicSparseTensor"></a>
### DynamicSparseTensor.new(reserve, coefMult) ###

This helper builder build SparseTensors by reducing the number of memory reallocation. It is similar to std::vector (g++). A sparse tensor is created with for `reserve` (default = 10) elements. It is then filled by calling the method `append(torch.Tensor(2))`. Whenever a sparse vector is full, its size is increased by `coefMult` (default = 2). Finally, the `build()` method resizes the final vector and it returns it. 

```lua
   local dynTensor = DynamicSparseTensor.new()
   for i = 1, 256 do
      dynTensor:append(torch.Tensor{1,1})
   end 
   local finalTensor = dynTensor:build()
   dynTensor:reset()
```



## Layers ##
  * [SparseLinearBatch](#nnsparse.SparseLinearBatch) : enable minibatch on sparse vectors 
  * [Densify](#nnsparse.Densify) : densify a sparse inputs  
  * [Sparsifier](#nnsparse.Sparsifier) : sparsify a dense inputs 
  * [Batchifier](#nnsparse.Batchifier) : Create minibatch on the fly  
 
<a name="nnsparse.SparseLinearBatch"></a>
### nnsparse.SparseLinearBatch(inputSize, outputSize, [ignoreAccGrad]) ###
This layer enables to use minibatch for sparse inputs with no loss in speed. This feature is not available in sparseLinear. the GPU is support is under development. If the layer `nn.SparseLinearBatch` is the input layer, then, it is advisable to desactivate the AccGrad feature. It will greatly increase the speed of backpropagation.

```lua
x = torch.Tensor(10,100):uniform()
x:apply(function(x) if torch.uniform() < 0.6 then return 0 else return x end end)
x = x:sparsify()

local sparseLayer = nn.SparseLinearBatch(100, 20)
sparseLayer:forward(x)

sparseLayer:backward(x,someLoss)
```

<a name="nnsparse.Densify"></a>
### nnsparse.Densify(inputSize) ###
This layer turns a sparse inputs (with 0) into a dense inputs 

```lua
x = torch.Tensor(10,100):uniform()
x:apply(function(x) if torch.uniform() < 0.6 then return 0 else return x end end)
x = x:sparsify()

local denseNetwork = nn.Sequential()
denseNetwork:add(nnsparse.Densify(100))
denseNetwork:add(nnsparse.Linear(100, 20))
denseNetwork:add(nn.Tanh())

local sparseNetwork = nn.Sequential()
sparseNetwork:add(nn.SparseLinearBatch(100, 20))
sparseNetwork:add(nn.Tanh())

local w1, dw1 = denseNetwork:getParameters()
local w2, dw2 = sparseNetwork:getParameters()

w2:copy(w1:clone())
dw2:copy(dw1:clone())

local outDense  = denseNetwork:forward(x)
local outSparse = sparseNetwork:forward(x)

assert(outDense:sum() == outSparse:sum())

```

<a name="nnsparse.Sparsifier"></a>
### nnsparse.Sparsifier([offset]) ###
This layer turns a dense input into a sparse one. One may add an offset option to increase the sparse tensor index. 


<a name="nnsparse.Batchifier"></a>
### nnsparse.Batchifier(network, inputSize, [batchSize]) ###
This network automatically create mini-batches on the forward step. 


```lua
x = torch.Tensor(200,100):uniform()
x:apply(function(x) if torch.uniform() < 0.6 then return 0 else return x end end)
x = x:sparsify()

local denseNetwork = nn.Sequential()
denseNetwork:add(nnsparse.Densify(10))
denseNetwork:add(nnsparse.Linear(10, 100))
denseNetwork:add(nn.Tanh())

batchifier = nn.Batchifier(denseNetwork, 100, 20) 
output     = batchifier:forward(newtrain, 20)     -- there is no memory explosion! 

```

## Criterions ##
  * [SparseCriterion](#nnsparse.SparseCriterion) : encapsulate nn.Criterion to handle sparse inputs/targets 
  * [SDAECriterion](#nnsparse.SDAECriterion) : Compute a denoising loss for autoencoders 
  * [SDAESparseCriterion](#nnsparse.SDAESparseCriterion) : Compute a denoising loss for sparse autoencoders 

Sparse criterion deals with sparse target vectors. This is mainly used with autoencoders with sparse input.

WARNING, sparse loss are averaged over the number of KWNOWN Values. 
Example, if the output values has 20 elements and the sparse target vector has 5 eleemnts. The final averaged loss will be divided by 5.
```
sparseCriterion(x, t)  = 1/t:size(1) \sum Criterion(x, t)
```

If t is a sparse matrix with a total of n elements, the sum operation still operates over all the elements, and divides by n.

The division by n can be avoided if one sets the internal variable sizeAverage to false:
```
criterion.sizeAverage = false
```

<a name="nnsparse.SparseCriterion"></a>
### nnsparse.SparseCriterion(denseEstimate, sparseTarget) ###
This layer enables to encapsulate a loss from the nn package. 

```lua
output = torch.Tensor(10,100):uniform()

sparseTarget = torch.Tensor(10,100):uniform()
sparseTarget:apply(function(x) if torch.uniform() < 0.6 then return 0 else return x end end)
sparseTarget = sparseTarget:sparsify()

criterion = nnsparse.SparseCriterion(nn.MSECriterion())

criterion:forward(output, sparseTarget)
criterion:backward(output, sparseTarget)
```


<a name="nnsparse.SDAECriterion"></a>
### nnsparse.SDAECriterion(criterion, SDAEconf) ###
Stacked Denoising Autoencoder criterion is based on Pascal Vincent et al. paper: http://dl.acm.org/citation.cfm?id=1953039. It aims at teaching an autoencoder to denoise data. Tis enable to learn more easily low-dimension features.

There is three ways to corrupt the input:
 - Adding Gaussian noise
 - Replacing one of the input by some predefined extrema (Salt&Paper)
 - Hide some of the values (MaskNoise)

The loss is then computed as follow:
```
 sparseCriterion(x, t)  = alpha* \sum (i in corrupted) Criterion(x_i, t) + beta* \sum_(i in non-corupted) Criterion(x_i, t)
```
 Where alpha, beta are respectively two hyperparameters that either strengthen the denoising aspect or the reconstruction apsect of the loss. 

```lua
criterion = nnsparse.SDAECriterion(nn.MSECriterion(), 
{
   alpha = 1
   beta  = 0.5
   hideRatio = 0.2,
   noiseRatio = 0.1,
   noiseMean  = 0,
   noiseStd   = 0.2,
   flipRatio = 0.1,
   flipRange = {-1, 1},
   })
   
input = torch.Tensor(10, 100):uniform()
   
-- corrupt the input. 
noisyInput = criterion:prepareInput(input)

output = autoencoder:forward(sparseInput)
  
-- compute the loss. /!\ the target is the clean input!
loss  = criterion:forward (output, sparseInput)
dloss = criterion:backward(output, sparseInput)

```

When the nnsparse module is loaded. All the nn.criterion gets a `prepareInput` method. It is equivalent to the identity method. Thus, one may switch from a classic criterion to a SDAE criterion wihtout modifying his soure code.
 

<a name="nnsparse.SDAESparseCriterion"></a>
### nnsparse.SDAESparseCriterion(criterion, SDAEconf) ###
This method encapsulate the SDAE criterion and apply it to sparse inputs.  

```lua
local criterion = nnsparse.SDAESparseCriterion(nn.MSECriterion(), 
{
      hideRatio = 0.2,
      alpha = 0.8,
      beta =  0.5,
})

input = torch.Tensor(10, 100):uniform()
input:apply(function(x) if torch.uniform() < 0.6 then return 0 else return x end end)

-- create a sparse input
sparseInput = input:sparsify()

-- corrupt the sparse input
noisyInput = criterion:prepareInput(sparseInput)
  
--compute the autoencoder output
output = autoencoder:forward(sparseInput)
  
-- compute the loss/dloss
loss  = criterion:forward (output, sparseInput)
dloss = criterion:backward(output, sparseInput)
