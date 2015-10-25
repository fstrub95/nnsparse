# nnsparse
Tools for sparse networks with the Torch Framework

To install the plugin:
```
luarocks install nnsparse
```

<a name="nn.Containers"></a>
## Sparse Tensor ##

Additional methods are added to Tensor to handle sparse inputs

  * [sparsify](#torch.Tensor.sparsify) : turn a dense matrix/vector into a sparse vector/matrix
  * [densify](#torch.Tensor.densify) : turn a sparse matrix/vector into a dense vector/matrix
  * [ssort](#torch.Tensor.ssort) : sort a sparse vector according its values
  * [ssortByIndex](#torch.Tensor.ssortByIndex) : sort the index of a sparseVector ;

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

## ssortByIndex([ascend], [inplace]) ##
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




## Layers ##
  * [SparseLinearBatch](#nn.SparseLinearBatch) : enable minibatch on sparse vectors 
 
<a name="nn.SparseLinearBatch"></a>
### nn.SparseLinearBatch(inputSize, outputSize, [ignoreAccGrad]) ###
This layer enables to use minibatch for sparse inputs with no loss in speed. This feature is not available in sparseLinear. the GPU is support is under development. If the layer `nn.SparseLinearBatch` is the input layer, then, it is advisable to desactivate the AccGrad feature. It will greatly increase the speed of backpropagation.

```lua
x = torch.Tensor(10,100):uniform()
x:apply(function(x) if torch.uniform() < 0.6 then return 0 else return x end end)
x = x:sparsify()

local sparseLayer = nn.SparseLinearBatch(10, 100)
sparseLayer:forward(x)

sparseLayer:backward(x,someLoss)

```


## Criterions ##
  * [SparseCriterion](#nn.SparseCriterion) : encapsulate nn.Criterion to handle sparse inputs/targets 
  * [SDAECriterion](#nn.SDAECriterion) : Compute a denoising loss for autoencoders 
  * [SDAESparseCriterion](#nn.SDAESparseCriterion) : Compute a denoising loss for sparse autoencoders 

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

<a name="nn.SparseCriterion"></a>
### nn.SparseCriterion(denseEstimate, sparseTarget) ###
This layer enables to encapsulate a loss from the nn package. 

```lua
output = torch.Tensor(10,100):uniform()

sparseTarget = torch.Tensor(10,100):uniform()
sparseTarget:apply(function(x) if torch.uniform() < 0.6 then return 0 else return x end end)
sparseTarget = sparseTarget:sparsify()

criterion = nn.SparseCriterion(nn.MSECriterion())

criterion:forward(output, sparseTarget)
criterion:backward(output, sparseTarget)
```


<a name="nn.SDAECriterion"></a>
### SDAECriterion(criterion, SDAEconf) ###
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
criterion = nn.SDAECriterion(nn.MSECriterion(), 
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
   
-- the input is corrupted. The corruption mask is stored inside the loss
noisyInput = criterion:prepareInput(input)

loss  = criterion:forward (output, noisyInput)
dloss = criterion:backward(output, noisyInput)
   
```

When the nnsparse module is loaded. All the nn.criterion gets a `prepareInput` method. It is equivalent to the identity method. Thus, one may switch from a classic criterion to a SDAE criterion wihtout modifying his soure code.
 





<a name="nn.Container.size"></a>
### size() ###
Returns the number of contained modules.

<a name="nn.Sequential"></a>
## Sequential ##

Sequential provides a means to plug layers together
in a feed-forward fully connected manner.

E.g. 
creating a one hidden-layer multi-layer perceptron is thus just as easy as:
```lua
mlp = nn.Sequential()
mlp:add( nn.Linear(10, 25) ) -- 10 input, 25 hidden units
mlp:add( nn.Tanh() ) -- some hyperbolic tangent transfer function
mlp:add( nn.Linear(25, 1) ) -- 1 output

print(mlp:forward(torch.randn(10)))
```
which gives the output:
```lua
-0.1815
[torch.Tensor of dimension 1]
```

<a name="nn.Sequential.remove"></a>
### remove([index]) ###

Remove the module at the given `index`. If `index` is not specified, remove the last layer.

```lua
model = nn.Sequential()
model:add(nn.Linear(10, 20))
model:add(nn.Linear(20, 20))
model:add(nn.Linear(20, 30))
model:remove(2)
> model
nn.Sequential {
  [input -> (1) -> (2) -> output]
  (1): nn.Linear(10 -> 20)
  (2): nn.Linear(20 -> 30)
}
```


<a name="nn.Sequential.insert"></a>
### insert(module, [index]) ###

Inserts the given `module` at the given `index`. If `index` is not specified, the incremented length of the sequence is used and so this is equivalent to use `add(module)`.

```lua
model = nn.Sequential()
model:add(nn.Linear(10, 20))
model:add(nn.Linear(20, 30))
model:insert(nn.Linear(20, 20), 2)
> model
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> output]
  (1): nn.Linear(10 -> 20)
  (2): nn.Linear(20 -> 20)      -- The inserted layer
  (3): nn.Linear(20 -> 30)
}
```



<a name="nn.Parallel"></a>
## Parallel ##

`module` = `Parallel(inputDimension,outputDimension)`

Creates a container module that applies its `ith` child module to the  `ith` slice of the input Tensor by using [select](https://github.com/torch/torch7/blob/master/doc/tensor.md#tensor-selectdim-index) 
on dimension `inputDimension`. It concatenates the results of its contained modules together along dimension `outputDimension`.

Example:
```lua
 mlp=nn.Parallel(2,1);     -- iterate over dimension 2 of input
 mlp:add(nn.Linear(10,3)); -- apply to first slice
 mlp:add(nn.Linear(10,2))  -- apply to first second slice
 print(mlp:forward(torch.randn(10,2)))
```
gives the output:
```lua
-0.5300
-1.1015
 0.7764
 0.2819
-0.6026
[torch.Tensor of dimension 5]
```

A more complicated example:
```lua

mlp=nn.Sequential();
c=nn.Parallel(1,2)
for i=1,10 do
 local t=nn.Sequential()
 t:add(nn.Linear(3,2))
 t:add(nn.Reshape(2,1))
 c:add(t)
end
mlp:add(c)

pred=mlp:forward(torch.randn(10,3))
print(pred)

for i=1,10000 do     -- Train for a few iterations
 x=torch.randn(10,3);
 y=torch.ones(2,10);
 pred=mlp:forward(x)

 criterion= nn.MSECriterion()
 local err=criterion:forward(pred,y)
 local gradCriterion = criterion:backward(pred,y);
 mlp:zeroGradParameters();
 mlp:backward(x, gradCriterion); 
 mlp:updateParameters(0.01);
 print(err)
end
```


<a name="nn.Concat"></a>
## Concat ##

```lua
module = nn.Concat(dim)
```
Concat concatenates the output of one layer of "parallel" modules along the
provided dimension `dim`: they take the same inputs, and their output is
concatenated.
```lua
mlp=nn.Concat(1);
mlp:add(nn.Linear(5,3))
mlp:add(nn.Linear(5,7))
print(mlp:forward(torch.randn(5)))
```
which gives the output:
```lua
 0.7486
 0.1349
 0.7924
-0.0371
-0.4794
 0.3044
-0.0835
-0.7928
 0.7856
-0.1815
[torch.Tensor of dimension 10]
```

<a name="nn.DepthConcat"></a>
## DepthConcat ##

```lua
module = nn.DepthConcat(dim)
```
DepthConcat concatenates the output of one layer of "parallel" modules along the
provided dimension `dim`: they take the same inputs, and their output is
concatenated. For dimensions other than `dim` having different sizes,
the smaller tensors are copied in the center of the output tensor, 
effectively padding the borders with zeros.

The module is particularly useful for concatenating the output of [Convolutions](convolution.md) 
along the depth dimension (i.e. `nOutputFrame`). 
This is used to implement the *DepthConcat* layer 
of the [Going deeper with convolutions](http://arxiv.org/pdf/1409.4842v1.pdf) article.
The normal [Concat](#nn.Concat) Module can't be used since the spatial 
dimensions (height and width) of the output Tensors requiring concatenation 
may have different values. To deal with this, the output uses the largest 
spatial dimensions and adds zero-padding around the smaller Tensors.
```lua
inputSize = 3
outputSize = 2
input = torch.randn(inputSize,7,7)
mlp=nn.DepthConcat(1);
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 1, 1))
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 3, 3))
mlp:add(nn.SpatialConvolutionMM(inputSize, outputSize, 4, 4))
print(mlp:forward(input))
```
which gives the output:
```lua
(1,.,.) = 
 -0.2874  0.6255  1.1122  0.4768  0.9863 -0.2201 -0.1516
  0.2779  0.9295  1.1944  0.4457  1.1470  0.9693  0.1654
 -0.5769 -0.4730  0.3283  0.6729  1.3574 -0.6610  0.0265
  0.3767  1.0300  1.6927  0.4422  0.5837  1.5277  1.1686
  0.8843 -0.7698  0.0539 -0.3547  0.6904 -0.6842  0.2653
  0.4147  0.5062  0.6251  0.4374  0.3252  0.3478  0.0046
  0.7845 -0.0902  0.3499  0.0342  1.0706 -0.0605  0.5525

(2,.,.) = 
 -0.7351 -0.9327 -0.3092 -1.3395 -0.4596 -0.6377 -0.5097
 -0.2406 -0.2617 -0.3400 -0.4339 -0.3648  0.1539 -0.2961
 -0.7124 -1.2228 -0.2632  0.1690  0.4836 -0.9469 -0.7003
 -0.0221  0.1067  0.6975 -0.4221 -0.3121  0.4822  0.6617
  0.2043 -0.9928 -0.9500 -1.6107  0.1409 -1.3548 -0.5212
 -0.3086 -0.0298 -0.2031  0.1026 -0.5785 -0.3275 -0.1630
  0.0596 -0.6097  0.1443 -0.8603 -0.2774 -0.4506 -0.5367

(3,.,.) = 
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000 -0.7326  0.3544  0.1821  0.4796  1.0164  0.0000
  0.0000 -0.9195 -0.0567 -0.1947  0.0169  0.1924  0.0000
  0.0000  0.2596  0.6766  0.0939  0.5677  0.6359  0.0000
  0.0000 -0.2981 -1.2165 -0.0224 -1.1001  0.0008  0.0000
  0.0000 -0.1911  0.2912  0.5092  0.2955  0.7171  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000

(4,.,.) = 
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000 -0.8263  0.3646  0.6750  0.2062  0.2785  0.0000
  0.0000 -0.7572  0.0432 -0.0821  0.4871  1.9506  0.0000
  0.0000 -0.4609  0.4362  0.5091  0.8901 -0.6954  0.0000
  0.0000  0.6049 -0.1501 -0.4602 -0.6514  0.5439  0.0000
  0.0000  0.2570  0.4694 -0.1262  0.5602  0.0821  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000

(5,.,.) = 
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000  0.3158  0.4389 -0.0485 -0.2179  0.0000  0.0000
  0.0000  0.1966  0.6185 -0.9563 -0.3365  0.0000  0.0000
  0.0000 -0.2892 -0.9266 -0.0172 -0.3122  0.0000  0.0000
  0.0000 -0.6269  0.5349 -0.2520 -0.2187  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000

(6,.,.) = 
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000  1.1148  0.2324 -0.1093  0.5024  0.0000  0.0000
  0.0000 -0.2624 -0.5863  0.3444  0.3506  0.0000  0.0000
  0.0000  0.1486  0.8413  0.6229 -0.0130  0.0000  0.0000
  0.0000  0.8446  0.3801 -0.2611  0.8140  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
[torch.DoubleTensor of dimension 6x7x7]
```
Note how the last 2 of 6 filter maps have 1 column of zero-padding 
on the left and top, as well as 2 on the right and bottom. 
This is inevitable when the component
module output tensors non-`dim` sizes aren't all odd or even. 
Such that in order to keep the mappings aligned, one need 
only ensure that these be all odd (or even).

<a name="nn.TableContainers"></a>
## Table Containers ##
While the above containers are used for manipulating input [Tensors](https://github.com/torch/torch7/blob/master/doc/tensor.md), table containers are used for manipulating tables :
 * [ConcatTable](table.md#nn.ConcatTable)
 * [ParallelTable](table.md#nn.ParallelTable)

These, along with all other modules for manipulating tables can be found [here](table.md).
