require("nn")


dofile("SparseLinearBatch.lua")
dofile("SparseTools.lua")


local tester = torch.Tester()

local sparseLinearBatch = {}
local sparsifier = function(x) if torch.uniform() < 0.6 then return 0 else return x end end


function sparseLinearBatch.oneSample() 

   local input = torch.Tensor(100):uniform():apply(sparsifier)
   local dloss = torch.Tensor(50):uniform()

   local sparseInput = input:sparsify()
   
   local inputSize, outputSize = input:size(1), dloss:size(1)

   local denseLayer  = nn.Linear(inputSize, outputSize)
   local sparseLayer = nn.SparseLinearBatch(inputSize, outputSize)

   -- provide the same weights
   denseLayer.bias   = sparseLayer.bias:clone()
   denseLayer.weight = sparseLayer.weight:clone()
   
   denseLayer:zeroGradParameters()
   sparseLayer:zeroGradParameters()
   
   --forward
   local expectedOutput = denseLayer:forward(input)
   local obtainedOutput = sparseLayer:forward(sparseInput)
   
   tester:assertTensorEq(expectedOutput, obtainedOutput, 1e-6, 'Fail to compute sparseLinearBatch:forward given a single sample')
 
   --backward
   denseLayer:backward (input      , dloss)
   sparseLayer:backward(sparseInput, dloss)
   
   tester:assertTensorEq(denseLayer.gradWeight, sparseLayer.gradWeight, 1e-6, 'Fail to compute sparseLinearBatch:backward given a single sample')
   tester:assertTensorEq(denseLayer.gradBias  , sparseLayer.gradBias  , 1e-6, 'Fail to compute sparseLinearBatch:backward given a single sample')
   
end


function sparseLinearBatch.miniBatch() 

   local input = torch.Tensor(10,100):uniform():apply(sparsifier)
   local dloss = torch.Tensor(10, 50):uniform()
                                    
   local sparseInput = input:sparsify()
   
   local inputSize, outputSize = input:size(2), dloss:size(2)

   local denseLayer  = nn.Linear(inputSize, outputSize)
   local sparseLayer = nn.SparseLinearBatch(inputSize, outputSize)

   -- provide the same weights
   denseLayer.bias   = sparseLayer.bias:clone()
   denseLayer.weight = sparseLayer.weight:clone()
   
   denseLayer:zeroGradParameters()
   sparseLayer:zeroGradParameters()
   
   --forward
   local expectedOutput = denseLayer:forward(input)
   local obtainedOutput = sparseLayer:forward(sparseInput)
   
   tester:assertTensorEq(expectedOutput, obtainedOutput, 1e-6, 'Fail to compute sparseLinearBatch:forward given for a minibatch')
 
 
   --backward
   denseLayer:backward (input      , dloss)
   sparseLayer:backward(sparseInput, dloss)
   
   tester:assertTensorEq(denseLayer.gradWeight, sparseLayer.gradWeight, 1e-6, 'Fail to compute sparseLinearBatch:backward for a minibatch : gradWeight')
   tester:assertTensorEq(denseLayer.gradBias  , sparseLayer.gradBias  , 1e-6, 'Fail to compute sparseLinearBatch:backward for a minibatch : bias')
   
end





print('')
print('Testing sparseLinearBatch.lua')
print('')

math.randomseed(os.time())


tester:add(sparseLinearBatch)
tester:run()