require("nn")


dofile("SparseCriterion.lua")
dofile("SparseTools.lua")

local tester = torch.Tester()
local sparsifier = function(x) if torch.uniform() < 0.6 then return 0 else return x end end


local sparseCriterionTester = {}

function sparseCriterionTester.oneSample()

   local rmseFct = nn.SparseCriterion(nn.MSECriterion())

   local input  = torch.Tensor(100):uniform():apply(sparsifier)
   local output = torch.Tensor(100):uniform()
   
   -- compute the objective with dense vector
   local mask    = input:ne(0)
   local rmseFct = nn.MSECriterion()
   local expectedScore = rmseFct:forward(output[mask], input[mask]) --autoencoder loss
   
   
   -- use sparse vector
   local sparseInput = input:sparsify()
   local sparseRmseFct = nn.SparseCriterion(nn.MSECriterion())
   local obtainedScore = sparseRmseFct:forward(output, sparseInput) --autoencoder loss
   
   tester:asserteq(expectedScore, obtainedScore, 'Fail to compute autoencoder sparse loss given a single sample')
   
   
   --autoencoder loss
   local expectedDLoss = rmseFct:backward(output, input) --autoencoder loss
   local obtainedDLoss = sparseRmseFct:backward(output, sparseInput) 
   
   expectedDLoss[mask:eq(0)] = 0 --set 0 to sparse values

   tester:assertTensorEq(expectedDLoss, obtainedDLoss, 0, 'Fail to compute autoencoder sparse Dloss given a single sample')
   
end


function sparseCriterionTester.miniBatch()

   local rmseFct = nn.SparseCriterion(nn.MSECriterion())

   local input  = torch.Tensor(10,100):uniform():apply(sparsifier)
   local output = torch.Tensor(10,100)
   
   -- compute the objective with dense vector
   local mask    = input:ne(0)
   local rmseFct = nn.MSECriterion()
   local expectedLoss  = rmseFct:forward(output[mask] , input[mask]) --we need to mult
   
   
   -- use sparse vector
   local sparseInput = input:sparsify()
   local sparseRmseFct = nn.SparseCriterion(nn.MSECriterion())
   local obtainedLoss  = sparseRmseFct:forward(output, sparseInput) --autoencoder loss
      
   tester:assertalmosteq(expectedLoss , obtainedLoss, 1e-6, 'Fail to compute autoencoder sparse Dloss given a mini-batch')
   
   
   --autoencoder loss
   local expectedDLoss = rmseFct:backward(output, input) --autoencoder loss
   local obtainedDLoss = sparseRmseFct:backward(output, sparseInput) 
   
   expectedDLoss[mask:eq(0)] = 0 --set 0 to sparse values

   tester:assertTensorEq(expectedDLoss, obtainedDLoss, 1e-6, 'Fail to compute autoencoder sparse loss given a mini-batch')
   
end






print('')
print('Testing sparseCriterion.lua')
print('')

math.randomseed(os.time())


tester:add(sparseCriterionTester)
tester:run()