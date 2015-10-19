require("nn")


dofile("SparseCriterion.lua")
dofile("SparseTools.lua")

local tester = torch.Tester()
local sparsifier = function(x) if torch.uniform() < 0.6 then return 0 else return x end end


local sparseCriterionTester = {}

function sparseCriterionTester.oneSample()


   local input  = torch.Tensor(100):uniform():apply(sparsifier)
   local output = torch.Tensor(100):uniform()
   
   local sparseInput = input:sparsify()
   
   -- compute the objective with dense vector
   local maskSparse = input:eq(0)
   output[maskSparse] = input[maskSparse] --artificially create a zero loss for sparseInput
   
   
   local mseFct       = nn.MSECriterion()
   local sparseMseFct = nn.SparseCriterion(nn.MSECriterion())
   
   
   local expectedLoss  = mseFct      :forward(output , input      ) --we need to mult
   local obtainedLoss  = sparseMseFct:forward(output , sparseInput) --autoencoder loss
      
   tester:assertalmosteq(expectedLoss , obtainedLoss, 1e-6, 'Fail to compute autoencoder sparse loss given a single sample')
   
   
   --autoencoder loss
   local expectedDLoss = mseFct      :backward(output, input) --autoencoder loss
   local obtainedDLoss = sparseMseFct:backward(output, sparseInput) 
   
   tester:assertTensorEq(expectedDLoss, obtainedDLoss, 1e-6, 'Fail to compute autoencoder sparse Dloss given a single sample')
   
end


function sparseCriterionTester.miniBatch()

   local rmseFct = nn.SparseCriterion(nn.MSECriterion())

   local input  = torch.Tensor(10,100):uniform():apply(sparsifier)
   local output = torch.Tensor(10,100):uniform()
   
   local sparseInput = input:sparsify()
   
   -- compute the objective with dense vector
   local maskSparse = input:eq(0)
   output[maskSparse] = input[maskSparse] --artificially create a zero loss for sparseInput
   
   
   local mseFct       = nn.MSECriterion()
   local sparseMseFct = nn.SparseCriterion(nn.MSECriterion())
   
   
   local expectedLoss  = mseFct      :forward(output , input      ) --we need to mult
   local obtainedLoss  = sparseMseFct:forward(output , sparseInput) --autoencoder loss
      
   tester:assertalmosteq(expectedLoss , obtainedLoss, 1e-6, 'Fail to compute autoencoder sparse loss given a mini-batch')
   
   
   --autoencoder loss
   local expectedDLoss = mseFct      :backward(output, input) --autoencoder loss
   local obtainedDLoss = sparseMseFct:backward(output, sparseInput) 
   
   tester:assertTensorEq(expectedDLoss, obtainedDLoss, 1e-6, 'Fail to compute autoencoder sparse Dloss given a mini-batch')
   
end

   
function sparseCriterionTester.miniBatchWithNoSizeAverage()

   local input  = torch.Tensor(10,100):uniform():apply(sparsifier)
   local output = torch.Tensor(10,100):uniform()
   
   local sparseInput = input:sparsify()
   
   -- compute the objective with dense vector
   local maskSparse = input:eq(0)
   output[maskSparse] = input[maskSparse] --artificially create a zero loss for sparseInput
   
   
   local mseFct       = nn.MSECriterion()
   local sparseMseFct = nn.SparseCriterion(nn.MSECriterion())
   
   
   mseFct.sizeAverage = false
   sparseMseFct.sizeAverage = false
   
   
   local expectedLoss  = mseFct      :forward(output , input      ) --we need to mult
   local obtainedLoss  = sparseMseFct:forward(output , sparseInput) --autoencoder loss
      
   if obtainedLoss == NaN then
      print("mmmh")
   end   
      
   tester:assertalmosteq(expectedLoss , obtainedLoss, 1e-6, 'Fail to compute autoencoder sparse loss given a mini-batch with no average')
   
   --autoencoder loss
   local expectedDLoss = mseFct      :backward(output, input) --autoencoder loss
   local obtainedDLoss = sparseMseFct:backward(output, sparseInput) 
   
   tester:assertTensorEq(expectedDLoss, obtainedDLoss, 1e-6, 'Fail to compute autoencoder sparse Dloss given a mini-batch with no average')
end



print('')
print('Testing sparseCriterion.lua')
print('')

math.randomseed(os.time())


tester:add(sparseCriterionTester)
tester:run()