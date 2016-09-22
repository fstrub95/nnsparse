require("nn")

local info = debug.getinfo(1,'S')
local src_path = info.source:gsub("@", ""):gsub("test/(.*)", "src/")

if not nnsparse then
   nnsparse = {}
end

if not nnsparse.Densify then
   dofile(src_path .. "SparseTools.lua")
end

dofile(src_path .. "SparseCriterion.lua")

local tester = torch.Tester()
local sparsifier = function(x) if torch.uniform() < 0.6 then return 0 else return x end end


local sparseCriterionTester = {}

function sparseCriterionTester.oneSample()

   local input  = torch.Tensor(100):uniform():apply(sparsifier)
   local output = torch.Tensor(100):uniform()
   
   local sparseInput = input:sparsify()
   
   -- compute the objective with dense vector
   local maskDense  = input:ne(0)
   local maskSparse = input:eq(0)

   
   local mseFct       = nn.MSECriterion()
   local sparseMseFct = nnsparse.SparseCriterion(nn.MSECriterion())
   
   
   local expectedLoss  = mseFct      :forward(output[maskDense], input[maskDense])
   local obtainedLoss  = sparseMseFct:forward(output           , sparseInput)
      
   tester:assertalmosteq(expectedLoss , obtainedLoss, 1e-6, 'Fail to compute autoencoder sparse loss given a single sample')
   
   
   --autoencoder loss
   local expectedDLoss = mseFct      :backward(output[maskDense], input[maskDense])
   local obtainedDLoss = sparseMseFct:backward(output          , sparseInput) 
   
   tester:assertalmosteq(expectedDLoss:sum(), obtainedDLoss:sum(), 1e-6, 'Fail to compute autoencoder sparse Dloss given a single sample')
   
end


function sparseCriterionTester.miniBatch()

   local rmseFct = nnsparse.SparseCriterion(nn.MSECriterion())

   local input  = torch.Tensor(10,100):uniform():apply(sparsifier)
   local output = torch.Tensor(10,100):uniform()

   local maskDense  = input:ne(0)
   local maskSparse = input:eq(0)
   output[maskSparse] = 0
   
   local sparseInput = input:sparsify()

   -- compute the objective with dense vector
   local mseFct       = nn.MSECriterion()
   local sparseMseFct = nnsparse.SparseCriterion(nn.MSECriterion())
   
   
   local expectedLoss  = mseFct      :forward(output[maskDense] , input[maskDense])
   local obtainedLoss  = sparseMseFct:forward(output            , sparseInput)
      
   tester:assertalmosteq(expectedLoss , obtainedLoss, 1e-6, 'Fail to compute autoencoder sparse loss given a mini-batch')
   
   
   --autoencoder loss
   local expectedDLoss = mseFct      :backward(output, input)
   local obtainedDLoss = sparseMseFct:backward(output, sparseInput)

   tester:assertTensorEq(expectedDLoss, obtainedDLoss, 1e-6, 'Fail to compute autoencoder sparse Dloss given a mini-batch')
   
end

   
function sparseCriterionTester.miniBatchWithNoSizeAverage()

   local input  = torch.Tensor(10,100):uniform():apply(sparsifier)
   local output = torch.Tensor(10,100):uniform()
   
   local sparseInput = input:sparsify()
   
   -- compute the objective with dense vector
   local maskDense  = input:ne(0)
   local maskSparse = input:eq(0)
   
   
   
   local mseFct       = nn.MSECriterion()
   local sparseMseFct = nnsparse.SparseCriterion(nn.MSECriterion())
   
   
   mseFct.sizeAverage = false
   sparseMseFct.sizeAverage = false
   
   
   local expectedLoss  = mseFct      :forward(output[maskDense] , input[maskDense])
   local obtainedLoss  = sparseMseFct:forward(output            , sparseInput)
      

   tester:assertalmosteq(expectedLoss , obtainedLoss, 1e-6, 'Fail to compute autoencoder sparse loss given a mini-batch with no average')
   
   --autoencoder loss
   local expectedDLoss = mseFct      :backward(output[maskDense], input[maskDense])
   local obtainedDLoss = sparseMseFct:backward(output           , sparseInput) 
   
   tester:assertalmosteq(expectedDLoss:sum(), obtainedDLoss:sum(), 1e-6, 'Fail to compute autoencoder sparse Dloss given a mini-batch with no average')
end




print('')
print('Testing sparseCriterion.lua')
print('')

math.randomseed(os.time())


tester:add(sparseCriterionTester)
tester:run()