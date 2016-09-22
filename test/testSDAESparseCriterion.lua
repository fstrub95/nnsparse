require("nn")

local info = debug.getinfo(1,'S')
local src_path = info.source:gsub("@", ""):gsub("test/(.*)", "src/")

if not nnsparse then
   nnsparse = {}
end

if not nnsparse.Densify then
   dofile(src_path .. "SparseTools.lua")
end

if not nnsparse.SDAECriterion then
   dofile(src_path .. "SDAECriterion.lua")
end

dofile(src_path .. "SDAESparseCriterion.lua")

local sparsifier = function(x) if torch.uniform() < 0.6 then return 0 else return x end end

function torch.Tester:assertbw(val, condition, tolerance, message)
   self:assertgt(val, condition*(1-tolerance), message)
   self:assertlt(val, condition*(1+tolerance), message)
end



local tester = torch.Tester()

local SDAESparseCriterionTester = {}


function SDAESparseCriterionTester.prepareHidden()

   local input = torch.ones(10, 10000):apply(sparsifier):sparsify()
   
   local hideRatio = 0.5
   local criterion = nnsparse.SDAESparseCriterion(nn.MSECriterion(), 
   {
      hideRatio = hideRatio
   })
   
   local noisyInput = criterion:prepareInput(input)
   
   for _, oneInput in pairs(noisyInput) do
   
      local oneInput = oneInput[{{}, 2}] --remove index

      local obtainedRatio = oneInput:ne(0):sum()/oneInput:size(1)
      tester:assertbw(obtainedRatio, hideRatio, 0.10, "Check number of corrupted input : Hide")

   end 

end

function SDAESparseCriterionTester.prepareGauss()

   local input = torch.ones(10, 10000):apply(sparsifier):sparsify()
   
   local noiseRatio = 0.3
   local mean = 5
   local std = 2
   local criterion = nnsparse.SDAESparseCriterion(nn.MSECriterion(), 
   {
      noiseRatio = noiseRatio,
      noiseMean  = mean,
      noiseStd   = std,
   })
   
   local noisyInput = criterion:prepareInput(input)
   
   for _, oneInput in pairs(noisyInput) do
   
      local oneInput = oneInput[{{}, 2}] --remove index

      local obtainedRatio = oneInput:ne(1):sum()/oneInput:size(1)
      
      local mask     = oneInput:ne(1)
      local obtainedMean  = oneInput[mask]:mean() - 1 
      local obtainedStd   = oneInput[mask]:std()
      
      tester:assertbw(obtainedRatio, noiseRatio, 0.10, "Check number of corrupted input : Gauss")
      tester:assertbw(obtainedMean,  mean      , 0.10, "Check number of corrupted input : Gauss (mean)")
      tester:assertbw(obtainedStd,   std       , 0.10, "Check number of corrupted input : Gauss (std)")
   end 
   
end


function SDAESparseCriterionTester.prepareSaltAndPepper()

   local input = torch.ones(10, 10000):apply(sparsifier):sparsify()
   
   local flipRatio = 0.8
   
   local criterion = nnsparse.SDAESparseCriterion(nn.MSECriterion(), 
   {
      flipRatio = flipRatio,
      flipRange = {-99, 99},
   })
   
   local noisyInput = criterion:prepareInput(input)
   
   for _, oneInput in pairs(noisyInput) do
   
      local oneInput = oneInput[{{}, 2}] --remove index

      local noFlip = oneInput:eq( 99):sum()/oneInput:size(1)
      local noFlap = oneInput:eq(-99):sum()/oneInput:size(1)
      
      tester:assertbw(noFlip, flipRatio/2, 0.10, "Check number of corrupted input : SaltAndPepper")
      tester:assertbw(noFlap, flipRatio/2, 0.10, "Check number of corrupted input : SaltAndPepper")
   
   end 
   
end


function SDAESparseCriterionTester.prepareMixture()

   local input = torch.ones(10, 10000):apply(sparsifier):sparsify()
   
   local hideRatio  = 0.2
   local flipRatio  = 0.3
   
   local criterion = nnsparse.SDAESparseCriterion(nn.MSECriterion(), 
   {
      hideRatio  = hideRatio,
      flipRatio  = flipRatio,
      flipRange  = {999, 999},
   })
   
   local noisyInput = criterion:prepareInput(input)
   
   for _, oneInput in pairs(noisyInput) do
   
      local oneInput = oneInput[{{}, 2}] --remove index

      local noHide  = oneInput:eq( 0 ):sum()/oneInput:size(1)
      local noFlip  = oneInput:eq(999):sum()/oneInput:size(1)

      tester:assertbw(noHide , hideRatio , 0.10, "Check number of corrupted input : hide          (mixture)")
      tester:assertbw(noFlip , flipRatio , 0.10, "Check number of corrupted input : SaltAndPepper (mixture)")

   end 
   
end



function SDAESparseCriterionTester.NoNoise()


   local beta = 0.5
   
   local basicCriterion = nn.MSECriterion()
   local sdaeCriterion  = nnsparse.SDAESparseCriterion(nn.MSECriterion(), 
   {
      alpha = 1,
      beta =  beta,
      hideRatio  = 0,
      noiseRatio = 0,
      flipRatio  = 0,
   })
   
   local input       = torch.ones(10, 100)
   local sparseInput = input:sparsify() 
   local noisyInput  = sdaeCriterion:prepareInput(sparseInput)

   local output = torch.Tensor(10, 100):uniform()
   
   
   local expectedLoss = basicCriterion:forward(output, input     )
   local obtainedLoss = sdaeCriterion:forward (output, noisyInput)

   tester:assertalmosteq(obtainedLoss, expectedLoss*beta, 1e-6, 'Fail to compute sparse SDAE loss with no noise')

   local expectedDLoss = basicCriterion:backward(output, input     )
   local obtainedDLoss = sdaeCriterion:backward (output, noisyInput)
   
   tester:assertTensorEq(obtainedDLoss, expectedDLoss:mul(beta), 1e-6, 'Fail to compute sparse SDAE Dloss with no noise')

end




function SDAESparseCriterionTester.NoNoiseNoSizeAverage()

   local beta = 0.5
   
   local basicCriterion = nn.MSECriterion()
   local sdaeCriterion  = nnsparse.SDAESparseCriterion(nn.MSECriterion(), 
   {
      alpha = 1,
      beta =  beta,
      hideRatio  = 0,
      noiseRatio = 0,
      flipRatio  = 0,
   })
   
   basicCriterion.sizeAverage = false
   sdaeCriterion.sizeAverage = false
   
   local input       = torch.ones(10, 100)
   local sparseInput = input:sparsify() 
   local noisyInput  = sdaeCriterion:prepareInput(sparseInput)

   local output = torch.Tensor(10, 100):uniform()
   
   
   local expectedLoss = basicCriterion:forward(output, input     )
   local obtainedLoss = sdaeCriterion:forward (output, noisyInput)

   tester:assertalmosteq(obtainedLoss, expectedLoss*beta, 1e-6, 'Fail to compute sparse SDAE loss with no noise')

   local expectedDLoss = basicCriterion:backward(output, input     )
   local obtainedDLoss = sdaeCriterion:backward (output, noisyInput)
   
   tester:assertTensorEq(obtainedDLoss, expectedDLoss:mul(beta), 1e-6, 'Fail to compute sparse SDAE Dloss with no noise')

end



function SDAESparseCriterionTester.WithNoise()

   local input       = torch.Tensor(10, 100):uniform():apply(sparsifier)
   local output      = torch.Tensor(10, 100):uniform()
   
   local sparseInput = input:sparsify()
   local sparseMask  = input:ne(0)

   local alpha = 0.8
   local beta  = 0.3
   
  local criterion = nnsparse.SDAESparseCriterion(nn.MSECriterion(), 
  {
      hideRatio = 0.0,
      alpha = alpha,
      beta =  beta,
  })
  
  local noisyInput = criterion:prepareInput(sparseInput)
  
  -- retrieve the noise mask
  local maskAlpha
  local contiguousInput
  for k, mask in pairs(criterion.masks) do
      
      if maskAlpha  then maskAlpha = maskAlpha:cat(mask.alpha)
      else               maskAlpha = mask.alpha:clone()
      end 
      
      if contiguousInput then contiguousInput = contiguousInput:cat(noisyInput[k][{{},2}])
      else                    contiguousInput = noisyInput[k][{{},2}]
      end 
  end
  local maskBeta= maskAlpha:eq(0)
  


  -- compute the SDAE loss using the formula
  local diff = output[sparseMask]:clone():add(-1,contiguousInput):pow(2):view(-1)
  diff[maskAlpha] = diff[maskAlpha]* alpha
  diff[maskBeta ] = diff[maskBeta ]* beta

  local expectedLoss = diff:sum() / diff:nElement()
  local obtainedLoss = criterion:forward(output, noisyInput)
   
  tester:assertalmosteq(expectedLoss, obtainedLoss, 1e-6, 'Fail to compute SDAE loss with noise')


  -- compute the SDAE dloss using the formula
  local diff = output[sparseMask]:clone():add(-1,contiguousInput):mul(2):view(-1)
  diff[maskAlpha] = diff[maskAlpha]* alpha
  diff[maskBeta ] = diff[maskBeta ]* beta

  local expectedDLossSum = diff:sum() / diff:nElement() 
  local obtainedDLossSum = criterion:backward(output,noisyInput):sum()
  
  tester:assertalmosteq(expectedDLossSum, obtainedDLossSum, 1e-6, 'Fail to compute SDAE Dloss with noise')
  
end




--function SDAESparseCriterionTester.NoNoiseSizeAverageWithReducedSize()
--
--   local beta = 0.5
--   
--   local basicCriterion = nn.MSECriterion()
--   local sdaeCriterion  = nnsparse.SDAESparseCriterion(nn.MSECriterion(), 
--   {
--      alpha = 1,
--      beta =  beta,
--      hideRatio  = 0,
--      noiseRatio = 0,
--      flipRatio  = 0,
--   })
--   
--   basicCriterion.sizeAverage = false
--   sdaeCriterion.sizeAverage  = false
--   
--   sdaeCriterion.sizeAverage2 = true
--   
--   local input       = torch.ones(10, 100)
--   local sparseInput = input:sparsify() 
--   local noisyInput  = sdaeCriterion:prepareInput(sparseInput)
--
--   local output = torch.Tensor(10, 100):uniform()
--   
--   
--   local expectedLoss = 0
--   for i = 1, input:size(1) do
--      local mask = input[i]:ne(0)
--      expectedLoss = expectedLoss + basicCriterion:forward(output[i][mask], input[i][mask]) / mask:size(1)
--   end
--   expectedLoss = expectedLoss * beta
--   
--   local obtainedLoss = sdaeCriterion:forward(output, noisyInput)
--   
--
--   tester:assertalmosteq(obtainedLoss, expectedLoss, 1e-6, 'Fail to compute sparse SDAE loss with no noise')
--
--
--   local expectedDLossSum = 0
--   for i = 1, input:size(1) do
--      local mask = input[i]:ne(0)
--      local curDLoss = basicCriterion:backward(output[i][mask], input[i][mask])
--      expectedDLossSum = expectedDLossSum + beta*(curDLoss:sum() / mask:size(1))
--   end
--   expectedDLossSum = expectedDLossSum * beta
--   
--   local obtainedDLossSum = sdaeCriterion:backward(output, noisyInput):sum()
--   
--   tester:assertalmosteq(obtainedDLossSum, obtainedDLossSum, 1e-6, 'Fail to compute sparse SDAE Dloss with no noise')
--
--
--end




print('')
print('Testing SDAESparseCriterion.lua')
print('')

math.randomseed(os.time())


tester:add(SDAESparseCriterionTester)
tester:run()