require("nn")

local info = debug.getinfo(1,'S')
local src_path = info.source:gsub("@", ""):gsub("test/(.*)", "src/")

if not nnsparse then
   nnsparse = {}
end

if not nnsparse.Densify then
   dofile(src_path .. "SparseTools.lua")
end

dofile(src_path .. "SDAECriterion.lua")


function torch.Tester:assertbw(val, condition, tolerance, message)
   self:assertgt(val, condition*(1-tolerance), message)
   self:assertlt(val, condition*(1+tolerance), message)
end

local tester = torch.Tester()



local SDAECriterionTester = {}


function SDAECriterionTester.prepareHidden()

   local input = torch.ones(10, 1000)
   local hideRatio = 0.5
   
   local criterion = nnsparse.SDAECriterion(nn.MSECriterion(), 
   {
      hideRatio = hideRatio
   })
   
   local noisyInput = criterion:prepareInput(input)
   
   local noHidden = noisyInput:eq(0):sum()/input:nElement()
   tester:assertbw(noHidden, hideRatio, 0.10, "Check number of corrupted input : Hidden")

end

function SDAECriterionTester.prepareGauss()

   local input = torch.ones(10, 1000)
   local noiseRatio = 0.3
   local mean = 5
   local std = 2
   
   local criterion = nnsparse.SDAECriterion(nn.MSECriterion(), 
   {
      noiseRatio = noiseRatio,
      noiseMean  = mean,
      noiseStd   = std,
   })
   
   local noisyInput = criterion:prepareInput(input)
   
   --check that the noise ratio is correct
   local mask    = noisyInput:ne(1)
   local noGauss = mask:sum()/noisyInput:nElement()
   tester:assertbw(noGauss, noiseRatio,0.10, "Check number of corrupted input : Gauss")
   
   --check the mean/std
   local obtainedMean = noisyInput[mask]:mean() - 1 
   local obtainedStd  = noisyInput[mask]:std()
   
   tester:assertbw(obtainedMean, mean, 0.10, "Check the gaussian SDAE mean (Warning: the test may fail for it is based on random values) ")
   tester:assertbw(obtainedStd , std , 0.10, "Check the gaussian SDAE std  (Warning: the test may fail for it is based on random values) ")
   

end


function SDAECriterionTester.prepareFlip()

   local input = torch.zeros(10, 1000)
   local flipRatio = 0.8
   
   local criterion = nnsparse.SDAECriterion(nn.MSECriterion(), 
   {
      flipRatio = flipRatio,
      flipRange = {-99, 99},
   })
   
   local noisyInput = criterion:prepareInput(input)
   
   local noFlip = noisyInput:eq( 99):sum()/noisyInput:nElement()
   local noFlap = noisyInput:eq(-99):sum()/noisyInput:nElement()
   tester:assertbw(noFlip, flipRatio/2, 0.10, "Check number of corrupted input : SaltAndPepper")
   tester:assertbw(noFlap, flipRatio/2, 0.10, "Check number of corrupted input : SaltAndPepper")

   local noFlipFlap = (noisyInput:eq(99):sum() + noisyInput:eq(-99):sum())/noisyInput:nElement()
   tester:assertbw(noFlipFlap, flipRatio, 0.10, "Check number of corrupted input : SaltAndPepper")

end


function SDAECriterionTester.prepareMixture()

   local input = torch.ones(10, 1000)
   local hideRatio  = 0.2
   local flipRatio  = 0.3
   local noiseRatio = 0.4
   
   local criterion = nnsparse.SDAECriterion(nn.MSECriterion(), 
   {
      hideRatio = hideRatio,
      noiseRatio = noiseRatio,
      flipRatio = flipRatio,
      flipRange = {-99, 99},
   })
   
   local noisyInput = criterion:prepareInput(input)
   
   local maskFlip  = noisyInput:eq(99)
   local maskFlap  = noisyInput:eq(-99)
   local maskHide  = noisyInput:eq(0)
   local maskClean = noisyInput:eq(1)
   local maskNoise = maskClean:eq(0) --inverse of clean
   local maskGauss = maskFlip:clone():add(maskFlap):add(maskHide):add(maskClean):eq(0) --inverse of all previous masks
   
   local noFlipFlap  = (maskFlip:sum() + maskFlap:sum()) / noisyInput:nElement()
   local noHide      = maskHide:sum()  / noisyInput:nElement()
   local noGauss     = maskGauss:sum() / noisyInput:nElement()
   
   tester:assertbw(noFlipFlap, flipRatio , 0.10, "Check number of corrupted input (Mixture) : SaltAndPepper")
   tester:assertbw(noHide    , hideRatio , 0.10, "Check number of corrupted input (Mixture) : Hide")
   tester:assertbw(noGauss   , noiseRatio, 0.10, "Check number of corrupted input (Mixture) : Gauss")

   -- Test private implementation -> was the mask correctly stored
   tester:assertTensorEq(criterion.maskAlpha:long(), maskNoise:long(), 0, 'Check alpha mask')
   tester:assertTensorEq(criterion.maskBeta:long() , maskClean:long(), 0, 'Check beta mask')    

end




function SDAECriterionTester.NoNoiseSingleSample()

   local input  = torch.ones(100)
   local output = torch.Tensor(100):uniform()
   
   local beta = 0.5
   
   local basicCriterion = nn.MSECriterion()
   local sdaeCriterion = nnsparse.SDAECriterion(nn.MSECriterion(), 
   {
      alpha = 1,
      beta =  beta,
      hideRatio  = 0,
      noiseRatio = 0,
      flipRatio  = 0,
   })
   
   local noisyInput = sdaeCriterion:prepareInput(input)
   
   local expectedLoss = basicCriterion:forward(output, input     )
   local obtainedLoss = sdaeCriterion:forward (output, noisyInput)

   tester:assertalmosteq(expectedLoss*beta, obtainedLoss, 1e-6, 'Fail to compute SDAE loss with no noise and single sample')

   local expectedDLoss = basicCriterion:backward(output, input     )
   local obtainedDLoss = sdaeCriterion:backward (output, noisyInput)
   
   tester:assertTensorEq(expectedDLoss:mul(beta), obtainedDLoss, 1e-6, 'Fail to compute SDAE Dloss with no noise and single sample')

end



function SDAECriterionTester.NoNoise()

   local input  = torch.ones(10, 100)
   local output = torch.Tensor(10, 100):uniform()
   
   local beta = 0.5
   
   local basicCriterion = nn.MSECriterion()
   local sdaeCriterion = nnsparse.SDAECriterion(nn.MSECriterion(), 
   {
      alpha = 1,
      beta =  beta,
      hideRatio  = 0,
      noiseRatio = 0,
      flipRatio  = 0,
   })
   
   local noisyInput = sdaeCriterion:prepareInput(input)
   
   local expectedLoss = basicCriterion:forward(output, input     )
   local obtainedLoss = sdaeCriterion:forward (output, noisyInput)

   tester:assertalmosteq(expectedLoss*beta, obtainedLoss, 1e-6, 'Fail to compute SDAE loss with no noise')

   local expectedDLoss = basicCriterion:backward(output, input     )
   local obtainedDLoss = sdaeCriterion:backward (output, noisyInput)
   
   tester:assertTensorEq(expectedDLoss:mul(beta), obtainedDLoss, 1e-6, 'Fail to compute SDAE Dloss with no noise')

end


function SDAECriterionTester.NoNoiseSizeAverage()

   local input  = torch.ones(10, 100)
   local output = torch.Tensor(10, 100):uniform()
   
   local beta = 0.5
   
   local basicCriterion = nn.MSECriterion()
   basicCriterion.sizeAverage = false
   
   local sdaeCriterion = nnsparse.SDAECriterion(nn.MSECriterion(), 
   {
      alpha = 1,
      beta =  beta,
      hideRatio  = 0,
      noiseRatio = 0,
      flipRatio  = 0,
   })
   sdaeCriterion.sizeAverage = false
   
   
   local noisyInput = sdaeCriterion:prepareInput(input)
   
   local expectedLoss = basicCriterion:forward(output, input     )
   local obtainedLoss = sdaeCriterion:forward (output, noisyInput)

   tester:assertalmosteq(expectedLoss*beta, obtainedLoss, 1e-6,'Fail to compute SDAE loss with no noise')

   local expectedDLoss = basicCriterion:backward(output, input     )
   local obtainedDLoss = sdaeCriterion:backward (output, noisyInput)
   
   tester:assertTensorEq(expectedDLoss:mul(beta), obtainedDLoss, 1e-6, 'Fail to compute SDAE Dloss with no noise')

end





function SDAECriterionTester.WithNoise()

   local input  = torch.Tensor(10, 100):uniform()
   local output = torch.Tensor(10, 100):uniform()
   
   local alpha = 0.8
   local beta  = 0.3
   
  local criterion = nnsparse.SDAECriterion(nn.MSECriterion(), 
  {
      hideRatio = 0.3,
      alpha = alpha,
      beta =  beta,
  })
  
  local noisyInput = criterion:prepareInput(input)
  local maskAlpha  = criterion.maskAlpha:view(-1)
  local maskBeta  = criterion.maskBeta:view(-1)

  -- compute the SDAE loss using the formula
  local diff = output:clone():add(-1,noisyInput):pow(2):view(-1)
  diff[maskAlpha] = diff[maskAlpha]* alpha
  diff[maskBeta ] = diff[maskBeta ]* beta

  local expectedLoss = diff:sum() / diff:nElement()
  local obtainedLoss = criterion:forward (output, noisyInput)
   
  tester:assertalmosteq(expectedLoss, obtainedLoss, 1e-6, 'Fail to compute SDAE Dloss with noise')



  -- compute the SDAE dloss using the formula
  local diff = output:clone():add(-1,noisyInput):mul(2):view(-1)
  diff[maskAlpha] = diff[maskAlpha]* alpha
  diff[maskBeta ] = diff[maskBeta ]* beta
  
  local expectedDLossSum = diff:sum() / diff:nElement()
  local obtainedDLossSum = criterion:backward(output, noisyInput):sum()
  
  tester:assertalmosteq(expectedDLossSum, obtainedDLossSum, 1e-6, 'Fail to compute SDAE Dloss with noise')
  
end





print('')
print('Testing SDAECriterion.lua')
print('')

math.randomseed(os.time())


tester:add(SDAECriterionTester)
tester:run()