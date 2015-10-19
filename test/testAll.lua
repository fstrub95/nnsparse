require("torch")

--NB some test may fail with float because of rounded values
--torch.setdefaulttensortype('torch.FloatTensor')

dofile("testSparseTools.lua")
dofile("testSparseLinearBatch.lua")
dofile("testSparseCriterion.lua")
dofile("testSDAECriterion.lua")