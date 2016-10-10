require("torch")

--NB some test may fail with float because of rounded values
--torch.setdefaulttensortype('torch.FloatTensor')

local info = debug.getinfo(1,'S')
local test_path = info.source:gsub("@", ""):gsub("test/(.*)", "test/")

dofile(test_path .. "testSparseTools.lua")--
dofile(test_path .. "testSparseLinearBatch.lua")-- needs to resolve an issue with the failing test
dofile(test_path .. "testSparseCriterion.lua")--
dofile(test_path .. "testSDAECriterion.lua")--
dofile(test_path .. "testSDAESparseCriterion.lua")--