require("torch")

package.path = package.path .. ';../src/*.lua'

torch.setdefaulttensortype('torch.FloatTensor')

dofile("testSparseTools.lua")
dofile("testSparseLinearBatch.lua")
dofile("testSparseCriterion.lua")
dofile("testSDAECriterion.lua")