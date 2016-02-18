require("torch")
require("nn")

nnsparse = {}

include('SparseTools.lua')

include('SDAECriterion.lua')
include('SDAESparseCriterion.lua')
include('SparseCriterion.lua')

include('SparseLinearBatch.lua')

include('SparseSorting.lua')
include('Batchifier.lua')
include('DynamicSparseTensor.lua')



return nnsparse
