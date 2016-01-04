require("torch")
require("nn")

nnsparse = {}

include('SparseTools.lua')

include('SDAECriterion.lua')
include('SDAESparseCriterion.lua')
include('SparseCriterion.lua')

include('SparseLinearBatch.lua')

return nnsparse
