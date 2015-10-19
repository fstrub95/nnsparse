dofile("SparseTools.lua")

local tester = torch.Tester()

local toolsTester = {}

function toolsTester.sparsifyVector()
   local input  = torch.Tensor{1,0,4,0,0,2,0,0}
   local target = torch.Tensor{{1,1},{3,4},{6,2}}
   local output = input:sparsify()
   
   tester:assertTensorEq(output, target, 0, 'Fail to turn a dense vector into a sparse one')
   
end

function toolsTester.sparsifyMatrix()
   local input  = torch.Tensor{{1,0,4,0,0,2},
                               {0,0,0,8,0,0}}
                               
   local target = {
         torch.Tensor{{1,1},{3,4},{6,2}},
         torch.Tensor{{4,8}}
         }
         
   local output = input:sparsify()
         
   for k, sparseVec in pairs(output) do
      tester:assertTensorEq(sparseVec, target[k], 0, 'Fail to turn a dense matrix into a sparse one')
   end
   
end

print('')
print('Testing tools.lua')
print('')

math.randomseed(os.time())


tester:add(toolsTester)
tester:run()

