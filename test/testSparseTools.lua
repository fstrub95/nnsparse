require("nn")

require("torch")
require("nn")
require("sys")

local info = debug.getinfo(1,'S')
local src_path = info.source:gsub("@", ""):gsub("test/(.*)", "src/")

if not nnsparse then
   nnsparse = {}
end

dofile(src_path .. "SparseTools.lua")
dofile(src_path .. "SparseSorting.lua")
dofile(src_path .. "DynamicSparseTensor.lua")

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



function toolsTester.densifyVector()
   local input = torch.Tensor{{1,1},{3,4},{6,2}}
   local target  = torch.Tensor{1,0,4,0,0,2}

   local output = input:densify()
   
   tester:assertTensorEq(output, target, 0, 'Fail to turn a sparse vector into a dense one : classic')

   local target  = torch.Tensor{1,0,4,0,0,2, 0, 0}
   local output = input:densify(0, 8)
   
   tester:assertTensorEq(output, target, 0,'Fail to turn a sparse vector into a dense one : choose dim')
   
 
   local target  = torch.Tensor{1,99,4,99,99,2}
   local output = input:densify(99)
   
   tester:assertTensorEq(output, target, 0,'Fail to turn a sparse vector into a dense one : choose element')
   
end

function toolsTester.densifyMatrix()

   local input = {
         torch.Tensor{{1,1},{3,4},{6,2}},
         torch.Tensor{{4,8}}
         }
   
   local target  = torch.Tensor{{1,0,4,0,0,2},
                               {0,0,0,8,0,0}}
                                 
   local output = torch.Tensor.densify(input)
   
   tester:assertTensorEq(output, target, 0, 'Fail to turn a dense vector into a sparse one')
   
end

function toolsTester.sortIndex()

   local input   = torch.Tensor{{3,4},{1,1},{6,2}}
   local target  = torch.Tensor{{1,1},{3,4},{6,2}}
                                             
   local output = input:ssortByIndex()
   
   tester:assertTensorEq(output, target, 0, 'Fail sparse vector by index')
   
end

function toolsTester.sortValueAscend()

   local input   = torch.Tensor{{1,1},{3,4},{6,2}}
   local target  = torch.Tensor{{1,1},{6,2},{3,4}}
                                             
   local output = input:ssort()
   
   tester:assertTensorEq(output, target, 0, 'Fail sparse vector by value (Ascend)')
   
end

function toolsTester.sortValueDescend()

   local input   = torch.Tensor{{1,1},{3,4},{6,2}}
   local target  = torch.Tensor{{3,4},{6,2},{1,1}}
                                             
   local output = input:ssort(true)
   
   tester:assertTensorEq(output, target, 0, 'Fail sparse vector by value (Descend)')
   
end

function toolsTester.dynamicSparseTensor()

   --compute dummy sparse vector
   local input = torch.Tensor(287,2)
   local i = 0
   input:apply(function() i = i + 1 return i end)


   --build a dynamic sparse vector
   sys.tic()
   local dynTensor = nnsparse.DynamicSparseTensor.new()
   for i = 1, input:size(1) do
      dynTensor:append(input[i])
   end 
   local finalTensor = dynTensor:build() 
   local tDyn = sys.toc()

   --
   tester:assertTensorEq(input, finalTensor, 0, 'Fail to build sparse dynamic Tensor')
   
   
   sys.tic()
   local staticTensor = torch.Tensor(1,2)
   staticTensor[1][1] = input[1][1]
   staticTensor[1][2] = input[1][2]
   for i = 2, input:size(1) do
      staticTensor = staticTensor:cat(input[i]:view(1,-1), 1)
   end 
   local tStat = sys.toc()
   
   -- check that the benchmark is consistent
   tester:assertTensorEq(input, staticTensor, 0, 'Fail to build sparse static Tensor')
   
   --check that dynamic allocation is faster
   tester:assertlt(tDyn, tStat, 'Static Tensor is faster than dynamic Tensor ;(')

end



print('')
print('Testing tools.lua')
print('')

math.randomseed(os.time())


tester:add(toolsTester)
tester:run()

