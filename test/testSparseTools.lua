require("nn")

require("torch")
require("nn")


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



print('')
print('Testing tools.lua')
print('')

math.randomseed(os.time())


tester:add(toolsTester)
tester:run()

