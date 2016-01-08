

---------------------------------------
-- Basic Constant 

NaN =  0/0 -- -999


function torch.Tensor.sparsify(self, elem, offset)

   elem = elem or 0
   offset = offset or 0

   if self:nDimension() == 1 then

      --I assume it is faster to compute the number of 0 to pre-allocate memory --> (benchamrk required)
      local sparseSize = self:size(1)-self:eq(elem):sum()  
      local sparseVector = torch.Tensor(sparseSize, 2) --pre-allocate memory 

      local sparseIndex = 1
      for i = 1, self:size(1) do
         if self[i] ~= elem then 
            sparseVector[sparseIndex][1] = i + offset
            sparseVector[sparseIndex][2] = self[i]
            sparseIndex = sparseIndex + 1
         end 
      end

      return sparseVector

   elseif self:nDimension() == 2 then
      local sparseMatrix = {}
      for i = 1, self:size(1) do
         sparseMatrix[i] = self[i]:sparsify()
      end

      return sparseMatrix

   else
      -- sparsification can be done recursively for dim > 2
      error("Cannot sparsify matrix with dimension greater than 2")
   end

end


local function GetSize(X,dim)
   if torch.isTensor(X)  then 
      if dim == nil then dim = 1 end
      return X:size(dim)
   elseif torch.type(X) == "table" then 
      return #X
   else return nil
   end
end

local function GetnElement(X) 
   if torch.isTensor(X)  then 
      return X:nElement()
   elseif torch.type(X) == "table" then 
      local size = 0
      for _, _ in pairs(X) do size = size + 1 end
      return size
   else return nil
   end

end


function torch.Tensor.densify(self, elem, dim)

   elem = elem or 0

   if torch.type(self) == "table" then       -- TODO: find densify for dimension > 2

      --Step 1 : Look for the dimension
      if GetSize(dim) ~= 2 then 
         dim = {0,0}
         for k, oneInput in pairs(self) do
            dim[1] = math.max(dim[1], k)                       -- look for the highest number of row when the table is not sorted
            dim[2] = math.max(dim[2], oneInput[{{},1}]:max())  -- look for the highest number of col
         end
   end

   --Step 2 : densify the matrix
   assert(dim)
   local denseMatrix = self[dim[1]].new()
   denseMatrix:resize(dim[1], dim[2]):fill(elem)

   for k, oneInput in pairs(self) do
      local index = oneInput[{{},1}]
      local data  = oneInput[{{},2}]

      denseMatrix[k]:indexCopy(1, index:long(), data)
   end            

   return denseMatrix

   elseif self:nDimension() == 2 then

      local index = self[{{},1}]
      local data  = self[{{},2}]

      dim = dim or index:max()

      local denseVector = self.new()
      denseVector:resize(dim):fill(elem)
      denseVector:indexCopy(1, index:long(), data)

      return denseVector

   else

      error("Cannot sparsify matrix with dimension greater than 2")
   end

end

local Densify, parent = torch.class('nnsparse.Densify', 'nn.Module')

function Densify:__init(inputSize)
   parent.__init(self)   
   self.inputSize = inputSize
   self.output = nil
end

function Densify:updateOutput(input)

   if torch.isTensor(input) then input = {input} end

   -- TODO: find densify for dimension > 2
   local xSize = #input
   local ySize = self.inputSize

   self.output = self.output or input[1].new()
   self.output:resize(xSize, ySize):zero()

   for k, oneInput in pairs(input) do
      local index = oneInput[{{},1}]
      local data  = oneInput[{{},2}]

      if torch.type(index) ~= "torch.CudaTensor" then
         index= index:long()
      end

      self.output[k]:indexCopy(1, index, data)
   end            

   return self.output

end

function Densify:updateGradInput(input, gradOutput)
   return gradOutput
end

local Sparsifier, parent = torch.class('nnsparse.Sparsifier', 'nn.Module')


function Sparsifier:__init(offset)
   parent.__init(self)  
   self.offset = offset or 0
end

function Sparsifier:updateOutput(input)
   assert(torch.isTensor(input))
   self.output = input:sparsify(0, self.offset)
   return self.output
end

function Sparsifier:updateGradInput(input, gradOutput)
   return gradOutput
end


local Batchifier, parent = torch.class('nnsparse.Batchifier')

function Batchifier:__init(network, outputSize)
   self.network    = network
   self.outputSize = outputSize
end



function Batchifier:forward(data, batchSize, nFrame)
   
   -- no need for batch for tense Tensor
   if torch.isTensor(data) then
      return self.network:forward(data)
   end
      
   batchSize = batchSize or 10
   nFrame    = nFrame    or GetnElement(data)

   --Prepare minibatch
   local inputs   = {}
   local outputs  = data[#data].new(nFrame, self.outputSize) 

   assert(torch.type(data) == "table")

   local i      = 1
   local cursor = 0
   for _, input in pairs(data) do

      inputs[i]  = input   
      i = i + 1

      --compute loss when minibatch is ready
      if #inputs == batchSize then
         local start =  cursor   *batchSize + 1
         local stop  = (cursor+1)*batchSize

         outputs[{{start,stop},{}}] = self.network:forward(inputs)
         
         inputs = {}
         
         i = 1
         cursor = cursor + 1       
      end
   end

   if #inputs > 0 then
      local start = nFrame-(i-1) + 1
      local stop  = nFrame

      outputs[{{start,stop},{}}] = self.network:forward(inputs)
   end  

   return outputs

end




function torch.Tensor.sortDim(self, dim, descend, inplace)

   descend = descend or false

   assert(self:nDimension() == 2)


   if torch.isTensor(self) and self:nDimension() == 2 then

      --TODO: optim -> implement inplace sort with no buffer.    
      local _ , index = self[{{},dim}]:sort(descend)

      local buf = self.new()
      buf:resizeAs(self)

      for k = 1, index:size(1) do
         buf[k] = self[index[k]]   
      end

      if inplace == true then
         self:copy(buf)
      else
         return buf
      end


   elseif torch.type(self) == "table" then
      for k, x in pairs(self) do
         self[k] = torch.ssort(x)
      end
   else
      error("Type or dimension [" .. torch.type(self) .. "] is not supported")
   end

end


function torch.Tensor.ssortByIndex(self, descend, inplace)
   return torch.Tensor.sortDim(self, 1, descend, inplace)
end

function torch.Tensor.ssort(self, descend, inplace)
   return torch.Tensor.sortDim(self, 2, descend, inplace)
end



local DynamicSparseTensor = torch.class("nnsparse.DynamicSparseTensor")

function DynamicSparseTensor:__init(reserve, multCoef)
  self.reserve = reserve or 10
  self.curSize = 0
  self.multCoef = multCoef or 2
  self.tensor  = torch.Tensor(self.reserve, 2)
end


function DynamicSparseTensor:set(X)
  self.curSize = X:size(1)
  self.tensor  = X
end

function DynamicSparseTensor:reset(X)
  self.curSize = 0
  self.tensor  = torch.Tensor(self.reserve, 2)
end

function DynamicSparseTensor:append(x)
  
  if self.curSize == self.tensor:size(1) then
      local buffer = torch.Tensor(self.tensor:size(1)*self.multCoef/2, 2)
      self.tensor = self.tensor:cat(buffer,1)
  end 
  
  self.curSize = self.curSize + 1
  self.tensor[self.curSize] = x
  
end

function DynamicSparseTensor:build()
   return self.tensor:resize(self.curSize, 2)
end

