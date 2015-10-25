

---------------------------------------
-- Basic Constant 

NaN =  0/0 -- -999


function torch.Tensor.sparsify(self, elem)

   elem = elem or 0

   if self:nDimension() == 1 then
   
      --I assume it is faster to compute the number of 0 to pre-allocate memory --> (benchamrk required)
      local sparseSize = self:size(1)-self:eq(elem):sum()  
      local sparseVector = torch.Tensor(sparseSize, 2) --pre-allocate memory 
      
      local sparseIndex = 1
      for i = 1, self:size(1) do
         if self[i] ~= elem then 
            sparseVector[sparseIndex][1] = i
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

function sparsify(M)
   return torch.Tensor.sparsify(M)
end
