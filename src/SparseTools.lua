

---------------------------------------
-- Basic Constant 

NaN =  1/0 -- -999


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




function torch.Tensor.sortDim(self, dim,inplace)
   
   assert(self:nDimension() == 2)

   if torch.isTensor(self) and self:nDimension() == 2 then

      --TODO: optim -> implement inplace sort with no buffer.    
      local _ , index = self[{{},dim}]:sort()
      local buf = torch.Tensor():resizeAs(self)
      
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


function torch.Tensor.ssortByIndex(self, inplace)
   return torch.Tensor.sortDim(self, 1, inplace)
end

function torch.Tensor.ssort(self, inplace)
   return torch.Tensor.sortDim(self, 2, inplace)
end

function sparsify(M)
   return torch.Tensor.sparsify(M)
end
