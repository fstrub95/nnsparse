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
