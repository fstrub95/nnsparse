local SparseCriterion, parent = torch.class('nn.SparseCriterion', 'nn.Criterion')


function SparseCriterion:__init(criterion)
   parent.__init(self)

   self.criterion = criterion
end


function SparseCriterion:prepareInput(input)
   
   self.prepareInputBuf = self.prepareInputBuf or torch.Tensor()
      
   self.prepareInputBuf:resizeAs(input):copy(input)
   self.prepareInputBuf[{{},2}] = self.criterion:prepareInput(input[{{},2}])

   return self.prepareInputBuf
end



--the target is sparse
function SparseCriterion:updateOutput(estimate, target)

   if torch.isTensor(target) then
      -- create a vector that only contains the "useful target"
      local index = target[{{},1}]
      local data  = target[{{},2}]
      
      local i = 0
      self.estimateBuf = self.estimateBuf or torch.Tensor()
      self.estimateBuf:resizeAs(index)
      self.estimateBuf:apply(function()
         i = i + 1
         return estimate[index[i]] 
      end
      )
      
      -- compute the loss on a dense vector
      return self.criterion:updateOutput(self.estimateBuf, data)       
   else
      -- iterate over each sparse vector and accumulate the loss
      local loss = 0
      local totalSize = 0
      for k, t in pairs(target) do
         local size = t:size(1)
         loss = loss + self:updateOutput(estimate[k], t)*size
         totalSize = totalSize + size
      end
      
      return loss/totalSize
   end
end




function SparseCriterion:updateGradInput(estimate, target)
   
   if torch.isTensor(target) then
      -- create a vector that only contains the "useful targets"
      local index = target[{{},1}]
      local data  = target[{{},2}]
      
      self.targetBuf = self.targetBuf or torch.Tensor()
      self.targetBuf:resizeAs(estimate):copy(estimate)
      self.targetBuf:indexCopy(1, index:long(), data)

      return self.criterion:updateGradInput(estimate, self.targetBuf, index)
   else
   
      self.dloss = self.dloss or estimate[1].new()
      self.dloss:resizeAs(estimate):zero()
      
      -- iterate over each sparse vector and accumulate the dloss
      for k, t in pairs(target) do
         self.dloss[k] = self:updateGradInput(estimate[k], t)
      end
      return self.dloss
      
   end

end
