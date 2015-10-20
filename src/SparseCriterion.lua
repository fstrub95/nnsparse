local SparseCriterion, parent = torch.class('nn.SparseCriterion', 'nn.Criterion')


function SparseCriterion:__init(criterion)
   parent.__init(self)

   self.criterion = criterion
   self.criterion.sizeAverage = false
   self.sizeAverage = true
end


function SparseCriterion:updateOutput(estimate, target)

      --compute recursively the loss
      local loss, nElem = self:__updateOutput(estimate, target)
    
      if self.sizeAverage == true then
         loss = loss/nElem
      end
     
     return loss
end

--the target is sparse
function SparseCriterion:__updateOutput(estimate, target)

   if torch.isTensor(target) then
      -- create a vector that only contains the "useful target"
      local index = target[{{},1}]
      local data  = target[{{},2}]
      
      local i = 0
      self.estimateBuf = self.estimateBuf or estimate.new()
      self.estimateBuf:resizeAs(index)
      self.estimateBuf:apply(function()
         i = i + 1
         return estimate[index[i]] 
      end
      )
      
      -- compute the loss on a dense vector
      return self.criterion:updateOutput(self.estimateBuf, data), target:size(1)
   else
      -- iterate over each sparse vector and accumulate the loss
      local loss = 0
      local noElem = 0
      for k, t in pairs(target) do
         loss = loss + self:__updateOutput(estimate[k], t)
         noElem = noElem + t:size(1)
      end
      
      return loss, noElem
   end
end

function SparseCriterion:updateGradInput(estimate, target)
      
      local nElem = 0
      
      --compute recursively dloss
      self.dloss, nElem = self:__updateGradInput(estimate, target)
       
      if self.sizeAverage == true then
         self.dloss:div(nElem)
      end
     
     return self.dloss
end   


function SparseCriterion:__updateGradInput(estimate, target)
   

   
   if torch.isTensor(target) then
      -- create a vector that only contains the "useful targets"
      local index = target[{{},1}]
      local data  = target[{{},2}]
      
      self.targetBuf = self.targetBuf or estimate.new()
      self.targetBuf:resizeAs(estimate):copy(estimate)
      self.targetBuf:indexCopy(1, index:long(), data)

      return self.criterion:updateGradInput(estimate, self.targetBuf, index), target:size(1)
   else
   
      self.dloss = self.dloss or estimate[1].new()
      self.dloss:resizeAs(estimate):zero()
      
      -- iterate over each sparse vector and accumulate the dloss
      local nElem = 0
      for k, t in pairs(target) do
         self.dloss[k] = self:__updateGradInput(estimate[k], t)
         nElem = nElem + t:size(1)
      end
      
      return self.dloss, nElem
   end
   
end
