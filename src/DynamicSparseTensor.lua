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