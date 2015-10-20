require("nn")

local SparseLinearBatch, parent = torch.class('nn.SparseLinearBatch', 'nn.Module')

function SparseLinearBatch:__init(inputSize, outputSize, useGradInput)
   parent.__init(self)

   self.layer = nn.SparseLinear(inputSize, outputSize)

   self.weight = self.layer.weight
   self.bias = self.layer.bias
   
   self.gradWeight = self.layer.gradWeight
   self.gradBias = self.layer.gradBias
   
   self.gradInput  = self.layer.gradInput
   
   self.useGradInput = useGradInput
   
end


function SparseLinearBatch:reset(stdv)
   self.layer:reset(stdv)
   return self
end



function SparseLinearBatch:updateOutput(input)

   if torch.isTensor(input) then

      self.layer:updateOutput(input)
      self.output = self.layer.output

   elseif type(input) == "table" then

      local nframe = #input

      self.outputBuf = self.outputBuf or self.output.new()
      self.outputBuf:resize(nframe, self.bias:size(1))

      local k = 0
      for _, oneInput in pairs(input) do
         k = k + 1 -- when input is not a contiguous table
         self.outputBuf[k] = self.layer:updateOutput(oneInput)
      end

      self.output = self.outputBuf

   else
      error('input must be a either a 2D matrices or a a table of 2D matrices')
   end

   return self.output
end



--TODO implement in C
local function AccumulateSparse(input, buffer, tmp)

            if buffer:nDimension() == 0 then
               buffer:resizeAs(input):copy(input)
            else
               tmp:resize(input:size(1) + buffer:size(1),2):zero()
               
               local i, j, k = 1, 1, 1
               
               while i <=  input:size(1) and j <= buffer:size(1) do
                  
                  local inp = input[i]
                  local buf = buffer[j]
                  
                  if inp[1] == buf[1] then
                     tmp[k] = torch.Tensor{ inp[1], inp[2] + buf[2] }
                     i = i + 1
                     j = j + 1
                  elseif inp[1] < buf[1] then
                     tmp[k]:copy(inp)
                     i = i + 1
                  else
                     tmp[k]:copy(buf)
                     j = j + 1
                  end
                  k = k + 1
                   
               end
               
               if i <=  input:size(1) then
                  local offset = input:size(1) - i
                  tmp[{{k,  k + offset}, {}}] = input[{{i, input:size(1)}, {}}]
                  k = k + offset
               
               elseif  j <= buffer:size(1) then
                  local offset = buffer:size(1) - j
                  tmp[{{k, k + offset}, {}}] = buffer[{{j, buffer:size(1)}, {}}]
                  k = k + offset
               else
                  k = k - 1
               end
               
               buffer:resize(k, 2):copy(tmp[{{1,k},{}}])
            end
            
     return buffer   
end


-- updateGradInput both compute updateGradInput and accGradParameters to simulate batch
function SparseLinearBatch:updateGradInput(input, gradOutput)


   if self.useGradInput == true then
     
      if torch.isTensor(input) then 
         self.layer:updateGradInput(input, gradOutput)
   
      else
           -- Bufferize data 
           self.gradInputBuf1 = self.gradInputBuf1 or gradOutput.new()
           self.gradInputBuf2 = self.gradInputBuf2 or gradOutput.new()
    
          -- accumulate gradInput into self.gradInputBuf1 
          local k = 0
          for _, oneInput in pairs(input) do
            k = k + 1
            self.layer:updateGradInput(oneInput, gradOutput[k])
            AccumulateSparse(self.gradInput, self.gradInputBuf1, self.gradInputBuf2)
          end
   
         -- Copy the buffer to keep the reference of gradInput
         self.gradInput:resizeAs(self.gradInputBuf1):copy(self.gradInputBuf1)
         
      end
      
   end

   return self.gradInput

end

-- updateGradInput both compute updateGradInput and accGradParameters to simulate batch
function SparseLinearBatch:accGradParameters(input, gradOutput)

   if torch.isTensor(input) then 
      self.layer:accGradParameters(input, gradOutput)

   elseif type(input) == "table" then

      -- initialize memory without parasatizing non-batch method
      self.gradWeightBuf = self.gradWeightBuf or torch.zeros(self.layer.gradWeight:size())
      self.gradBiasBuf   = self.gradBiasBuf   or torch.zeros(self.layer.gradBias:size())

      local k = 0
      for _, oneInput in pairs(input) do
         k = k + 1
         
         self.layer:zeroGradParameters()
         
         -- computing gradient
         self.layer:accGradParameters(oneInput, gradOutput[k])
         
         -- accumulating gradient
         self.gradWeightBuf:add(self.layer.gradWeight)
         self.gradBiasBuf:add(self.layer.gradBias) 
      end

      -- Copy the buffer to keep the reference to gradWeight/gradBias unmodified
      self.gradWeight:copy(self.gradWeightBuf)
      self.gradBias:copy(self.gradBiasBuf)

   else
      error('input must be a either a 2D matrices or a table of 2D matrices')
   end

end


function SparseLinearBatch:zeroGradParameters()
   self.layer:zeroGradParameters()
      
   if self.gradBiasBuf and self.gradBiasBuf then
      self.gradWeightBuf:zero()
      self.gradBiasBuf:zero()
      if self.gradInputBuf1 then
            self.gradInputBuf1:resize(0)
      end

   end

end


function SparseLinearBatch:__tostring__()
   return torch.type(self) ..
      string.format('(%d -> %d)', self.weight:size(2), self.weight:size(1))
end





