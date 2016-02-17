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