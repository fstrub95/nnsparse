function nn.Criterion:prepareInput(input)
   return input
end


local SDAECriterion, parent = torch.class('nnsparse.SDAECriterion', 'nn.Criterion')

function SDAECriterion:__init(criterion, SDAEconf)
   parent.__init(self)

   self.criterion = criterion

   self.alpha = SDAEconf.alpha or 1   
   self.beta  = SDAEconf.beta  or 0

   self.noiseRatio = SDAEconf.noiseRatio or 0
   self.noiseMean = SDAEconf.noiseMean or 0
   self.noiseStd = SDAEconf.noiseStd or 0.2

   self.flipRatio = SDAEconf.flipRatio or 0
   self.flipRange = SDAEconf.flipRange


   self.hideRatio = SDAEconf.hideRatio or 0

   self.sizeAverage = true
   
   self.criterion.sizeAverage = false
end


function SDAECriterion:prepareInput(input)

   self.input = self.input or input.new()
   self.input:resizeAs(input):copy(input)

   self.maskAlpha    = self.maskAlpha or torch.Tensor():byte()
   self.maskBeta     = self.maskBeta  or torch.Tensor():byte()
      
   self.maskAlpha:resize(input:size())
   self.maskBeta:resize(input:size())


   local vInput     = self.input:view(-1)
   local vMaskAlpha = self.maskAlpha:view(-1)
   local vMaskBeta  = self.maskBeta:view(-1)


   for i = 1, vInput:nElement() do

         vMaskAlpha[i] = 1
         vMaskBeta[i]  = 0

         local r = torch.uniform()
         if      r < self.noiseRatio                                   then vInput[i] = vInput[i] + torch.normal(self.noiseMean, self.noiseStd)  -- add gaussian noise
         elseif  r < self.noiseRatio + self.flipRatio                  then vInput[i] = self.flipRange[torch.uniform() > 0.5 and 1 or 2] -- either return min/max
         elseif  r < self.noiseRatio + self.flipRatio + self.hideRatio then vInput[i] = 0                                                -- remove data
         else
            vMaskAlpha[i] = 0
            vMaskBeta[i]  = 1
         end
   end
   
   return self.input

end



function SDAECriterion:updateOutput(estimate, target)

      local loss = 0
      local totalSize = 0

      local _estimate = estimate[self.maskAlpha]
      if _estimate:nDimension() > 0 then 
         loss = loss + self.alpha * self.criterion:updateOutput(_estimate, target[self.maskAlpha])
      end 
      
      local _estimate = estimate[self.maskBeta]
      if _estimate:nDimension() > 0 then 
         loss = loss + self.beta  *  self.criterion:updateOutput(_estimate , target[self.maskBeta])
      end
      
      if self.sizeAverage == true then
         loss = loss/estimate:nElement()
      end
      
      return loss
end



function SDAECriterion:updateGradInput(estimate, target, index)

   local dloss = self.criterion:updateGradInput(estimate , target)

   if index == nil then -- not-sparse

        dloss[self.maskAlpha] = dloss[self.maskAlpha]* self.alpha
        dloss[self.maskBeta ] = dloss[self.maskBeta ]* self.beta


      --crash on some very rare case with luajit...

--      local viewAlpha = self.maskAlpha
--      if viewAlpha:nDimension() > 1 then
--         viewAlpha = viewAlpha:view(-1)
--      end
--
--      local i = 0
--      dloss:apply(function(x)
--         i = i +1
--         if viewAlpha[i] == 1 then
--            return self.alpha * x 
--         else
--            return self.beta  * x
--         end
--      end
--      )

   else --sparse

      local i = 1
      local k = 1

      dloss:apply(function(x)


            if k <= index:size(1) and index[k] == i then 
               if self.maskAlpha[k] == 1 then
                  x = x * self.alpha 
               else
                  x = x * self.beta
               end
               k = k + 1
            end  
            i = i + 1

            return x

      end
      )
   end

   if self.sizeAverage == true then
       dloss:div(estimate:nElement())
   end

   return dloss

end

function SDAECriterion:__tostring__()
   return torch.type(self) .. " alpha: ".. self.alpha .. ", beta: " ..self.beta .. ",noiseRatio: " .. self.noiseRatio .. " ,flipRatio: " .. self.flipRatio .. ", hideratio:" .. self.hideRatio
end



