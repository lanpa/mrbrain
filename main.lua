require 'cunn'
require 'dexter'
require 'optim'
require 'image'
require 'cudnn'
require 'xlua'
--display = require 'display'
--display.configure({hostname='localhost', port=8000})
--opts = require 'opts'
--print(opts.parse())
--resnet_model = require 'model/resnet_3d'
--model = resnet_model(opts.parse())
--print(model)

opt_string = [[
    -h,--help                                       print help
    -s,--save               (default "logs")        subdirectory to save logs
    -b,--batchSize          (default 16)            batch size
    -r,--learningRate       (default 0.0001)          learning rate
    --learningRateDecay     (default 1e-4)          learning rate decay
    --weightDecay           (default 0.001)        weight decay
    -m,--momentum           (default 0.9)           mementum
    --dataset               (default "aoba_trgy_norm")        dataset
    --model                 (default "model/AlexNet_BN") model file
    --epoch_step            (default 20)            epoch step
    -g,--gpu_index          (default 0)             GPU index (start from 0)
    --max_epoch             (default 10000)           maximum number of epochs
    --nThreads              (default 4)           number of threads
]]
volcon = false
opt = lapp(opt_string)
paths.mkdir(opt.save)

criterion = nn.MSECriterion():cuda()
dataloader = require 'dataloader'
opt.gen = 'gen'
opt.manualSeed = 689

print(opt)

model = require (opt.model)
local tmpChannel = model.modules[1].nOutputPlane
model:remove(1)
print 'init dataloader'
trainLoader, testLoader = dataloader.create(opt)
for n, sample in trainLoader:run() do
    if n==1 then
        print(#sample.input)
        local batchDim = sample.input:size():totable()
        batchDim[1] = -1
        if volcon then
            table.insert(batchDim, 2, 1)
            model:insert(nn.VolumetricDilatedConvolution(1, tmpChannel, 3,3,3, 2,2,2, 3,3,3, 2,2,2),1)
        else
            model:insert(nn.SpatialConvolution(sample.input:size(2), tmpChannel, 3,3, 3,3, 1,1),1)
        end
        model:insert(nn.View(unpack(batchDim)), 1)
        model:cuda()
        local tmp = model:forward(sample.input:cuda())
        print(tmp:size())
        local ndims = tmp:size():totable() --tmp:size(2)*tmp:size(3)*tmp:size(4
        table.remove(ndims, 1)
        local ndim = torch.prod(torch.DoubleTensor(ndims))
        model:add(nn.View(-1, ndim))
--        model:add(nn.ReLU())
        model:add(nn.Dropout(0.5))
        model:add(nn.Linear(ndim, ndim))
--        model:add(nn.ReLU())
        model:add(nn.Dropout(0.5))
        model:add(nn.Linear(ndim, 1))
        model:cuda()
        break
    end
end
model:reset()
cudnn.fastest = true
cudnn.benchmark = true
--cudnn.convert(model, cudnn)

collectgarbage()

parameters, gradParameters = model:getParameters()

trainLoader, testLoader = dataloader.create(opt)
print 'dataloader ready'

optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,  
}




function train()
    model:training()
    epoch = epoch or 1
    local tic = torch.tic()
    local loss = 0
    local err = 0
    local ntrain = 0
    for n, sample in trainLoader:run() do
        --collectgarbage()
        --xlua.progress(n, trainLoader:size())
        inputs = sample.input:cuda()
        targets = sample.target:cuda()
        local feval = function(x)
            assert(x==parameters)
            gradParameters:zero()
            local outputs = model:forward(inputs)
            local f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            model:backward(inputs, df_do)
            -- gradParameters in model have been updated
            loss = loss+f*targets:size(1)
            err = err+torch.abs(outputs - targets):sum()
            ntrain = ntrain+ targets:size(1)
            return f, gradParameters
        end
        optim.sgd(feval, parameters, optimState)
        
    end
    if epoch % 10 == 0 then
        print('current learning rate:'..optimState.learningRate/(1+optimState.evalCounter*optimState.learningRateDecay))
        print('epoch '..epoch..' train: MAE '..sys.COLORS.cyan..err/ntrain..sys.COLORS.none.. ', total '..ntrain..' images.')
        fd = io.open('logs/train.txt', 'a+')
        fd:write(err/ntrain..'\n')
        fd:close()
    end
--    print(('Train loss MSE (in total): '..'%f'..' \t time: %.2f s'):format(loss , torch.toc(tic)))
    if epoch % 1000 == 0 then
        local filename = 'mriaobanet_'..epoch..'.t7'
        print('==> saving model to '..filename)
        torch.save(filename, model:clearState())
    end
    epoch = epoch+1
end

function test()
   model:evaluate()
   collectgarbage()
   local err = 0
   local res = torch.Tensor(opt.batchSize, 2):zero()
   local ntest = 0
   for n, sample in testLoader:run() do
      xlua.progress(n, testLoader:size())
      inputs = sample.input:cuda()
      targets = sample.target:cuda()
      local outputs = model:forward(inputs)
--      local tmp = outputs:t():float()
--      local med = tmp:median()
--      print(outputs, outputs:mean(), outputs:t():float():median(),  targets)
    --  err = err+ torch.abs(outputs - targets):sum()
      ntest = ntest+ targets:size(1)
      err = err+(outputs:t():float():median()-targets:float()):abs()
      if targets:size(1)==opt.batchSize then
    --     res = torch.cat(res, torch.cat(outputs, targets):float())
      end
   end
   err = err[1][1]
   print('test: mean absolute error '..sys.COLORS.cyan..err/ntest..sys.COLORS.none.. ', total '..ntest..' images.')
   fd = io.open('logs/test.txt', 'a+')
   fd:write(err/ntest..'\n')
   fd:close()
end


print(model)
for i = 1,opt.max_epoch do
    train()
    if i %100==0 then
    test()
    end
    --
end


        
