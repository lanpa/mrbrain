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
    --nThreads              (default 3)           number of threads
]]
volcon = false
opt = lapp(opt_string)

criterion = nn.MSECriterion():cuda()
dataloader = require 'dataloader'
opt.gen = 'gen'
opt.manualSeed = 689

print(opt)
trainLoader, testLoader = dataloader.create(opt)
print 'dataloader ready'

optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,  
}



function test(x,y)
   model:evaluate()
   collectgarbage()
   local err = 0
   local res = torch.Tensor(opt.batchSize, 2):zero()
   local ntest = 0
   for n, sample in testLoader:run() do
      xlua.progress(n, testLoader:size())
      inputs = sample.input:cuda()
      xc = x*20
      yc = y*20
      inputs[{{},{},{xc-10, xc+10},{yc-10,yc+10}}] = torch.rand(1)[1]
      targets = sample.target:cuda()
      local outputs = model:forward(inputs)
      ntest = ntest+ targets:size(1)
      err = err+(outputs:t():float():median()-targets:float()):abs()
      if targets:size(1)==opt.batchSize then
    --     res = torch.cat(res, torch.cat(outputs, targets):float())
      end
   end
   err = err[1][1]
   print('test: mean absolute error '..sys.COLORS.cyan..err/ntest..sys.COLORS.none.. ', total '..ntest..' images.')
   return err/ntest
end

model = torch.load('mriaobanet_200.t7')
--cudnn.convert(model, nn)

--model:float()
print(model)

--189 157
accuracy = torch.Tensor(8, 7):zero()
for x = 1,8 do
    for y = 1, 7 do
        print(x,y)
        accuracy[x][y]= test(x,y)
    end
end

--torch.save('occu.t7', accuracy)
accuracy = accuracy-accuracy:min()
image.save('visualize.png', accuracy/accuracy:max())
       
