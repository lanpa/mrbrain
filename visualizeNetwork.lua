require 'cunn'
require 'cudnn'
image = require 'image'

function dumptensor(t, layer)
    if t:nDimension()==4 then t = t[1] end
    b = 2
    n = 12
    ww = t:size(2)
    hh = t:size(3)
    w = math.ceil((n+1)*(ww+b))
    h = math.ceil((n+1)*(hh+b))
    print(w,h)
    res = torch.Tensor(w,h):zero()


    for i=1, n do
        shiftw = i*(ww+b)
        for j = 1,n do
            if (i-1)*n+j>t:size(1) then break end
            shifth = j*(hh+b)
            res[{{1+shiftw,ww+shiftw},{1+shifth,hh+shifth}}] = normalize(t[(i-1)*n+j]:double())
        end
    end
    image.save('debug/output_layer'..layer..'.png', (res))
    
end


model_short = nn.Sequential()
function visualize(net)
    net = torch.load('bestmodel.t7')
    --net = torch.load('/mnt/data/dexter/vgg16/vgg16.t7')
    
    for i,v in ipairs(net.modules) do 
        if torch.typename(v) == 'nn.SpatialConvolution' then
            --v:reset()
           break
        end
    end
    --[[
    cnt=1
    for i,v in ipairs(net.modules) do
        model_short:add(v:clone())
        --print(v)
        if torch.typename(v) == 'cudnn.SpatialConvolution'  then
            cnt = cnt+1
        end
        if cnt==3 then 
            model_short:add(nn.ReLU())
            break
        end
    end
    model_short:cuda()
    print(model_short)
    --]]
    --input = torch.rand(1,15,189,157)
    --a = model_short:forward(input:cuda())
end

function normalize(img)
img = img-img:mean()
img = img/(img:std()+0.0001)
img = img*0.1

img = img+0.5
img[img:gt(1)] =1
img[img:lt(0)] =0 
 
--M = img:max()
--m = img:min()
return img
end

function denormalize(img)
M = img:max()
m = img:min()
return (img-m)/(M-m)
end

--visualize()
require 'optim'
--model = model_short
model = torch.load('mriaobanet_10000.t7')
model:cuda()
criterion = nn.MSECriterion():cuda()
function train()
    model:training()
    input = torch.rand(1,15,189,157)
    
    local kk = 31
    for s = 1, 15 do 
        input[1][s] = image.load('data/normalized/widac_aoba_00448_t1/image_'..kk..'.jpg')
        kk = kk+8
    end
    input = (input:cuda()-0.5)*2
    --input = torch.rand(1,15,189,157):cuda()*2-1
--        input = torch.rand(1,3,128,128):cuda()*2-1
    
    --for i=1,2000  do
    local outputs = model:forward(input)
    for layer = 3, 30 do
    print(model.modules[layer].output:size())
        dumptensor(model.modules[layer].output, layer)
    end
end

train()
