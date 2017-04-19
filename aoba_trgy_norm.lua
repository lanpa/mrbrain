--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet and CIFAR-10 datasets
--

local M = {}


local d = torch.class('DepthDataSet')
function d:__init(imageInfo, split)
    self.imageInfo = imageInfo[split]
    self.split = split
    self.mean = imageInfo['mean']
    self.std = imageInfo['std']
    --print(self.mean, self.std)
end
local folder = 'fakedata/normalized/widac_aoba_%05d_t1/'
local folder_trgy = 'fakedata/normalized/widac_trgy_%05d_t1/'
--local folder = 'data/raw/idac_aoba_%05d_t1/'
--local folder_trgy = 'data/raw/idac_trgy_%05d_t1/'

--local folder = 'data/mW/mWwidac_aoba_%05d_t1_HMRF/'--mWmCmG
--local folder_trgy = 'data/mW/mWwidac_trgy_%05d_t1_HMRF/'--mWmCmG
local imagename = 'image_%d.jpg'
local startID = 31-- 30
local lastID = 150--150
local step = 8
function d:get(i)
    local input = torch.Tensor(math.floor((lastID-startID+1)/step), 189, 157)
    local c = 0
    for ch = 1, lastID-startID+1, step do
        c = c+1
        input[c] = imread(self.imageInfo[i][1]..imagename:format(ch+startID-1))
    end
--    input = input:transpose(1,3)
return {
    input = input,
    target = self.imageInfo[i][2] 
}
end

function d:size()
    return #self.imageInfo
end

function d:preprocess_fortest()
    return function(img)       
        local batchsize = 10
        local input = torch.Tensor(batchsize, img:size(1), img:size(2), img:size(3)) --input
         for b = 1, batchsize do
            local crop = torch.random(5)
            local flip =  math.random()
            local r = img:size(2)
            local c = img:size(3)
            local dx = torch.random(10)-5
            local dy = torch.random(10)-5
            im = image.translate(img, dx,dy)
            for i = 1, im:size(1) do
                   tmp = image.crop(im[i], crop, crop, c-crop, r-crop)
                   tmp = image.scale(tmp, c, r)
                if flip>0.5 then
                    im[i] = image.hflip(tmp) --nobug here?(assignment)
                else
                    im[i] = tmp
                end
            end
            --collectgarbage()
            input[b] = (im-self.mean):cdiv(self.std)
         end
        --collectgarbage()
        --print(m, #input)
        return input
    end
end


function d:preprocess()
    if self.split =='train' then
    return function(img)       
        local crop = torch.random(10)
        local flip =  math.random()
        local r = img:size(2)
        local c = img:size(3)
        local dx = torch.random(10)-5
        local dy = torch.random(10)-5
--        gn = torch.randn(img:size())*0.1-0.05
        for i = 1, img:size(1) do
               tmp = image.translate(img[i], dx,dy)
               tmp = image.crop(tmp, crop, crop, c-crop, r-crop)
               tmp = image.scale(tmp, c, r)
            if flip>0.5 then
                img[i] = image.hflip(tmp) --nobug here?(assignment)
            else
                img[i] = tmp
            end
        end
        for i = 1, img:size(1) do
--            image.save('debug/output'..i..'.png',img[i]) 
        end
        
--        collectgarbage()
        img = (img-self.mean):cdiv(self.std)
--        img = img+gn
        return img
    end
    else
    
    end
end

local function isvalid(opt, cachePath)
   local imageInfo = torch.load(cachePath)
   if imageInfo.basedir and imageInfo.basedir ~= opt.data then
      return false
   end
   return true
end

function M.create(opt, split)
   local cachePath = paths.concat(opt.gen, opt.dataset .. '.t7')
   if not paths.filep(cachePath) or not isvalid(opt, cachePath) then
      paths.mkdir('gen')
   
    subject_dirs = {}
    ages = {}
    i = 0
    f = io.open('data.txt')
    while true do
        local id, age = f:read("*n", "*n")
        if id==nil then break end
        if age>80 or age<20 then goto continue end
        i = i+1
        table.insert(subject_dirs, folder:format(id))
        table.insert(ages, age)
        ::continue::
    end

    f = io.open('data_tsuru.txt')
    while true do
        local id, age = f:read("*n", "*n")
        if id==nil then break end
        if age>80 or age<20 then goto continue end
        i = i+1
        table.insert(subject_dirs, folder_trgy:format(id))
        table.insert(ages, age)
        ::continue::
    end

    local ntrain = 10
    shuffle = torch.randperm(i)
    train_id = shuffle[{{1, ntrain}}]    
    val_id = shuffle[{{ntrain+1, i}}]
    trainpairs = {}
    valpairs = {}
    local input = torch.Tensor(189, 157)
    local mean = torch.Tensor(math.floor((lastID-startID+1)/step), 189, 157):zero()
    local std = torch.Tensor(math.floor((lastID-startID+1)/step), 189, 157):zero()
    require 'paths'
    for i = 1, train_id:size(1) do
        local subject_id = train_id[i]
        local dir = subject_dirs[subject_id]
        assert(paths.dirp(dir), dir..' not found')
        table.insert(trainpairs, {(subject_dirs[subject_id]), ages[subject_id]})
        local c = 0
        for ch = 1, lastID-startID+1, step do
            input = imread(subject_dirs[subject_id]..imagename:format(ch+startID-1))
            c = c+1
            mean[c] = mean[c]+input:mean()
            std[c] = std[c]+input:std()
        end
    end 
    mean = mean/ntrain
    std = std/ntrain
    --print(mean)
    for i = 1, val_id:size(1) do
        local subject_id = val_id[i]
        local dir = subject_dirs[subject_id]
        assert(paths.dirp(dir), dir..' not found')
        table.insert(valpairs, {(subject_dirs[subject_id]), ages[subject_id]})
    end
    torch.save(cachePath, {
      train = trainpairs,
      val = valpairs,
      mean = mean,
      std = std,
   })
   end
   collectgarbage()
   print('start loading')
   local imageInfo = torch.load(cachePath)
   print('loading (from disk) completed')
   
   dd = DepthDataSet(imageInfo, split)
   return dd
end


return M
