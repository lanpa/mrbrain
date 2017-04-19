require 'image'

imshow = function (img_in)
  if img_in:max()>1 then print('range > 1!, auto normalize') end
  if img_in:max()<1 then print('auto normalize disabled by dexter') end
  local img = img_in:clone()
  s = img:storage()
  s[1] = 1
  itorch.image(img)
end
imread = image.load
torch.setdefaulttensortype('torch.FloatTensor')

function caffe2torch(net)
--caffe expects [0,255] range in BGR format
--itorch load the image in the range of [0,1] in RGB
--remember to substract the image mean by yourself!!
  tmp = net.modules[1].weight[{{},{1},{},{}}]:clone()
  net.modules[1].weight[{{},{1},{},{}}] = net.modules[1].weight[{{},{3},{},{}}]:clone()
  net.modules[1].weight[{{},{3},{},{}}] = tmp
  net.modules[1].weight:mul(255)
  return net
end
