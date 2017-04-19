from PIL import Image
import os
im = Image.new("L",(157, 189))

aoba = 'normalized/widac_aoba_{:05}_t1/'
trgy = 'normalized/widac_trgy_{:05}_t1/'

for i in range(10):
    os.makedirs(aoba.format(i))
    os.makedirs(trgy.format(i))
    for j in range(1, 157):
        im.save(os.path.join(aoba.format(i), 'image_{}.jpg'.format(j)))
        im.save(os.path.join(trgy.format(i), 'image_{}.jpg'.format(j)))
