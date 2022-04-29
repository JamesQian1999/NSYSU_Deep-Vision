import os
import random

arr = os.listdir('/home/jamesqian/Documents/DeepVision/datasets/segmentation/segmentation_data/images')

# Delete Thumbs.db
data = []
for i in arr:
    if(i != 'Thumbs.db'):
        data.append(i[:-4])

# Shuffle data
random.shuffle(data)

# Split data into train, validation, and test (0.8 : 0.1 : 0.1) 
train = data[           0         : int(len(data)*0.8) ]
val =   data[ int(len(data)*0.8)  : int(len(data)*0.9) ]
test =  data[ int(len(data)*0.9)  :                    ]

# Print count
print('train:',len(train),'\nval:',len(val),'\ntest:',len(test),'\ndata:',len(data))

# Write into file
with open('/home/jamesqian/Documents/DeepVision/datasets/segmentation/segmentation_data/train.txt', 'w') as f:
    for item in train:
        f.write("%s\n" % item)

with open('/home/jamesqian/Documents/DeepVision/datasets/segmentation/segmentation_data/val.txt', 'w') as f:
    for item in val:
        f.write("%s\n" % item)

with open('/home/jamesqian/Documents/DeepVision/datasets/segmentation/segmentation_data/test.txt', 'w') as f:
    for item in test:
        f.write("%s\n" % item)