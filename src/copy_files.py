import numpy as np
import os
import shutil

path = '/mnt/nfs/work1/mccallum/smysore/data/concretely_annotated_nyt/data/comms'
target = 'test'

if not os.path.exists(target):
    os.makedirs(target)

files = os.listdir(path)

indexes = np.random.randint(len(files), size=20000)

for i in indexes:
	print(files[i])
	shutil.copy(path+'/'+files[i], target)


