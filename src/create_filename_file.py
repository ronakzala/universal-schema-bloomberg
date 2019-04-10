import os
import concrete.util
from concrete.util import read_communication_from_file as rcff

#directory = '../data/anyt_sample/'
#directory = '../../data/anyt_sample'
#directory = 'test/'
#directory = '/mnt/nfs/work1/mccallum/smysore/data/concretely_annotated_nyt/data/comms/'
directory = '../data/test5k'

file = open("../data/test5k_filenames.txt","w") 

for filename in os.listdir(directory):

	file.write(filename)
	file.write('\n')
file.close()

