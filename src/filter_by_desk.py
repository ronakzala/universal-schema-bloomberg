import os
import concrete.util
from concrete.util import read_communication_from_file as rcff
import shutil

#path = '../data/anyt_sample/'
articlesSourcePath = '/mnt/nfs/work1/mccallum/smysore/data/concretely_annotated_nyt/data/comms' 
filteredArticlesTargetPath = 'filtered/'
#directory = 'test/'
#directory = '/mnt/nfs/work1/mccallum/smysore/data/concretely_annotated_nyt/data/comms/'


#------------ create the target directory -------------------------
if not os.path.exists(target):
    os.makedirs(target)

#------------ create the set of news Desks that should not be considered ----------------------
newsDeskFilePath = 'sections_edited.txt'

setNewsDesk = set()

with open(newsDeskFilePath, 'r') as filehandle:  
	filecontents = filehandle.readlines()
	for line in filecontents:
		desk = line.rstrip()
		setNewsDesk.add(desk)

#------------- create the set of politicians we care about -----------------------------
politiciansFilePath = '../universal-schema-bloomberg/data/congressperson_data/unique_congress.txt'

setPoliticians = set()

with open(politiciansFilePath, 'r') as filehandle:  
	filecontents = filehandle.readlines()
	for line in filecontents:
		count += 1
		setPoliticians.add(line.rstrip())
		print(line.rstrip())
		for name in line.split():
			print(name)
			setPoliticians.add(name)


#-------------- filter articles ---------------------------------------
for filename in os.listdir(path):
	count += 1 #count to track progress
	comm =  rcff(path + filename)

	if comm.communicationMetadata.nitfInfo.newsDesk not in setNewsDesk:
		#print(comm.communicationMetadata.nitfInfo.newsDesk) #print newsDesk of current article
		if 
		shutil.copy(path+'/'+filename, target)

	if(count%500 == 0):
		print(count) #tracking progress