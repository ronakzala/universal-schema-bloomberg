from concrete.util import read_communication_from_file as rcff
import os
import concrete.util
import shutil


politiciansFilePath = '../universal-schema-bloomberg/data/congressperson_data/unique_congress.txt' #path of file which has the names of politicians

setPoliticians = set()

with open(politiciansFilePath, 'r') as filehandle:  #creating the set of politicians we care about from the corresponding file
	filecontents = filehandle.readlines()
	for line in filecontents:
		count += 1
		setPoliticians.add(line.rstrip())
		print(line.rstrip())
		for name in line.split():
			print(name)
			setPoliticians.add(name)


path = '../data/anyt_sample/AnnotatedNYT-1001191.comm' #TODO : replace by path of directory and add loop to iterate over all the files.


comm =  rcff(path)

filteredDictionary = {} #final dictionary - key. : politician name, value : list of filenames

count = 0
for (uuid,tokenizationObject) in comm.tokenizationForUUID.items(): #iterating through tokenization objects for one comm file
	
	print("Count : ", count+1)
	
	#picking up the NER tagged token list
	taggedTokenListObject = None		#
	for tto in (tokenizationObject.tokenTaggingList):
		if(tto.taggingType == 'NER'):
			taggedTokenListObject = tto.taggedTokenList
			break

	for idx in range(len(tokenizationObject.tokenList)): #iterating throught the token list to find all the 'PERSON' tags

		tokenObject = tokenizationObject.tokenList[idx]		#token object
		taggedTokenObject = taggedTokenListObject[tokenObject.tokenIndex] 		#tagged token Object from the NER tagged token List

		if(taggedTokenObject.tag == 'PERSON'):
			
			person = ''
			while(taggedTokenObject.tag == 'PERSON'):	#if the cuurent token was a 'PERSON' add the text to person variable, continue doing this for all continuous 'PERSON' tags
				person += tokenObject.text
				idx = idx+1
				tokenObject = tokenizationObject.tokenList[idx]
				taggedTokenObject = taggedTokenListObject[tokenObject.tokenIndex]


			if(person in setPoliticians): 	#check if this person is in the politician set, add to final dictionary if true
				if(person in filteredDictionary):
					filteredDictionary[person].append(filename) 
				else:
					tempList = []
					tempList[0] = filename
					filteredDictionary[person] = tempList 

	count = count + 1


	#TODO : store the dictionary
	#Combine this with filter by desk ???
