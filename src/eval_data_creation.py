import os
import numpy as np
import h5py
import json

'''
Eval Set construction for each Congress:
	-> Find all congresspeople with less than max number of votes ("limited data")
	-> Select 3 senators from both parties
	-> Select 7 house representatives from both parties
	-> These 20 congresspeople make up the "no data" set
	-> Zero out their data in the vote matrix
'''

eval_info = {}

# These indices won't be used again, just adding here as a reference
# From now on the names will be used directly
dict_indices = {
	106: {
		"senate": list(range(100)),
		"house": list(range(100, 539))
	},
	107: {
		"senate": list(range(434, 534)),
		"house": list(range(434)) + list(range(534, 543))
	},
	108: {
		"senate": list(range(436, 536)), 
		"house": list(range(436)) + list(range(536, 539))
	},
	109: {
		"senate": list(range(437, 537)),
		"house": list(range(437)) + list(range(537, 542))
	}
}

for congress in ["106", "107", "108", "109"]:
	eval_info[congress] = {"limited_data": [], "no_data": []}
	with open("/Users/pallavi/UMass/Semester4/CS696/universal-schema-bloomberg/data/preprocessing_metadata/cp_info_%s.txt" % congress, 'r') as f:
	    cp_list = f.readlines()
	cp_list = [x.strip() for x in cp_list]

	data_file = h5py.File("../data/%s.hdf5" % congress, 'r')

	big_matrix_train_cp = data_file['big_matrix_train_out'][0]
	counts = np.count_nonzero(big_matrix_train_cp, axis=0)
	print(len(counts))

	senate_counts = counts[dict_indices[int(congress)]["senate"]]
	house_counts = counts[dict_indices[int(congress)]["house"]]
	max_senate = max(senate_counts)
	max_house = max(house_counts)
	print(max_senate, max_house)

	list_less_count_indices = []
	for idx in dict_indices[int(congress)]["senate"]:
	    if counts[idx] < max_senate:
	            list_less_count_indices.append(idx)

	for idx in dict_indices[int(congress)]["house"]:
	    if counts[idx] < max_house:
	            list_less_count_indices.append(idx)

	eval_info[congress]["limited_data"] = [cp_list[i] for i in list_less_count_indices]

eval_info["109"]["no_data"] = [
	"LindseyGraham Republican",
	"BarackObama Democrat",
	"JohnRockefeller Democrat",
	"HillaryClinton Democrat",
	"SusanCollins Republican",
	"MitchMcConnell Republican",
	"NancyPelosi Democrat",
	"SolomonOrtiz Democrat",
	"DennisCardoza Democrat",
	"JimCosta Democrat",
	"AdamSchiff Democrat",
	"MajorOwens Democrat",
	"GregoryMeeks Democrat",
	"PeterKing Republican",
	"JohnSweeney Republican",
	"RoyBlunt Republican",
	"EricCantor Republican",
	"FrankLoBiondo Republican",
	"JebBradley Republican",
	"MichaelBilirakis Republican"
]

eval_info["108"]["no_data"] = [
	"LindseyGraham Republican",
	"JosephBiden Democrat",
	"HarryReid Democrat",
	"HillaryClinton Democrat",
	"SusanCollins Republican",
	"MitchMcConnell Republican",
	"NancyPelosi Democrat",
	"AnthonyWeiner Democrat",
	"MichaelHonda Democrat",
	"FrankPallone Democrat",
	"AdamSchiff Democrat",
	"MajorOwens Democrat",
	"GregoryMeeks Democrat",
	"VitoFossella Republican",
	"SherwoodBoehlert Republican",
	"RoyBlunt Republican",
	"EricCantor Republican",
	"FrankLoBiondo Republican",
	"C.Cox Republican",
	"MichaelBilirakis Republican"
]

eval_info["107"]["no_data"] = [
	"JeffersonSessions Republican",
	"JosephBiden Democrat",
	"HarryReid Democrat",
	"HillaryClinton Democrat",
	"SusanCollins Republican",
	"MitchMcConnell Republican",
	"NancyPelosi Democrat",
	"AnthonyWeiner Democrat",
	"MichaelHonda Democrat",
	"FrankPallone Democrat",
	"AdamSchiff Democrat",
	"MajorOwens Democrat",
	"GregoryMeeks Democrat",
	"VitoFossella Republican",
	"SherwoodBoehlert Republican",
	"RoyBlunt Republican",
	"LindseyGraham Republican",
	"FrankLoBiondo Republican",
	"C.Cox Republican",
	"MichaelBilirakis Republican"
]

eval_info["106"]["no_data"] = [
	"JeffersonSessions Republican",
	"JosephBiden Democrat",
	"HarryReid Democrat",
	"CharlesSchumer Democrat",
	"SusanCollins Republican",
	"MitchMcConnell Republican",
	"NancyPelosi Democrat",
	"AnthonyWeiner Democrat",
	"LouiseSlaughter Democrat",
	"FrankPallone Democrat",
	"BarbaraLee Democrat",
	"MajorOwens Democrat",
	"GregoryMeeks Democrat",
	"VitoFossella Republican",
	"SherwoodBoehlert Republican",
	"RoyBlunt Republican",
	"LindseyGraham Republican",
	"FrankLoBiondo Republican",
	"C.Cox Republican",
	"MichaelBilirakis Republican"
]


with open("../data/preprocessing_metadata/eval_info.json", 'w') as out_f:
	json.dump(eval_info, out_f, indent=4)

