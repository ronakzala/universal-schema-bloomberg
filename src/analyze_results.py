import json
import sys
import os

output_dir = "../output/run_edited_set/"

with open(output_dir + "politicians_filtered_articles_test5k.json","r") as f:
  data = f.read()

filtered_dictionary = json.loads(data)
count_dictionary = {}

for k,v in filtered_dictionary.items():
	count_dictionary[k] = len(v)

with open(output_dir + "analyzed_dictionary.json", 'w') as f:
		json.dump(count_dictionary, f, indent=4)

count_0 = 0
count_10 = 0
count_30 = 0
count_50 = 0
count_100 = 0
count_300 = 0
count_500 = 0
count_over_500 = 0
for k,v in count_dictionary.items():
	if (v == 0):
		count_0 += 1
	else :
		if(v <= 10):
			count_10 += 1
		else:
			if(v <= 30):
				count_30 += 1
			else:
				if(v <= 50):
					count_50 += 1
				else:
					if(v <= 100):
						count_100 += 1
					else:
						if(v <= 300):
							count_300 += 1
						else:
							if(v <= 500):
								count_500 += 1
							else:
								count_over_500 += 1


print("Politicians with :")
print("No articles: %d" % count_0)
print("No more than 10 articles: %d" % count_10)
print("No more than 30 articles: %d" % count_30)
print("No more than 50 articles: %d" % count_50)
print("No more than 100 articles: %d" % count_100)
print("No more than 300 articles: %d" % count_300)
print("No more than 500 articles: %d" % count_500)
print("More than 500 articles: %d" % count_over_500)




