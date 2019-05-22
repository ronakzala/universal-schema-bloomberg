freq_mapping={}
with open("rel2count_106.txt", "r") as f:
        for line in f:
                splitted_line=line.split()
                if not len(splitted_line)==2:
                        continue
                object_freq=int(splitted_line[0])
                object_name=splitted_line[1]
                #print (object_name+" , "+ str(object_freq))
                freq_mapping[object_name]=object_freq

target=40
target2=3
f2=open('politicians_filtered_relations_only2_freq_uniq_sorted_filtered_106.txt','w')
with open("politicians_filtered_relations_only2_freq_uniq_sorted_106.txt", "r") as f:
        for line in f:
                splitted_line=line.split('\t')
                #print ("wer: ",line," ",len(splitted_line))
                if (freq_mapping[splitted_line[2][:-1]] <= target and freq_mapping[splitted_line[2][:-1]] >= target2):
                        line2=splitted_line[0]+"\t"+splitted_line[1]+"\t"+splitted_line[2][:-1]+"\n"
                        f2.write(line2)

f2.close()
