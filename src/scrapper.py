from random import shuffle
import argparse


def process_freebase_sample(input_file_path):
    input_file = open(input_file_path)
    shuffled_relations_output_file = open(input_file_path+"_shuffled", mode='x')
    unique_entities_output_file = open(input_file_path+"_unique_entities", mode='x')

    entities_set = set()
    entity_tup = []
    relations = []

    for line in input_file:
        tokens = line.split()
        entity1, entity2, relation = tokens[:3]
        entities_set.add(entity1)
        entities_set.add(entity2)
        entity_tup.append((entity1, entity2))
        relations.append(relation)

    # Shuffle the relations order
    shuffle(relations)

    for (e1,e2),r in zip(entity_tup, relations):
        shuffled_relations_output_file.write("{} {} {}\n".format(e1, e2, r))

    for entity in entities_set:
        unique_entities_output_file.write(entity+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--inputfile', help='input freebase data file containing')
    opt = parser.parse_args()

    process_freebase_sample(opt.inputfile)


if __name__ == '__main__':
        main()
