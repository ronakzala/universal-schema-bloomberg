import codecs
import sys
import os, argparse
import json
import time


def convert2json(in_path):

    start_time = time.time()
    splits = ['dev', 'test', 'train']
    for split in splits:
        file = open(in_path + "/" + split + ".txt")
        pos_split_fname = os.path.join(in_path, split) + '.json'
        pos_split_file = codecs.open(pos_split_fname, 'w', 'utf-8')
        pos_doc_id = 0
        for line in file:
            line = line.split()
            neg_col = line[0:2]
            pos_row = [line[2]]
            im_data = {
                'row': pos_row,
                'col': neg_col,
                'doc_id': pos_doc_id
            }
            pos_doc_id += 1
            jsons = json.dumps(im_data)
            pos_split_file.write(jsons + '\n')
            if pos_doc_id % 10000 == 0:
                sys.stdout.write('Processing example: {:d}\n'.format(pos_doc_id))
        pos_split_file.close()
        sys.stdout.write('Wrote: {:s}\n'.format(pos_split_file.name))
        sys.stdout.write('Took: {:4.4f}s\n\n'.format(time.time() - start_time))


def convert_split(in_path, entity_id_filename, relationship_id_filename, train_split_size, dev_split_size, number_of_lines):
    entity_id_file = open(in_path + "/" + entity_id_filename)
    relationship_id_file = open(in_path + "/" + relationship_id_filename)
    train_file = open(in_path + "/train.txt", 'w')
    dev_file = open(in_path + "/dev.txt", 'w')
    test_file = open(in_path + "/test.txt", 'w')

    train_lines = int((train_split_size/100)*number_of_lines)
    dev_lines = int(((train_split_size + dev_split_size) / 100) * number_of_lines)

    id2ent = {}

    sys.stdout.write('Starting storing id to entity map\n')
    for line_no, line in enumerate(entity_id_file):
        line = line.split()
        if line_no % 10000 == 0:
            sys.stdout.write('Processing example: {:d}\n'.format(line_no))
        if line[0] not in id2ent:
            id2ent[line[0]] = line[1]

    sys.stdout.write('Starting creating dev,train, and test set\n')
    for line_no, line in enumerate(relationship_id_file):
        line = line.split()
        line_data = "{} {} {}\n".format(id2ent[line[0]], id2ent[line[1]], line[2])
        if line_no % 10000 == 0:
            sys.stdout.write('Processing example: {:d}\n'.format(line_no))
        if line_no < train_lines:
            train_file.write(line_data)
        elif line_no < dev_lines:
            dev_file.write(line_data)
        else:
            test_file.write(line_data)
    train_file.close()
    dev_file.close()
    test_file.close()


def main():
    # /Users/ronakzala/696ds/universal-schema-bloomberg/universal_schema/datasets_proc/freebase/latfeatus
    # project_dir = "/Users/ronakzala/696ds/universal-schema-bloomberg/universal_schema"
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--entity_id_file', required=True,
                              help='Name of the file containing the [id entity] entity to id mapping.')
    parser.add_argument('-r', '--relationship_id_file', required=True,
                        help='Name of the file containing the [entity1 enitity 2 relationship] relationship to id mapping.')
    parser.add_argument('-t', '--train_split_size', required=True,
                        help='Size of the training set to be made from given set of relations(in percentage)')
    parser.add_argument('-d', '--dev_split_size', required=True,
                        help='Size of the dev set to be made from given set of relations(in percentage)')
    parser.add_argument('-n', '--num_of_lines', required=True,
                        help='Number of lines of relationship_id file')
    cl_args = parser.parse_args()

    project_dir = os.environ['CUR_PROJ_DIR']
    in_path = project_dir + "/datasets_proc/freebase/latfeatus"

    convert_split(in_path, cl_args.entity_id_file, cl_args.relationship_id_file, int(cl_args.train_split_size),
                  int(cl_args.dev_split_size), int(cl_args.num_of_lines))
    convert2json(in_path)


if __name__ == '__main__':
    main()