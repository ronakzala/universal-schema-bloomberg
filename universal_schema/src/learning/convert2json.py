import codecs
import sys
import os
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


def main():
    # /Users/ronakzala/696ds/universal-schema-bloomberg/universal_schema/datasets_proc/freebase/latfeatus
    # project_dir = "/Users/ronakzala/696ds/universal-schema-bloomberg/universal_schema"
    project_dir = os.environ['CUR_PROJ_DIR']
    in_path = project_dir + "/datasets_proc/freebase/latfeatus"
    convert2json(in_path)


if __name__ == '__main__':
    main()