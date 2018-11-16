import json
import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

train_files = ["bytecup.corpus.train.0.txt", "bytecup.corpus.train.1.txt", "bytecup.corpus.train.2.txt", "bytecup.corpus.train.3.txt"]
val_files = ["bytecup.corpus.train.8.txt"]
test_files = ["bytecup.corpus.validation_set.txt"]

finished_files_dir = "finished_files"
raw_files_dir = "data/raw"
split_files_dir = "data/split"
tokenized_files_dir = "data/tokenized"

chunks_dir = os.path.join(finished_files_dir, "chunked")

VOCAB_SIZE = 200000
CHUNK_SIZE = 2000 # num examples per chunk, for the chunked data


def chunk_file(set_name):
    in_file = 'finished_files/%s.bin' % set_name
    reader = open(in_file, "rb")
    chunk = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            chunk += 1


def chunk_all():
    # Make a dir to hold the chunks
    if not os.path.isdir(chunks_dir):
        os.mkdir(chunks_dir)
    # Chunk the data
    for set_name in ['train', 'val', 'test']:
        print("Splitting %s data into chunks..." % set_name)
        chunk_file(set_name)
    print("Saved chunked data in %s" % chunks_dir)


def tokenize_stories(stories_dir, tokenized_stories_dir):
    """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
    print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
    stories = os.listdir(stories_dir)
    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
        for s in stories:
            f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def get_TCs(story_file):
    # Lowercase everything
    TCs = [["", ""]]
    with open(story_file) as f:
        for line in f:
            line = line.strip().lower();
            if line == "": continue # empty line
            if line.startswith("ychzhou"):
                TCs[-1][0] = "%s %s %s" % (SENTENCE_START, line.replace("ychzhou", "").strip(), SENTENCE_END)
                TCs.append(["", ""])
            else:
                if line[-1] not in END_TOKENS:
                    line += ' .'
                TCs[-1][1] += ' linesep ' + line
    return TCs[: -1]


def write_to_bin(in_files, out_file, makevocab=False):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""

    if makevocab:
        vocab_counter = collections.Counter()

    with open(out_file, 'wb') as writer:
        for fname in in_files:
            print("Making bin file for URLs listed in %s..." % fname)
            TCs = get_TCs(tokenized_files_dir + "/" + fname)
            num_stories = len(TCs)

            for idx, (title, content) in enumerate(TCs):
                if idx % 10000 == 0:
                    print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))
                # Write to tf.Example
                if makevocab and len(content) > 2000:
                    continue
                tf_example = example_pb2.Example()
                tf_example.features.feature['article'].bytes_list.value.extend([content.encode()])
                tf_example.features.feature['abstract'].bytes_list.value.extend([title.encode()])
                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))

                # Write the vocab to file, if applicable
                if makevocab:
                    art_tokens = content.split(' ')
                    abs_tokens = title.split(' ')
                    abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
                    tokens = art_tokens + abs_tokens
                    tokens = [t.strip() for t in tokens] # strip
                    tokens = [t for t in tokens if t!=""] # remove empty
                    vocab_counter.update(tokens)

    print("Finished writing file %s\n" % out_file)

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")

if __name__ == '__main__':
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)
    if not os.path.exists(raw_files_dir): os.makedirs(raw_files_dir)
    if not os.path.exists(split_files_dir): os.makedirs(split_files_dir)
    if not os.path.exists(tokenized_files_dir): os.makedirs(tokenized_files_dir)

    for fname in os.listdir(raw_files_dir):
        with open("%s/%s" % (raw_files_dir, fname)) as f, open("%s/%s" % (split_files_dir, fname), 'w') as fc:
            for cnt, line in enumerate(f):
                obj = json.loads(line)
                print(obj['content'], file=fc)
                print("ychzhou " + obj.get('title', ""), file=fc)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
    tokenize_stories(split_files_dir, tokenized_files_dir)

    # Read the tokenized stories, do a little postprocessing then write to bin files
    #write_to_bin(test_files, os.path.join(finished_files_dir, "test.bin"))
    write_to_bin(val_files, os.path.join(finished_files_dir, "val.bin"))
    #write_to_bin(train_files, os.path.join(finished_files_dir, "train.bin"), makevocab=True)

    # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
    chunk_all()
