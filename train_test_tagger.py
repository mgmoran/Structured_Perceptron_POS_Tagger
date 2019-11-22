from perceptron_pos_tagger import Perceptron_POS_Tagger
from data_structures import Sentence, extract_vocabulary


def read_in_gold_data(filename):
    print("reading in gold data...")
    with open(filename) as f:
        lines = f.readlines()
        lines = [[tup.split('_') for tup in line.split()] for line in lines]
        sents = [Sentence(line) for line in lines]
    return sents

def read_in_plain_data(filename):
    print("reading in plain data...")
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        sents = [Sentence(line) for line in lines]
    return sents

def output_auto_data(auto_data,filename='auto_dev.tagged'):
    """ Outputs tagged data into a file, which can be read in by the scorer to calculate p-r-f. """
    print("outputting tagged data into file...")
    with open(filename,'w+') as f:
        for sent in auto_data:
            sent = ['_'.join(token) for token in sent]
            f.write(" ".join(sent) + "\n")

if __name__ == '__main__':
    ### Extracting vocab/featurizing data
    extract_vocabulary("train/ptb_02-21.tagged")
    gold_dev_data = read_in_gold_data("dev/ptb_22.tagged")
    train_data = read_in_gold_data("train/ptb_02-21.tagged")
    test_file = 'test/ptb_23.snt'
    test_data = read_in_plain_data(test_file)
    plain_dev_file = 'dev/ptb_22.snt'
    plain_dev_data = read_in_plain_data(plain_dev_file)

    ### Training tagger
    my_tagger = Perceptron_POS_Tagger()
    my_tagger.train(train_data, 'dev/ptb_22.tagged', plain_dev_data)

    ### Tagging dev and test data
    auto_dev_data = my_tagger.tag(plain_dev_data)
    auto_test_data = my_tagger.tag(test_data)

    ### Outputting readable tagged data
    output_auto_data(auto_dev_data)
    output_auto_data(auto_test_data, 'auto_test.tagged')

    my_tagger.run_scorer('dev/ptb_22.tagged','auto_dev.tagged')
