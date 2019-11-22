vocabulary = set(['*START*','*STOP*','*UNKNOWN*'])
labels = set()

def extract_vocabulary(filename):
    """ Extracts training vocabulary. """
    print("reading in vocabulary...")
    with open(filename) as f:
        lines = f.readlines()
        words = [tup.split('_')  for line in lines for tup in line.split()]
        vocabulary.update([token[0] for token in words])
        labels.update([token[1] for token in words])

class Sentence(object):
    def __init__(self, snt):
        self.snt = snt
        features = self.features()
        self.featurelist = features

    def features(self):
        """ Collects basic features from an unannotated sentence, using a window size of 5."""
        position = 0
        sent = self.snt
        if isinstance(sent[0],list):
            sent = [tagged_token[0] for tagged_token in self.snt]
        featurelists = []
        sentlength = len(sent)
        for position in range(sentlength):
            featurelist = ["bias"]
            tagged_token = sent[position]
            if tagged_token not in vocabulary:
                tagged_token = '*UNKNOWN*'
            featurelist.append("word0=" + tagged_token)
            tokenlength = len(tagged_token)
            if tokenlength > 3 and tagged_token !='*UNKNOWN*':
                prefix = tagged_token[:3]
                featurelist.append("prefix=" + prefix)
                if tokenlength > 4 and tagged_token !='*UNKNOWN*':
                    suffix = tagged_token[-3:]
                    featurelist.append("suffix=" + suffix)
            if position==0:
                previous_token = "*START*"
                featurelist.append("word_1=" + previous_token)
            elif position==1:
                previous_token = sent[position -1]
                if previous_token not in vocabulary:
                    previous_token = '*UNKNOWN*'
                previous_2_token = "*START*"
                featurelist.append("word_1=" + previous_token)
                featurelist.append("word_2=" + previous_2_token)
            else:
                previous_token = sent[position -1]
                if previous_token not in vocabulary:
                    previous_token = '*UNKNOWN*'
                previous_2_token = sent[position - 2]
                if previous_2_token not in vocabulary:
                    previous_2_token = '*UNKNOWN*'
                featurelist.append("word_1=" + previous_token)
                featurelist.append("word_2=" + previous_2_token)
            if position== (sentlength -1):
                next_token = "*STOP*"
                featurelist.append("word1=" + next_token)
            elif position== (sentlength - 2):
                next_token = sent[position + 1]
                if next_token not in vocabulary:
                    next_token = '*UNKNOWN*'
                next_2_token = "*STOP*"
                featurelist.append("word1=" + next_token)
                featurelist.append("word2=" + next_2_token)
            else:
                next_token = sent[position + 1]
                if next_token not in vocabulary:
                    next_token = '*UNKNOWN*'
                featurelist.append("word1=" + next_token)
                next_2_token = sent[position + 2]
                if next_2_token not in vocabulary:
                    next_2_token = '*UNKNOWN*'
                featurelist.append("word2=" + next_2_token)
            position +=1
            featurelists.append(featurelist)
        return featurelists
