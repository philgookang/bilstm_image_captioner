
def load_dataset():

    # ########################
    # DATASET PREPROCESS
    # ########################

    corpus = []
    word_to_idx = { }
    idx_to_word = { }
    train_dataset = [] # { "data" : ['<s>', 'john'], "target" : "a" }
    i = 0

    # dataset
    dataset = [
        ['<s>', "john", "went", "to", "the", "store", "<e>"],
        ['<s>', "john", "went", "to", "the", "mall", "<e>"],
        ['<s>', "john", "went", "home", "early", "<e>"],
        ['<s>', "john", "is", "a", "engineer", "<e>"],
        ['<s>', "john", "is", "at", "home", "<e>"],
        ['<s>', "john", "can", "run", "fast", "<e>"],
        ['<s>', "john", "can", "go", "rest", "<e>"],
        ['<s>', "john", "can", "fly", "<e>"],
        ['<s>', "john", "can", "fly", "high", "<e>"],
        ['<s>', "john", "is", "not", "a", "data", "<e>"],
        ['<s>', "john", "is", "only", "at", "home", "<e>"],
        ['<s>', "john", "will", "jump", "<e>"],
        ['<s>', "john", "will", "jump", "high", "<e>"],
        ['<s>', "john", "will", "jump", "fast", "<e>"],
        ['<s>', "john", "is", "faster", "than", "me", "<e>"],
        ['<s>', "john", "is", "faster", "than", "a", "dog", "<e>"],
        ['<s>', "john", "is", "really", "fast", "<e>"],
        ['<s>', "john", "is", "only", "dead", "<e>"],
        ['<s>', "john", "is", "in", "love", "<e>"],
        ['<s>', "john", "is", "fly", "tomorrow", "<e>"],
        ['<s>', "john", "is", "fly", "high", "now", "<e>"]
    ]

    # generate corpus & naive label encoding! (give each word a id #)
    for words in dataset:
        for w in words:
            if w not in corpus:
                corpus.append(w)
                word_to_idx[w] = i
                idx_to_word[i] = w
                i += 1

    # change dataset format to many-to-one format
    for sentence in dataset:
        tmp = []
        for i in range(len(sentence)-1):
            tmp.append(sentence[i]) # we need to cumulate the sentence
            target = sentence[i + 1] # set the next word in the sentence as the target!
            train_dataset.append({"data" : tmp.copy(), "target" : target})

    return corpus, word_to_idx, idx_to_word, train_dataset
