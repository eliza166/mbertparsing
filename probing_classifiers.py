import collections
import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
from bert import tokenization as tokenization_d
from bert_english import tokenization as tokenization_o
from Language_Analysis import load_pickle
import pickle


def clean_data():
    """embed_tweets from http://spinningbytes.ch/resources/wordembeddings has
    some line with inappropiate format, this code create a new file
    without this lines.
    """
    with open("./data/glove/embed_tweets_multi_300M_52D") as f, \
        open("./data/glove/embed_tweets_multi_300M_52D_clean","w+") as sample:
        for i, line in enumerate(f):
            data = line.strip().split(" ")
            if len(data) == 53:
                sample.write(line)
            if i%10000 == 0:
                print(i)

def search_badlines():
    """embed_tweets from http://spinningbytes.ch/resources/wordembeddings has
    some line with inappropiate format, this code find and display information
    about this lines and the numbers of occurrences of each one.
    """
    from collections import defaultdict
    c = 0
    datos = defaultdict(int)
    with open("./data/glove/embed_tweets_multi_300M_52D") as f:  
        for i, line in enumerate(f):
            data = line.strip().split(" ")
            if len(data) != 53:
                datos[(data[0],len(data[1:]))] += 1
                c += 1
                if c%100 == 0:
                    print(c,i)
    print(datos)

def create_vocab_embeddings_pkl():
    datos = {}
    with open("./data/word2vec/embed_tweets_multi_400m_52D") as f:
        for li,line in enumerate(f):
            data = line.strip().split(" ")
            datos[data[0]] = [float(i) for i in data[1:]]
            if li%10000==0:
                print(li)
    n = len(datos)
    k_0 = next(iter(datos))
    embeddings = np.empty([n,len(datos[k_0])])
    for i,(k,v) in enumerate(datos.items()):
        embeddings[i,:] = v
        datos[k] = i
    
    pickle.dump( datos, open( "./data/word2vec/vocab_embed_tweets_multi_400m_52D.pkl", "wb" ))
    pickle.dump( embeddings, open( "./data/word2vec/embeddings_embed_tweets_multi_400m_52D.pkl", "wb" ))


class WordEmbeddings(object):
    """Class for loading/using pretrained GloVe embeddings"""

    def __init__(self,dataset="default"):
        if dataset == "word2vec_tweets_400m_52":
            self.pretrained_embeddings = np.float32(load_pickle("./data/word2vec/embeddings_embed_tweets_multi_400m_52D.pkl"))
            self.vocab = load_pickle("./data/word2vec/vocab_embed_tweets_multi_400m_52D.pkl")
        else:
            #type(pre_embeddings) numpy.ndarray
            #pre_embeddings.shape (400001, 100)
            self.pretrained_embeddings = load_pickle("./data/glove/embeddings.pkl")
            #type(vocab) == dict
            # type(vocab[k]) == int (word embedding)
            # len(vocab.keys()) 400001
            self.vocab = load_pickle("./data/glove/vocab.pkl")
        print("pre_embeddings", self.pretrained_embeddings.shape)
        print("vocab", len(self.vocab.keys()))
        print("vocab", list(self.vocab.keys())[:100])
        print("vocab", [self.vocab[k] for k in list(self.vocab.keys())[:100]])

    def tokid(self, w):
        return self.vocab.get(w.lower(), 0)


N_DISTANCE_FEATURES = 8


def make_distance_features(seq_len):
    """Constructs distance features for a sentence."""
    # how much ahead/behind the other word is
    distances = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            if i < j:
                distances[i, j] = (j - i) / float(seq_len)
    feature_matrices = [distances, distances.T]

    # indicator features on if other word is up to 2 words ahead/behind
    for k in range(3):
        for direction in ([1] if k == 0 else [-1, 1]):
            feature_matrices.append(np.eye(seq_len, k=k*direction))
    features = np.stack(feature_matrices)

    # additional indicator feature for ROOT
    features = np.concatenate(
        [np.zeros([N_DISTANCE_FEATURES - 1, seq_len, 1]),
         features], -1)
    root = np.zeros((1, seq_len, seq_len + 1))
    root[:, :, 0] = 1

    return np.concatenate([features, root], 0)


def attn_linear_combo(tokenizer="default"):
    return Probe(tokenizer=tokenizer)


def attn_and_words(dataset="default",tokenizer="default"):
    return Probe(use_words=True,dataset=dataset,tokenizer=tokenizer)


def words_and_distances(dataset="default",tokenizer="default"):
    return Probe(use_distance_features=True, use_attns=False,
                 use_words=True, hidden_layer=True,dataset=dataset,
                 tokenizer=tokenizer)


class Probe(object):
    """The probing classifier used in Section 5."""

    def __init__(self, use_distance_features=False, use_words=False,
                 use_attns=True, include_transpose=True, hidden_layer=False,
                 dataset="default", tokenizer="default"):
        
        if tokenizer == "bert_o":
            tokenization = tokenization_o
        else:
            tokenization = tokenization_d

        self._embeddings = WordEmbeddings(dataset=dataset)

        # We use a simple model with batch size 1
        self._attns = tf.placeholder(
            shape=[12, 12, None, None], dtype=tf.float32)
        self._labels = tf.placeholder(
            shape=[None], dtype=tf.int32)
        self._features = tf.placeholder(
            shape=[N_DISTANCE_FEATURES, None, None], dtype=tf.float32)
        self._words = tf.placeholder(shape=[None], dtype=tf.int32)

        if use_attns:
            seq_len = tf.shape(self._attns)[-1]
            if include_transpose:
                # Include both directions of attention
                attn_maps = tf.concat(
                    [self._attns,
                     tf.transpose(self._attns, [0, 1, 3, 2])], 0)
                attn_maps = tf.reshape(attn_maps, [288, seq_len, seq_len])
            else:
                attn_maps = tf.reshape(self._attns, [144, seq_len, seq_len])
            # Use attention to start/end tokens to get score for ROOT
            root_features = (
                (tf.get_variable("ROOT_start", shape=[]) * attn_maps[:, 1:-1, 0]) +
                (tf.get_variable("ROOT_end", shape=[]) * attn_maps[:, 1:-1, -1])
            )
            attn_maps = tf.concat([tf.expand_dims(root_features, -1),
                                   attn_maps[:, 1:-1, 1:-1]], -1)
        else:
            # Dummy attention map for models not using attention inputs
            n_words = tf.shape(self._words)[0]
            attn_maps = tf.zeros((1, n_words, n_words + 1))

        if use_distance_features:
            attn_maps = tf.concat([attn_maps, self._features], 0)

        if use_words:
            print("With embeddings")
            word_embedding_matrix = tf.get_variable(
                "word_embedding_matrix",
                initializer=self._embeddings.pretrained_embeddings,
                trainable=False)
            word_embeddings = tf.nn.embedding_lookup(
                word_embedding_matrix, self._words)
            n_words = tf.shape(self._words)[0]
            tiled_vertical = tf.tile(tf.expand_dims(word_embeddings, 0),
                                     [n_words, 1, 1])
            tiled_horizontal = tf.tile(tf.expand_dims(word_embeddings, 1),
                                       [1, n_words, 1])
            word_reprs = tf.concat([tiled_horizontal, tiled_vertical], -1)
            word_reprs = tf.concat([word_reprs, tf.zeros(
                (n_words, 1, tf.shape(word_embedding_matrix)[-1]*2))], 1)  # dummy for ROOT
            if not use_attns:
                attn_maps = tf.concat([
                    attn_maps, tf.transpose(word_reprs, [2, 0, 1])], 0)
        else:
            print("Without embeddings")

        attn_maps = tf.transpose(attn_maps, [1, 2, 0])
        if use_words and use_attns:
            # attention-and-words probe
            weights = tf.layers.dense(word_reprs, attn_maps.shape[-1])
            self._logits = tf.reduce_sum(weights * attn_maps, axis=-1)
        else:
            if hidden_layer:
                # 1-hidden-layer MLP for words-and-distances baseline
                attn_maps = tf.layers.dense(attn_maps, 256,
                                            activation=tf.nn.tanh)
                self._logits = tf.squeeze(tf.layers.dense(attn_maps, 1), -1)
            else:
                # linear combination of attention heads
                attn_map_weights = tf.get_variable("attn_map_weights",
                                                   shape=[attn_maps.shape[-1]])
                self._logits = tf.reduce_sum(
                    attn_map_weights * attn_maps, axis=-1)

        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self._logits, labels=self._labels))
        opt = tf.train.AdamOptimizer(learning_rate=0.002)
        self._train_op = opt.minimize(loss)

        self.tokenizer = tokenization.FullTokenizer(
            vocab_file="bert/vocab.txt",
            do_lower_case=True)

    def _create_feed_dict(self, example):
        shape = example["attns"].shape
        """
    print("attns",shape)
    print(len(example["heads"]))
    print(len(example["words"]))
    print(len([self._embeddings.tokid(w) for w in example["words"]]))
    print(shape[2] - len(example["words"]))
    """
        """
    for i in range(shape[2] - len(example["heads"])-2):
      example["heads"].append(0)
    """
        return {
            self._attns: example["attns"],
            self._labels: example["heads"],
            self._features: make_distance_features(len(example["words"])),
            self._words: [self._embeddings.tokid(w) for w in example["words"]]
        }

    def train(self, sess, example):
        return sess.run(self._train_op, feed_dict=self._create_feed_dict(example))

    def test(self, sess, example):
        return sess.run(self._logits, feed_dict=self._create_feed_dict(example))

def run_training(probe, train_data, dev_data):
    """Trains and evaluates the given attention probe."""
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1):
            print(40 * "=")
            print("EPOCH", (epoch + 1))
            print(40 * "=")
            print("Training...")
            for i, example in enumerate(train_data):
                if i == 0:
                    print(example["attns"][0][0],example["attns"].shape)
                if i % 2000 == 0:
                    print("{:}/{:}".format(i, len(train_data)))
                # print(list(example.keys()))
                probe.train(sess, example)

            print("Evaluating...")
            correct, total = 0, 0
            predicted_results = []
            true_results = []
            for i, example in enumerate(dev_data):
                if i % 1000 == 0:
                    print("{:}/{:}".format(i, len(dev_data)))
                logits = probe.test(sess, example)
                for i, (head, prediction, reln) in enumerate(
                        zip(example["heads"], logits.argmax(-1), example["relns"])):
                    # it is standard to ignore punct for Stanford Dependency evaluation
                    if reln != "punct":
                        predicted_results.append(prediction)
                        true_results.append(head)
                        if head == prediction:
                            correct += 1
                        total += 1
            print(total)
            print("UAS: {:.1f}".format(100 * correct / total))
            #print(metrics.confusion_matrix(true_results, predicted_results))
            print(metrics.classification_report(true_results, predicted_results, digits=3))
