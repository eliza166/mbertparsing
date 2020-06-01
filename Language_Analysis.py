import collections
import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import csv
import scipy.spatial

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

# Code for evaluating individual attention maps and baselines


def evaluate_predictor(prediction_fn, dev_data):
    """Compute accuracies for each relation for the given predictor."""
    n_correct, n_incorrect = collections.Counter(), collections.Counter()
    for example in dev_data:
        words = example["words"]
        #predictions.shape == (n,)
        predictions = prediction_fn(example)

        # the next cycle truncates some values on predictions, don't know why
        # it seams that len(example["heads"]) != len(predictions)
        for i, (p, y, r) in enumerate(zip(predictions, example["heads"],
                                          example["relns"])):
            is_correct = (p == y)
            if r == "poss" and p < len(words):
                # Special case for poss (see discussion in Section 4.2)
                if i < len(words) and words[i + 1] == "'s" or words[i + 1] == "s'":
                    is_correct = (predictions[i + 1] == y)
            if is_correct:
                n_correct[r] += 1
                n_correct["all"] += 1
            else:
                n_incorrect[r] += 1
                n_incorrect["all"] += 1
    return {k: n_correct[k] / float(n_correct[k] + n_incorrect[k])
            for k in set(list(n_incorrect.keys())+list(n_correct.keys()))}


def attn_head_predictor(layer, head, mode="normal"):
    """Assign each word the most-attended-to other word as its head."""
    def predict(example):
        attn = np.array(example["attns"][layer][head])
        if mode == "transpose":
            attn = attn.T
        elif mode == "both":
            attn += attn.T
        else:
            assert mode == "normal"
        # ignore attention to self and [CLS]/[SEP] tokens
        # this will set to 0 attn[0,0], attn[1,1], ..., attn[n,n]
        attn[range(attn.shape[0]), range(attn.shape[0])] = 0
        # del first and last column also row
        attn = attn[1:-1, 1:-1]
        # output is (n-1,) [in the image, is like the max of each row for head]
        return np.argmax(attn, axis=-1) + 1  # +1 because ROOT is at index 0
    return predict


def offset_predictor(offset):
    """Simple baseline: assign each word the word a fixed offset from
    it (e.g., the word to its right) as its head."""
    def predict(example):
        return [max(0, min(i + offset + 1, len(example["words"])))
                for i in range(len(example["words"]))]
    return predict


def get_scores(dev_data, mode="normal"):
    """Get the accuracies of every attention head."""
    scores = collections.defaultdict(dict)
    for layer in range(12):
        for head in range(12):
            # for each layer calculate the argmax attention of each head in it for each word
            scores[layer][head] = evaluate_predictor(
                attn_head_predictor(layer, head, mode), dev_data)
    return scores

def accuracies(dev_data_dict):
    #dev_data_dict = {<language>:<dev_data>}

    #attn_head_scores_lan = {<reln>:{<lan>:{<direction>:<12*12 ndarray>} } }
    attn_head_scores_lan = collections.defaultdict(dict)
    for language,dev_data in dev_data_dict.items():
        # attn_head_scores[direction][layer][head][dep_relation] = accuracy
        attn_head_scores = {
            "dep->head": get_scores(dev_data, "normal"),
            "head<-dep": get_scores(dev_data, "transpose")
        }
        
        #relns_scores = {<reln>: {<direction>: <12*12 ndarray>}}
        relns_scores = {}
        for direction, layer_head_scores in attn_head_scores.items():
            for layer, head_scores in layer_head_scores.items():
                for head, scores in head_scores.items():
                    for reln in scores:
                        if reln in relns_scores:
                            relns_scores[reln][direction][layer][head] = scores[reln]
                        else:
                            relns_scores[reln] ={
                                "dep->head": np.zeros((12,12)),
                                "head<-dep": np.zeros((12,12))
                            }
        
        for reln, data in relns_scores.items():
            attn_head_scores_lan[reln][language] = data
    
    return attn_head_scores_lan

def get_cosine_distances(dev_data_dict):

    #distances = {<reln>:{<lani-lanj>:<direction>:<double>}}
    distances = collections.defaultdict(dict)
    languages = list(dev_data_dict.keys())

    for reln,att_scores_lan in accuracies(dev_data_dict).items():
        for i,lani in enumerate(languages):
            for j,lanj in enumerate(languages[i+1:]):
                lanij = lani+"-"+lanj
                directions = {}
                for direction in ("dep->head","head<-dep"):
                    if lani in att_scores_lan and lanj in att_scores_lan:
                        ti = att_scores_lan[lani][direction].reshape((144,))
                        tj = att_scores_lan[lanj][direction].reshape((144,))
                        if sum(ti)!= 0 and sum(tj)!=0:
                            d = scipy.spatial.distance.cosine(ti,tj)
                        directions[direction] = d
                if directions != {}:
                    distances[reln][lanij] = directions
    
    return distances
                        

def get_all_scores(reln, attn_head_scores):
    """Get all attention head scores for a particular relation."""
    all_scores = []
    for key, layer_head_scores in attn_head_scores.items():
        for layer, head_scores in layer_head_scores.items():
            for head, scores in head_scores.items():
                all_scores.append((scores[reln], layer, head, key))
    return sorted(all_scores, reverse=True)


def show_best_relns(dev_data):
    # attn_head_scores[direction][layer][head][dep_relation] = accuracy
    attn_head_scores = {
        "dep->head": get_scores(dev_data, "normal"),
        "head<-dep": get_scores(dev_data, "transpose")
    }
    # baseline_scores[offset][dep_relation] = accuracy
    baseline_scores = {
        i: evaluate_predictor(offset_predictor(i), dev_data) for i in range(-3, 3)
    }

    reln_counts = collections.Counter()
    reln_counts = collections.Counter()
    for example in dev_data:
        for reln in example["relns"]:
            reln_counts[reln] += 1

    for __, (reln, _) in enumerate([("all", 0)] + reln_counts.most_common()):
        if reln == "root" or reln == "punct":
            continue
        if reln_counts[reln] < 100 and reln != "all":
            break

        uas, layer, head, direction = get_all_scores(reln, attn_head_scores)[0]
        baseline_uas, baseline_offset = max(
            (scores[reln], i) for i, scores in baseline_scores.items())
        print("{:8s} | {:5d} | attn: {:.1f} | offset={:2d}: {:.1f} | {:}-{:} {:}".format(
            reln[:8], reln_counts[reln], 100 *
            uas, baseline_offset, 100 * baseline_uas,
            layer, head, direction))

def save_all_relns(dev_data, filename):
	"""save in <filename> all the accuracies for each relation in <dev_data>

	The output is the following
	rel1 score1 layer1 head1 direction1
	.
	.
	.
	reln scoren layern headn directionn
	- - -
	rel1 baseline_score1  offset
	.
	.
	.
	relm baseline_scorem  offset
    """
	# attn_head_scores[direction][layer][head][dep_relation] = accuracy
	attn_head_scores = {
        "dep->head": get_scores(dev_data, "normal"),
        "head<-dep": get_scores(dev_data, "transpose")
    }
	# baseline_scores[offset][dep_relation] = accuracy
	baseline_scores = {
        i: evaluate_predictor(offset_predictor(i), dev_data) for i in range(-3, 3)
    }
	
	all_scores = []
	for direction, layer_head_scores in attn_head_scores.items():
		for layer, head_scores in layer_head_scores.items():
			for head, scores in head_scores.items():
				for reln in scores:
					all_scores.append((reln[:8],scores[reln]*100, layer, head, direction))
	
	all_offset_scores = [(reln[:8],scores[reln]*100, i) for i, scores in baseline_scores.items() for reln in scores]
	with open(filename, 'w', newline='') as csvfile:
		fwriter = csv.writer(csvfile, delimiter=' ',
                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
		fwriter.writerows(all_scores+["---"]+all_offset_scores)

def save_best_relns(dev_data, filename):
    """save in <filename> the best accuracy for each relation in <dev_data>
    """
    # attn_head_scores[direction][layer][head][dep_relation] = accuracy
    attn_head_scores = {
        "dep->head": get_scores(dev_data, "normal"),
        "head<-dep": get_scores(dev_data, "transpose")
    }
    # baseline_scores[offset][dep_relation] = accuracy
    baseline_scores = {
        i: evaluate_predictor(offset_predictor(i), dev_data) for i in range(-3, 3)
    }

    reln_counts = collections.Counter()
    reln_counts = collections.Counter()
    for example in dev_data:
        for reln in example["relns"]:
            reln_counts[reln] += 1

    rows = []
    for __, (reln, _) in enumerate([("all", 0)] + reln_counts.most_common()):
        uas, layer, head, direction = get_all_scores(reln, attn_head_scores)[0]
        baseline_uas, baseline_offset = max(
            (scores[reln], i) for i, scores in baseline_scores.items())
        rows.append([reln[:8], reln_counts[reln], 100 * uas, baseline_offset, 100 * baseline_uas,
                     layer, head, direction])

    with open(filename, 'w', newline='') as csvfile:
        fwriter = csv.writer(csvfile, delimiter=' ',
                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
        fwriter.writerows(rows)

def save_cosine_distances(dev_data_dict, filename):

    #ds = {<reln>:{<lani-lanj>:<direction>:<double>}}
    ds = get_cosine_distances(dev_data_dict)
    rows = [["relation","lani-lanj","direction","distance"]]
    for reln,lang_dist in ds.items():
        for lang, direction_dist in lang_dist.items():
            for direction,dist in direction_dist.items():
                rows.append([reln,lang,direction,dist])
    with open(filename, 'w', newline='') as csvfile:
        fwriter = csv.writer(csvfile, delimiter=' ',
                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
        fwriter.writerows(rows)

def plot_attn(title, examples, layer, head, color_words,
              color_from=True, width=3, example_sep=3,
              word_height=1, pad=0.1, hide_sep=False):
    """Plot BERT's attention for a particular head/example."""
    plt.figure(figsize=(4, 4))
    for i, example in enumerate(examples):
        yoffset = 0
        if i == 0:
            yoffset += (len(examples[0]["words"]) -
                        len(examples[1]["words"])) * word_height / 2
        xoffset = i * width * example_sep
        attn = example["attns"][layer][head]
        if hide_sep:
            attn = np.array(attn)
            attn[:, 0] = 0
            attn[:, -1] = 0
            attn /= attn.sum(axis=-1, keepdims=True)

        words = ["[CLS]"] + example["words"] + ["[SEP]"]
        n_words = len(words)
        for position, word in enumerate(words):
            for x, from_word in [(xoffset, True), (xoffset + width, False)]:
                color = "k"
            if from_word == color_from and word in color_words:
                color = "#cc0000"
            plt.text(x, yoffset - (position * word_height), word,
                     ha="right" if from_word else "left", va="center",
                     color=color)

        for i in range(n_words):
            for j in range(n_words):
                color = "b"
            if words[i if color_from else j] in color_words:
                color = "r"
            plt.plot([xoffset + pad, xoffset + width - pad],
                     [yoffset - word_height * i, yoffset - word_height * j],
                     color=color, linewidth=1, alpha=attn[i, j])
    plt.axis("off")
    plt.title(title)
    plt.show()


def image_plot_order(language):
    if language == "English":
        return 1
    elif language == "Spanish":
        return 3
    elif language == "Italian":
        return 4
    elif language == "German":
        return 2
    return 1

def plot_reln_score(reln,dev_data_dict, direction="dep->head"):
    plt.figure(figsize=(24, 18))
    for language, dev_data in dev_data_dict.items():
        attn_head_scores = {}
        if direction == "dep->head":
            attn_head_scores[direction] = get_scores(dev_data, "normal")
        else:
            attn_head_scores[direction] = get_scores(dev_data, "transpose")
        #get_all_scores -> sorted([(scores[reln], layer, head, key)],reverse=True)
        scores = get_all_scores(reln, attn_head_scores)
        scrs = np.ndarray((12,12))
        for score,layer,head,_ in scores:
            scrs[layer][head] = score
        plt.subplot(2,2,image_plot_order(language))
        plt.imshow(scrs, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar()
        plt.grid(False)
        plt.title('Scores of {} per layer-head in {}'.format(reln,language))
        plt.xlabel('Head id')
        plt.ylabel('Layer id')