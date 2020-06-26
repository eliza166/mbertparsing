import collections
import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import csv
import scipy.spatial
import sklearn.metrics as metrics


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


def extract_n_keys(data_dict, n, i=0, values=None):
    """The purpose of this functions is to extract
    keys and values from data_dict (that is a
    dict structure with dict inside it, in a
    recursive way).
    """
    if values == None:
        values = [None]*(n+1)
    for k, v in data_dict.items():
        values[i] = k
        if i < n-1:
            yield from extract_n_keys(v, n, i+1, values)
        else:
            values[i+1] = v
            yield values

# Code for evaluating individual attention maps and baselines
def evaluate_predictor(prediction_fn, dev_data):
    """Compute accuracies for each relation for the given predictor."""
    n_correct, n_incorrect = collections.Counter(), collections.Counter()
    for example in dev_data:
        words = example["words"]
        # predictions.shape == (n,)
        predictions = prediction_fn(example)

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

# Code for evaluating individual attention maps and baselines
def predictor_f1_score(prediction_fn, dev_data, avg_type = 'macro'):
    """Compute accuracies for each relation for the given predictor."""
    f1_scores = collections.defaultdict(list)
    for example in dev_data:
        # predictions.shape == (n,)
        predictions = prediction_fn(example)
        predict_real = collections.defaultdict(lambda  : [[],[]])
        for _, (p, y, r) in enumerate(zip(predictions, example["heads"],
                                          example["relns"])):
            predict_real[r][0].append(p)
            predict_real[r][1].append(y)
        for r,(y_pred, y_true) in predict_real.items():
            f1_scores[r].append(metrics.f1_score(y_pred,y_true, average= avg_type))
    return {k: sum(scores)/len(scores) for k,scores in f1_scores.items()}


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

#### SCORE FUNCTIONS
def get_score_param(score_fn, att_head_pre_fn, dev_data, mode="normal",**score_fn_params):
    """Get the accuracies of every attention head."""
    scores = collections.defaultdict(dict)
    for layer in range(12):
        for head in range(12):
            # for each layer calculate the argmax attention of each head in it for each word
            scores[layer][head] = score_fn(
                att_head_pre_fn(layer, head, mode), dev_data,**score_fn_params)
    return scores
def get_scores(dev_data, mode="normal"):
    """Get the accuracies of every attention head."""
    return get_score_param(evaluate_predictor, attn_head_predictor, dev_data, mode)
def get_f1_scores(dev_data, mode="normal", avg_type = 'macro'):
    return get_score_param(predictor_f1_score, attn_head_predictor, dev_data, mode, avg_type = avg_type)


def get_offsets_relns_examples(dev_data_dict):
    offsets = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
    for language, dev_data in dev_data_dict.items():
        for example in dev_data:
            for i, head in enumerate(example["heads"]):
                offset = head-i
                offsets[language][example["relns"][i]][offset] += 1
    return offsets

def get_score_lan_param(score_fn, dev_data_dict, **score_fn_params):
    # dev_data_dict = {<language>:<dev_data>}

    # attn_head_scores_lan = {<reln>:{<lan>:{<direction>:<12*12 ndarray>} } }
    # in the 12*12 ndarray, the first component refers to layers and the second to heads
    attn_head_scores_lan = collections.defaultdict(dict)
    for language, dev_data in dev_data_dict.items():
        # attn_head_scores[direction][layer][head][dep_relation] = accuracy
        attn_head_scores = {
            "dep->head": score_fn(dev_data, "normal", **score_fn_params),
            "head<-dep": score_fn(dev_data, "transpose", **score_fn_params)
        }

        # relns_scores = {<reln>: {<direction>: <12*12 ndarray>}}
        relns_scores = {}

        for direction, layer, head, scores in extract_n_keys(attn_head_scores, 3):
            for reln in scores:
                if reln not in relns_scores:
                    relns_scores[reln] = {
                        "dep->head": np.zeros((12, 12)),
                        "head<-dep": np.zeros((12, 12))
                    }
                relns_scores[reln][direction][layer][head] = scores[reln]

        for reln, data in relns_scores.items():
            attn_head_scores_lan[reln][language] = data
    return attn_head_scores_lan
def accuracies(dev_data_dict):
    return get_score_lan_param(get_scores, dev_data_dict)
def get_f1_scores_lan(dev_data_dict,avg_type = 'macro'):
    return get_score_lan_param(get_f1_scores, dev_data_dict, avg_type = avg_type)

def language_layer_score(score_fn, dev_data_dict, **score_fn_params):
    # dev_data_dict = {<language>:<dev_data>}

    # attn_head_scores_lan = {<lan>:[sc01, sc02, ..., sc12] } }
    # in the 12*12 ndarray, the first component refers to layers and the second to heads
    attn_head_scores_lan = {}
    for language, dev_data in dev_data_dict.items():
        # attn_head_scores[direction][layer][head][dep_relation] = accuracy
        attn_head_scores = {
            "dep->head": score_fn(dev_data, "normal", **score_fn_params),
            "head<-dep": score_fn(dev_data, "transpose", **score_fn_params)
        }
        layer_score = [0]*12
        layer_count = [0]*12
        for direction, layer, head, reln, score in extract_n_keys(attn_head_scores, 4):
            layer_score[layer]+=score
            layer_count[layer]+=1
        attn_head_scores_lan[language] = [ layer_score[i] / lc if lc > 0 else 0 for i,lc in enumerate(layer_count)]
    return attn_head_scores_lan
def language_layer_accuracy(dev_data_dict):
    return language_layer_score(get_scores, dev_data_dict)

def get_fn_distance(fn,dev_data_dict,flat=True):
    # distances = {<reln>:{<lani-lanj>:<direction>:<double>}}
    distances = collections.defaultdict(dict)
    languages = list(dev_data_dict.keys())

    for reln, att_scores_lan in accuracies(dev_data_dict).items():
        for i, lani in enumerate(languages):
            for j, lanj in enumerate(languages[i+1:]):
                lanij = lani+"-"+lanj
                directions = {}
                for direction in ("dep->head", "head<-dep"):
                    if lani in att_scores_lan and lanj in att_scores_lan:
                        if flat:
                            ti = att_scores_lan[lani][direction].reshape((144,))
                            tj = att_scores_lan[lanj][direction].reshape((144,))
                        else:
                            ti = att_scores_lan[lani][direction]
                            tj = att_scores_lan[lanj][direction]
                        d = fn(ti,tj)
                        if d != None:
                            directions[direction] = d
                if directions != {}:
                    distances[reln][lanij] = directions
    return distances
def get_cosine_distances(dev_data_dict):
    def calculate(ti,tj):
        if any(ti) and any(tj):
            #return scipy.spatial.distance.cosine(ti, tj)
            return metrics.mean_squared_error(ti, tj)
    return get_fn_distance(calculate,dev_data_dict)
def get_mse(dev_data_dict):
    def calculate(ti,tj):
        if any(ti) and any(tj):
            return metrics.mean_squared_error(ti, tj)
    return get_fn_distance(calculate,dev_data_dict)
def get_max_perm_cosine_d(dev_data_dict):
    def calculate(ti,tj):
        """This code use the fact that the max cosine distance of
        all posible permutations of two vectors is equal to
        ordering the two vector and calculate the cosine distance
        """
        prom_cosine_d = 0
        for i in range(12):
            li,lj = ti[i],tj[i]
            if any(li) and any(lj):
                li,lj = sorted(li,reverse=True),sorted(lj, reverse=True)
                #prom_cosine_d += scipy.spatial.distance.cosine(li, lj)
                prom_cosine_d += scipy.metrics.mean_squared_error(li, lj)
        return prom_cosine_d/12
    return get_fn_distance(calculate,dev_data_dict,flat=False)


def get_all_scores(reln, attn_head_scores):
    """Get all attention head scores for a particular relation."""
    all_scores = []
    for key, layer, head, scores in extract_n_keys(attn_head_scores, 3):
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


def write_rows_to_csv(rows, filename):
    with open(filename, 'w', newline='') as csvfile:
        fwriter = csv.writer(csvfile, delimiter=' ',
                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
        fwriter.writerows(rows)


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
    for direction,layer,head,reln,score in extract_n_keys(attn_head_scores,4):
        all_scores.append((reln[:8],score*100, layer, head, direction))
    all_offset_scores = [(reln[:8],scores[reln]*100, i) for i, scores in baseline_scores.items() for reln in scores]
    write_rows_to_csv(all_scores+["---"]+all_offset_scores, filename)

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
    
    write_rows_to_csv(rows, filename)


def save_all_relns_f1_scores(dev_data, filename, avg_type='macro'):
    # attn_head_scores[direction][layer][head][dep_relation] = f1_score
    attn_head_scores = {
        "dep->head": get_f1_scores(dev_data, "normal", avg_type=avg_type),
        "head<-dep": get_f1_scores(dev_data, "transpose", avg_type=avg_type)
    }
    all_scores = []
    for direction,layer,head,reln,score in extract_n_keys(attn_head_scores,4):
        all_scores.append((reln,score*100, layer, head, direction))
    write_rows_to_csv(all_scores, filename)


def save_fn_distance(fn,dev_data_dict, filename,
                    header = ["relation","lani-lanj","direction","distance"]):
    # ds = {<reln>:{<lani-lanj>:<direction>:<double>}}
    ds = fn(dev_data_dict)
    rows = [header]
    for reln,lang,direction,dist in extract_n_keys(ds,3):
        rows.append([reln,lang,direction,dist])
    write_rows_to_csv(rows, filename)
def save_cosine_distances(dev_data_dict, filename):
    save_fn_distance(get_cosine_distances, dev_data_dict, filename)
def save_mse(dev_data_dict, filename):
    save_fn_distance(get_mse, dev_data_dict, filename,
                    ["relation","lani-lanj","direction","mse"])
def save_max_perm_cosine_d(dev_data_dict, filename):
    save_fn_distance(get_max_perm_cosine_d, dev_data_dict, filename)


def save_relns_offsets(dev_data_dict, filename):
    rows = [["language","relation","offset","count"]]
    goffsets = get_offsets_relns_examples(dev_data_dict)
    for lang, reln, off_set, count in extract_n_keys(goffsets,3):
        rows.append([lang,reln,off_set,count])
    
    write_rows_to_csv(rows, filename)

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

default_style_plot_labels = {
    "title":{
        'weight':'bold',
        'color': "black",
        'size': 16
    },
    "xlabel":{
        'weight':'normal',
        'color': "black",
        'size': 12
    },
    "ylabel":{
        'weight':'normal',
        'color': "black",
        'size': 12
    }
}
def plot_reln_score(reln,dev_data_dict, direction="dep->head",
                        style = default_style_plot_labels):
    image_plot_order ={
        "English_o": 5,
        "English": 3,
        "Spanish":4,
        "Italian":2,
        "German":1
    }

    nlang = len(dev_data_dict)
    unk_pos = set(range(1,nlang+1)).difference(image_plot_order.values())
    npost = iter(unk_pos)
    rows = (nlang-1)//2+1
    columns = 2 if nlang > 1 else 1

    plt.figure(figsize=(9*columns, 9*rows))
    for language, dev_data in dev_data_dict.items():
        print("language ",language)
        attn_head_scores = {}
        if direction == "dep->head":
            attn_head_scores[direction] = get_scores(dev_data, "normal")
        else:
            attn_head_scores[direction] = get_scores(dev_data, "transpose")
        # get_all_scores -> sorted([(scores[reln], layer, head, key)],reverse=True)
        scores = get_all_scores(reln, attn_head_scores)
        scrs = np.ndarray((12,12))
        for score,layer,head,_ in scores:
            scrs[layer][head] = score
        if language in image_plot_order:
            plt.subplot(rows,columns,image_plot_order[language])
        else:
            plt.subplot(rows,columns,npost.__next__())
        plt.imshow(scrs, cmap='Blues', vmin=0, vmax=1)
        plt.colorbar(shrink=0.845)
        plt.grid(False)
        plt.title('{}'.format(language),fontdict=style["title"])
        plt.xlabel('Head',fontdict=style["xlabel"])
        plt.ylabel('Layer',fontdict=style["ylabel"])

def plot_reln_score_layer(relns, dev_data_dict, style=default_style_plot_labels):
    """relns is a list with the relations to graph"""
    all_scores_layer = {}
    for reln, att_scores_lan in accuracies(dev_data_dict).items():
        lans = {}
        for language, att_scores_dir in att_scores_lan.items():
            directions = {}
            for direction, att_scores in att_scores_dir.items():
                directions[direction] = att_scores.max(1).reshape((12,))
            lans[language] = directions
        all_scores_layer[reln] = lans
    
    properties = {
        "English_o": {
            "marker": 's',
            "color": "tab:blue"
            },
        "English": {
            "marker": 's',
            "color": "tab:orange"
            },
        "Spanish": {
            "marker": '^',
            "color": "tab:green"
            },
        "Italian": {
            "marker": 'o',
            "color": "tab:red"
            },
        "German": {
            "marker": 'P',
            "color": "tab:purple"
            }
    }
    
    plt.figure(figsize=(12, 9*len(relns)))
    for i,reln in enumerate(relns):
        all_scores_reln = all_scores_layer[reln]
        plt.subplot(len(relns),1,i+1)
        for lan in all_scores_reln:
            v = all_scores_reln[lan]["dep->head"]
            plt.plot(list(range(12)), v, label=lan, ms=10, **properties[lan])
        plt.xlabel('Layer',fontdict=style["xlabel"])
        plt.xticks(list(range(12)))
        plt.ylabel('Accuracy',fontdict=style["ylabel"])
        #plt.title("Accuracy for {} per layer".format(reln),fontdict=style["title"])
        plt.legend(loc='lower left')
    plt.show()

def plot_language_layer(dev_data_dict, style=default_style_plot_labels):
    lnly = language_layer_accuracy(dev_data_dict)
    properties = {
        "English_o": {
            "marker": 's',
            "color": "tab:blue"
            },
        "English": {
            "marker": 's',
            "color": "tab:orange"
            },
        "Spanish": {
            "marker": '^',
            "color": "tab:green"
            },
        "Italian": {
            "marker": 'o',
            "color": "tab:red"
            },
        "German": {
            "marker": 'P',
            "color": "tab:purple"
            }
    }
    general = [0]*12
    plt.figure(figsize=(10, 7))
    for lan,scores in lnly.items():
        plt.plot(scores,label=lan,linestyle='',**properties[lan])
        for i in range(12):
            general[i]+=scores[i]
    for i in range(12):
        general[i] /= len(lnly)
    plt.plot(general,label="All avg",color="tab:brown")
    plt.legend(loc='lower left')
    plt.xticks(list(range(12)))
    plt.yticks([i/100 for i in range(0,20,2)],list(range(0,20,2)))
    plt.xlabel('Layer',fontdict=style["xlabel"])
    plt.ylabel('Accuracy',fontdict=style["ylabel"])

def plot_relative_freq_relns_examples(dev_data_dict, relns, style=default_style_plot_labels):
    """relns = {<reln>: {"offsets": [<offsets>], "color":<color> }}
    <color> is an acceptable color for plotting with matplotlib
    """
    goffsets = get_offsets_relns_examples(dev_data_dict)
    
    values = collections.defaultdict(dict)
    for lang, reln, offset_count in extract_n_keys(goffsets,2):
        if reln in relns:
            total = sum([count for _,count in offset_count.items()])
            y = [offset_count[off]/total if off in offset_count else 0  for off in relns[reln]["offsets"]]
            values[lang][reln]= y
    
    langs = sorted(list(goffsets.keys()))
    bars_labels = list(relns.keys())
    

    width = 0.85  # the width of the bars [0-1]

    rows, columns = len(bars_labels),len(langs)
    #len_offset_values
    #loffv = [len(relns[r]["offsets"]) for r in bars_labels]
    loffv = [1 for r in langs]
    #gridspec_kw with width_ratios key is used to ensure that the graphs have a proportional size
    fig, axs = plt.subplots(rows, columns,gridspec_kw={'width_ratios': loffv})

    
    for r in range(rows):
        #lang = langs[r]
        #fig.text(0.5, 1-r/rows, lang, ha='center', va='center', fontdict=style["title"])
        reln = bars_labels[r]
        for c in range(columns):
            lang = langs[c]
            ax = axs[r][c]
            reln_att = values[lang][reln]
            ax.bar(relns[reln]["offsets"], reln_att, width, color=relns[reln]["color"])

            ax.set_xticks(relns[reln]["offsets"])
            ax.set_yticks([x/10 for x in range(0, 11, 2)])
            #ax.set_xlabel(reln,fontdict=style["xlabel"])
            ax.set_xlabel(lang,fontdict=style["xlabel"])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if c == 0:
                ax.set_ylabel('Relative frequency density',fontdict=style["ylabel"])
            else:
                ax.spines['left'].set_visible(False)
                ax.get_yaxis().set_visible(False)
    #width = 0.5*sum(loffv)
    width = 22
    fig.set_size_inches(width if width>1 else 1, 5*rows )
    fig.tight_layout()

    plt.show()