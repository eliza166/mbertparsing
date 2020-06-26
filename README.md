# Finding Universal Dependency patterns in multilingual BERT’s self-attention mechanisms

The methodology to extract the weights of attention, accuracy and train probing classifiers is based on the paper [What Does BERT Look At? An Analysis of BERT's Attention](https://arxiv.org/abs/1906.04341) and the code available on the [github repository](https://github.com/clarkkev/attention-analysis) of the paper.

The procedure that is used to replicate this work is:
1. Use separated_by_examples_mapping_file_econded_nocolons.py to extract word-heads from CoNLL-U files for each language taken from
[Universal Dependencies](https://universaldependencies.org/)
2. Extract attention heads using [github repository](https://github.com/clarkkev/attention-analysis)
3. Analyse using the script [Language_Analysis.py](./Language_Analysis.py). This part calculates accuracy using the attention weights and      using off-sets. The cosine distances between the care heads corresponding to each language are calculated comparing languages.
