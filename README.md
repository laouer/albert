# Finetune Albert on french corpus

## Introduction

ALBERT is "A Lite" version of BERT, a popular unsupervised language representation learning algorithm. ALBERT uses parameter-reduction techniques that allow for large-scale configurations, overcome previous memory limitations, and achieve better behavior with respect to model degradation.

For a technical detail description of the algorithm, see the paper:

[ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) (Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut) and the official repository [Google ALBERT](https://github.com/google-research/ALBERT)

Google researchers introduced three standout innovations with ALBERT. [1](https://medium.com/syncedreview/googles-albert-is-a-leaner-bert-achieves-sota-on-3-nlp-benchmarks-f64466dd583)

Factorized embedding parameterization: Researchers isolated the size of the hidden layers from the size of vocabulary embeddings by projecting one-hot vectors into a lower dimensional embedding space and then to the hidden space, which made it easier to increase the hidden layer size without significantly increasing the parameter size of the vocabulary embeddings.

Cross-layer parameter sharing: Researchers chose to share all parameters across layers to prevent the parameters from growing along with the depth of the network. As a result, the large ALBERT model has about 18x fewer parameters compared to BERT-large.

Inter-sentence coherence loss: In the BERT paper, Google proposed a next-sentence prediction technique to improve the model’s performance in downstream tasks, but subsequent studies found this to be unreliable. Researchers used a sentence-order prediction (SOP) loss to model inter-sentence coherence in ALBERT, which enabled the new model to perform more robustly in multi-sentence encoding tasks.



## Data preparation

To finetune Albert, we need to get the targetted corpus (her we detail the wikipedia data preparation)

* Download the French wikipedia corpus from [Wikipedia](https://dumps.wikimedia.org/)

* Data is preprocessed and extracted using [WikiExtractor](https://github.com/attardi/wikiextractor)

* Training SentencePiece model for producing vocab file, I used 30000 words from this model on French wikipedia corpus. The SPM model was trained by [SentencePiece](https://github.com/google/sentencepiece)
  * To train spm vocab model You need to build&Install sentencePiece not only the python module

### The dump wiki corpus 

First step is to get your corpus: I choosed the wikipedia corpus as it globally clean and rich. <https://dumps.wikimedia.org/mirrors.html>

1. Get your the articles-multistream (from a suitable mirror) example:

    ```bash
    wget https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/frwiki/20201220/frwiki-20201220-pages-articles-multistream.xml.bz2 
    ```

2. We need to extract and clean the corpus,. So after installing wikiextractor (see the above link for details), run the following command on the wikipedia dump:

    ```bash
    python -m wikiextractor.WikiExtractor -o Output -b 100M frwiki-20201220-pages-articles-multistream.xml.bz2
    ```

   * It will generate something like 480 xml files
   * For training we will need just the text of articles, so running the next commandline Will :
     * Concat all articles in one file
     * keep only articles text removing the html tags
     * Remove empty lines
     * Will keep lines that are longer than 5 words (really ampirical approach)

    ```bash
    find Output -type f -exec grep -v -e  "<doc" -e "</doc" -e '^$' {} \; | awk  'NF>=5' > wikipedia_fr.txt
    ```

### Generate the spm vocal files

Train the spm files (As said above, you need to Build&Install to get the spm_train command). You can look to the rich [options](https://github.com/google/sentencepiece/blob/master/doc/options.md) of spm_train. 

```bash
spm_train --input=Output/wikipedia_fr.txt --model_prefix=wikifr2M --vocab_size=30000 --num_threads=12 --input_sentence_size=2000000  --pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1 --control_symbols="[CLS],[SEP],[MASK],(,),-,£,$,€"
```

Once run, It will generate wikifr2M.model and wikifr2M.vocab files that will be used later for training.

* Ex: Head from wikifr2M.vocab

    ```text
    <pad>	0
    <unk>	0
    [CLS]	0
    [SEP]	0
    [MASK]	0
    (	    0
    )	    0
    -       0
    £	    0
    $	    0
    €	    0
    ▁	   -2.94116
    ▁de	   -3.13249
    ...

    ```

## Pretraining

### Get needed tools

To run Albert finetuning, You will need tensorflow 1.5 and moreover you will need albert google code

```bash
git clone https://github.com/google-research/albert albert
```

### Creating data for pretraining

We WILL train ALBERT model on config version 2 of base and large the Other configurations in the folder [config](config/base/)

* Base Config

    ```Json
    {
    "attention_probs_dropout_prob": 0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0,
    "embedding_size": 128,
    "hidden_size": 768,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "max_position_embeddings": 512,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "num_hidden_groups": 1,
    "net_structure_type": 0,
    "gap_size": 0,
    "num_memory_blocks": 0,
    "inner_group_num": 1,
    "down_scale_factor": 1,
    "type_vocab_size": 2,
    "vocab_size": 30000
    }```

* Large Config

    ```Json
    {
    "attention_probs_dropout_prob": 0,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0,
    "embedding_size": 128,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "max_position_embeddings": 512,
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "num_hidden_groups": 1,
    "net_structure_type": 0,
    "gap_size": 0,
    "num_memory_blocks": 0,
    "inner_group_num": 1,
    "down_scale_factor": 1,
    "type_vocab_size": 2,
    "vocab_size": 30000
    }```

### Create tfrecord for training data:

```/bin/bash
python create_pretraining_data.py \
  --input_file=Output/wikipedia_fr.txt \
  --dupe_factor=10 \
  --output_file=Output/wikipedia_fr.tf_record \
  --vocab_file=wikifr2M.model.vocab \
  --spm_model_file=wikifr2M.model 
```
