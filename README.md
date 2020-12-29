# Finetune Albert on french corpus

## Data preparation

To fine Albert, we need to get the targetted corpus (her we detail the wikipedia data preparation)

* Download the French wikipedia corpus from [Wikipedia](https://dumps.wikimedia.org/)

* Data is preprocessed and extracted using [WikiExtractor](https://github.com/attardi/wikiextractor)

* Training SentencePiece model for producing vocab file, I used 30000 words from this model on French wikipedia corpus. The SPM model was trained by [SentencePiece]((https://github.com/google/sentencepiece))
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

Train the spm files (As said above, you need to Build&Install to get the spm_train command). You can look to the rich options of spm_train. 

```bash
spm_train --input=Output/wikipedia_fr.txt --model_prefix=wikifr2M --vocab_size=30000 --num_threads=12 --input_sentence_size=2000000  --pad_id=0 --unk_id=1 --bos_id=-1 --eos_id=-1 --control_symbols="[CLS],[SEP],[MASK],(,),-,£,$,€"
```

Once run, It will generate wikifr2M.model and wikifr2M.vocab files that will be used later for training




