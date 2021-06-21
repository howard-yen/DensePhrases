# DensePhrases

<div align="center">
  <img alt="DensePhrases Demo" src="https://github.com/princeton-nlp/DensePhrases/blob/main/densephrases/demo/static/files/preview.gif" width="750px">
</div>

<em>DensePhrases</em> is an extractive phrase search tool based on your natural language inputs. From 5 million Wikipedia articles, it can search phrase-level answers to your questions or find related entities to (subject, relation) pairs in real-time. Due to the extractive nature of DensePhrases, it always provides an evidence passage for each phrase. Please see our paper [
Learning Dense Representations of Phrases at Scale (Lee et al., 2021)](https://arxiv.org/abs/2012.12624) for more details.

**\*\*\*\*\* You can try out our online demo of DensePhrases [here](http://densephrases.korea.ac.kr)! \*\*\*\*\***

#### Updates
**\*\*\*\*\* New June 14, 2021: Major code updates \*\*\*\*\***

## Quick Links
* [Installation](#installation)
* [Resources](#resources)
* [Creating a Custom Phrase Index with DensePhrases](#creating-a-custom-phrase-index-with-densephrases)
* [Playing with a DensePhrases Demo](#playing-with-a-densephrases-demo)
* [Traning, Indexing and Inference](#densephrases-training-indexing-and-inference)
* [Pre-processing](#pre-processing)

## Installation
```bash
# Install torch with conda (please check your CUDA version)
conda create -n densephrases python=3.7
conda activate densephrases
conda install pytorch=1.7.1 cudatoolkit=11.0 -c pytorch

# Install apex
git clone https://www.github.com/nvidia/apex.git
cd apex
python setup.py install
cd ..

# Install DensePhrases
git clone https://github.com/princeton-nlp/DensePhrases.git
cd DensePhrases
pip install -r requirements.txt
python setup.py develop
```

## Resources
Before downloading the required files below, please set the default directories as follows and ensure that you have enough storage to download and unzip the files:
```bash
# Running config.sh will set the following three environment variables:
# DATA_DIR: for datasets (including 'kilt', 'open-qa', 'single-qa', 'truecase', 'wikidump')
# SAVE_DIR: for pre-trained models or index; new models and index will also be saved here
# CACHE_DIR: for cache files from huggingface transformers
source config.sh
```
To download the resources described below, you can use `download.sh` as follows:
```bash
# Use bash script to download data (change data to models or index accordingly)
source download.sh
Choose a resource to download [data/wiki/models/index]: data
data will be downloaded at ...
...
Downloading data done!
```

### 1. Datasets
* [Datasets](https://nlp.cs.princeton.edu/projects/densephrases/densephrases-data.tar.gz) (1GB) - Pre-processed datasets including reading comprehension, generated questions, open-domain QA and slot filling. Download and unzip it under `$DATA_DIR` or use `download.sh`.
* [Wikipedia dumps](https://nlp.cs.princeton.edu/projects/densephrases/wikidump.tar.gz) (5GB) - Pre-processed Wikipedia dumps in different sizes. See [here](#2-creating-a-phrase-index) for more details. Download and unzip it under `$DATA_DIR` or use `download.sh`.
```bash
# Check if the download is complete
ls $DATA_DIR
kilt  open-qa  single-qa  truecase  wikidump
```

### 2. Pre-trained Models
* [Pre-trained models](https://nlp.cs.princeton.edu/projects/densephrases/outputs.tar.gz) (8GB) - Pre-trained DensePhrases models (including cross-encoder teacher models). Download and unzip it under `$SAVE_DIR` or use `download.sh`.
```bash
# Check if the download is complete
ls $SAVE_DIR
densephrases-multi  densephrases-multi-query-nq  ...  spanbert-base-cased-squad
```
You can also download each of pre-trained DensePhrases models as listed below.
|              Model              | Evaluation (Test) | OpenQA (EM) |
|:-------------------------------|:--------:|:--------:|
| [densephrases-multi](https://nlp.cs.princeton.edu/projects/densephrases/models/densephrases-multi.tar.gz) | NaturalQuestions | 31.9 |
| [densephrases-multi-query-nq](https://nlp.cs.princeton.edu/projects/densephrases/models/densephrases-multi-query-nq.tar.gz) | NaturalQuestions | 41.3 |
| [densephrases-multi-query-trec](https://nlp.cs.princeton.edu/projects/densephrases/models/densephrases-multi-query-trec.tar.gz) | CuratedTREC | 52.9 |
| [densephrases-multi-query-wq](https://nlp.cs.princeton.edu/projects/densephrases/models/densephrases-multi-query-wq.tar.gz) | WebQuestions | 41.5 |
| [densephrases-multi-query-tqa](https://nlp.cs.princeton.edu/projects/densephrases/models/densephrases-multi-query-tqa.tar.gz) | TriviaQA | 53.5 |
| [densephrases-multi-query-sqd](https://nlp.cs.princeton.edu/projects/densephrases/models/densephrases-multi-query-sqd.tar.gz) | SQuAD | 34.5 |
| [densephrases-multi-query-multi](https://nlp.cs.princeton.edu/projects/densephrases/models/densephrases-multi-query-multi.tar.gz) | NaturalQuestions | 40.9 |

|              Model              | Evaluation (Test) | SlotFilling (KILT-AC) |
|:-------------------------------|:--------:|:--------:|
| [densephrases-multi-query-trex](https://nlp.cs.princeton.edu/projects/densephrases/models/densephrases-multi-query-trex.tar.gz) | T-REx | 22.3 |
| [densephrases-multi-query-zsre](https://nlp.cs.princeton.edu/projects/densephrases/models/densephrases-multi-query-zsre.tar.gz) | Zero shot RE | 40.0 |

- `densephrases-multi`                      : DensePhrases trained on multiple reading comprehension datasets (C\_phrase = {NQ, WQ, TREC, TQA, SQuAD}) without any query-side fine-tuning
- `densephrases-multi-query-*`          : `densephrases-multi` query-side fine-tuned on *
- `densephrases-multi-query-multi`         : `densephrases-multi` query-side fine-tuned on 5 open-domain QA datasets (NQ, WQ, TREC, TQA, SQuAD); Used for the [demo]
- `spanbert-base-cased-*`             : cross-encoder teacher models trained on \*

Test set performance was measured on [the phrase index for the full Wikipedia scale](#3-phrase-index). Note that the query-side fine-tuned models are trained with a different index structure (i.e., IVFOPQ) compared to IVFSQ described in the paper, hence showing slightly different performances.

### 3. Phrase Index
Please note that you don't need to download this phrase index unless you want to work on the full Wikipedia scale.
* [densephrases-multi_wiki-20181220](https://nlp.cs.princeton.edu/projects/densephrases/densephrases-multi_wiki-20181220.tar.gz) (74GB) - Phrase index for the 20181220 version of Wikipedia. Download and unzip it under `$SAVE_DIR` or use `download.sh`.
```bash
# Check if the download is complete
ls $SAVE_DIR
...  densephrases-multi_wiki-20181220
```
#### From 320GB to 74GB
Since hosting the 320GB phrase index described in our paper is costly, we provide an index with a much smaller size (74GB), which includes our recent efforts to reduce the size of the phrase index using [Optimized Product Quantization](https://ieeexplore.ieee.org/document/6678503) with Inverted File System (IVFOPQ). With IVFOPQ, you do not need any SSDs for the real-time inference (the index is loaded on RAM), and you can also reconstruct the phrase vectors from it for the query-side fine-tuning (hence do not need the additional 500GB).

## Creating a Custom Phrase Index with DensePhrases
Basically, DensePhrases uses a text corpus pre-processed in the following format:
```
{
    "data": [
        {
            "title": "America's Got Talent (season 4)",
            "paragraphs": [
                {
                    "context": " The fourth season of \"America's Got Talent\", ... Country singer Kevin Skinner was named the winner on September 16, 2009 ..."
                },
                {
                    "context": " Season four was Hasselhoff's final season as a judge. This season started broadcasting live on August 4, 2009. ..."
                },
                ...
            ]
        },
    ]
}
```
Each `context` contains a single natural paragraph of a variable length. See `sample/articles.json` for example. The following command creates phrase vectors for the custom corpus (`sample/articles.json`) with the `densephrases-multi` model.

```bash
python generate_phrase_vecs.py \
    --model_type bert \
    --pretrained_name_or_path SpanBERT/spanbert-base-cased \
    --data_dir ./ \
    --cache_dir $CACHE_DIR \
    --predict_file sample/articles.json \
    --do_dump \
    --max_seq_length 512 \
    --doc_stride 500 \
    --fp16 \
    --filter_threshold -2.0 \
    --append_title \
    --load_dir $SAVE_DIR/densephrases-multi \
    --output_dir $SAVE_DIR/densephrases-multi_sample
```
The phrase vectors (and their metadata) will be saved under `$SAVE_DIR/densephrases-multi_sample/dump/phrase`. Now you need to create a faiss index as follows:
```bash
python build_phrase_index.py \
    $SAVE_DIR/densephrases-multi_sample/dump all \
    --replace \
    --num_clusters 32 \
    --fine_quant OPQ96 \
    --doc_sample_ratio 1.0 \
    --vec_sample_ratio 1.0 \
    --cuda

# Compress metadata for faster inference
python scripts/preprocess/compress_metadata.py \
    --input_dump_dir $SAVE_DIR/densephrases-multi_sample/dump/phrase \
    --output_dir $SAVE_DIR/densephrases-multi_sample/dump
```
Note that this example uses a very small text corpus and the hyperparameters for `build_phrase_index.py` in a larger scale corpus can be found [here](#densephrases-training-indexing-and-inference).
The phrase index (with IVFOPQ) will be saved under `$SAVE_DIR/densephrases-multi_sample/dump/start`. You can use this phrase index to run a [demo](#playing-with-a-densephrases-demo) or evaluate your set of queries.
For instance, you can feed a set of questions (`sample/questions.json`) to the custom phrase index as follows:
```bash
python eval_phrase_retrieval.py \
    --run_mode eval \
    --cuda \
    --dump_dir $SAVE_DIR/densephrases-multi_sample/dump \
    --index_dir start/32_flat_OPQ96 \
    --query_encoder_path $SAVE_DIR/densephrases-multi \
    --test_path sample/questions.json \
    --save_pred \
    --truecase
```
The prediction file will be saved as `$SAVE_DIR/densephrases-multi/pred/questions_3_top10.pred`, which shows the answer phrases and the passages that contain the phrases:
```
{
    "1": {
        "question": "Who won season 4 of America's got talent",
        ...
        "prediction": [
            "Kevin Skinner",
            ...
        ],
        "evidence": [
            "The fourth season of \"America's Got Talent\", an American television reality show talent competition, premiered on the NBC network on June 23, 2009. Country singer Kevin Skinner was named the winner on September 16, 2009.",
            ...
        ],
    }
    ...
}
```
For creating a large-scale phrase index (e.g., Wikipedia), see [dump_phrases.py](https://github.com/princeton-nlp/DensePhrases/blob/main/scripts/parallel/dump_phrases.py) for an example, which is also explained [here](#2-creating-a-phrase-index).

## Playing with a DensePhrases Demo
There are two ways of using DensePhrases demo.
1. You can simply use the [demo] that we are serving on our server (Wikipedia scale). The running demo is using `densephrases-multi-query-multi` (NQ=40.8 EM) as a query encoder and `densephrases-multi_wiki-20181220` as a phrase index.
2. You can run the demo on your own server where you can change the phrase index (obtained from [here](#creating-a-custom-phrase-index-with-densephrases)) or the query encoder (e.g., to `densephrases-multi-query-nq`).

The minimum resource requirement for running the full Wikipedia scale demo is:
* 100GB RAM
* Single 11GB GPU (optional)

Note that you no longer need any SSDs to run the demo unlike previous phrase retrieval models ([DenSPI](https://github.com/uwnlp/denspi), [DenSPI+Sparc](https://github.com/jhyuklee/sparc)), but setting `$SAVE_DIR` to an SSD can reduce the loading time of the required resources. The following commands serve exactly the same demo as [here](http://densephrases.korea.ac.kr) on your `http://localhost:51997`.
```bash
# Serve a query encoder on port 1111
nohup python run_demo.py \
    --run_mode q_serve \
    --cache_dir $CACHE_DIR \
    --query_encoder_path $SAVE_DIR/densephrases-multi-query-multi \
    --cuda \
    --max_query_length 32 \
    --query_port 1111 > $SAVE_DIR/logs/q-serve_1111.log &

# Serve a phrase index on port 51997 (takes several minutes)
nohup python run_demo.py \
    --run_mode p_serve \
    --index_dir start/1048576_flat_OPQ96 \
    --cuda \
    --truecase \
    --dump_dir $SAVE_DIR/densephrases-multi_wiki-20181220/dump/ \
    --query_port 1111 \
    --index_port 51997 > $SAVE_DIR/logs/p-serve_51997.log &

# Below are the same but simplified commands using Makefile
make q-serve MODEL_NAME=densephrases-multi-query-multi Q_PORT=1111
make p-serve DUMP_DIR=$SAVE_DIR/densephrases-multi_wiki-20181220/dump/ Q_PORT=1111 I_PORT=51997
```
Please change `--query_encoder_path` or `--dump_dir` if necessary and remove `--cuda` for CPU-only version. Once you set up the demo, the log files in `$SAVE_DIR/logs/` will be automatically updated whenever a new question comes in. You can also send queries to your server using mini-batches of questions for faster inference.

```bash
# Test on NQ test set
python run_demo.py \
    --run_mode eval_request \
    --index_port 51997 \
    --test_path $DATA_DIR/open-qa/nq-open/test_preprocessed.json \
    --eval_batch_size 64 \
    --save_pred \
    --truecase

# Same command with Makefile
make eval-demo I_PORT=51997

# Result
(...)
INFO - eval_phrase_retrieval -   {'exact_match_top1': 40.83102493074792, 'f1_score_top1': 48.26451418695196}
INFO - eval_phrase_retrieval -   {'exact_match_top10': 60.11080332409972, 'f1_score_top10': 68.47386731458751}
INFO - eval_phrase_retrieval -   Saving prediction file to $SAVE_DIR/pred/test_preprocessed_3610_top10.pred
```
For more details (e.g., changing the test set), please see the targets in `Makefile` (`q-serve`, `p-serve`, `eval-demo`, etc).

## DensePhrases: Training, Indexing and Inference
In this section, we introduce a step-by-step procedure to train DensePhrases, create phrase vectors and indexes, and run inferences with the trained model.
All of our commands here are simplified as `Makefile` targets, which include exact dataset paths, hyperparameter settings, etc.

If the following test run completes without an error after the installation and the download, you are good to go!
```bash
# Test run for checking installation (takes about 10 mins; ignore the performance)
make draft MODEL_NAME=test
```

<div align="center">
  <img alt="DensePhrases Steps" src="https://github.com/princeton-nlp/DensePhrases/blob/main/densephrases/demo/static/files/overview_new.png" width="850px">
</div>

- A figure summarizing the overall process below

### 1. Training phrase and query encoders
To train DensePhrases from scratch, use `run-rc-nq` in `Makefile`, which trains DensePhrases on NQ (pre-processed for the reading comprehension task) and evaluate it on reading comprehension as well as on (semi) open-domain QA.
You can simply change the training set by modifying the dependencies of `run-rc-nq` (e.g., `nq-rc-data` => `sqd-rc-data` and `nq-param` => `sqd-param` for training on SQuAD).
You'll need a single 24GB GPU for training DensePhrases on reading comprehension tasks, but you can use smaller GPUs by setting `--gradient_accumulation_steps` properly.
```bash
# Train DensePhrases on NQ with Eq. 9
make run-rc-nq MODEL_NAME=densephrases-nq
```

`run-rc-nq` is composed of the six commands as follows (in case of training on NQ):
1. `make train-rc ...`: Train DensePhrases on NQ with Eq. 9 (L = lambda1 L\_single + lambda2 L\_distill + lambda3 L\_neg) with generated questions.
2. `make train-rc ...`: Load trained DensePhrases in the previous step and further train it with Eq. 9 with pre-batch negatives.
3. `make gen-vecs`: Generate phrase vectors for D\_small (= set of all passages in NQ dev).
4. `make index-vecs`: Build a phrase index for D\_small.
5. `make compress-meta`: Compresss metadata for faster inference.
6. `make eval-index ...`: Evaluate the phrase index on the development set questions.

At the end of step 2, you will see the performance on the reading comprehension task where a gold passage is given (about 72.0 EM on NQ dev). Step 6 gives the performance on the semi-open-domain setting (denoted as D\_small; see Table 6 in the paper) where the entire passages from the NQ development set is used for the indexing (about 62.0 EM with NQ dev questions). The trained model will be saved under `$SAVE_DIR/$MODEL_NAME`. Note that during the single-passage training on NQ, we exclude some questions in the development set, whose annotated answers are found from a list or a table.

###  2. Creating a phrase index
Let's assume that you have a pre-trained DensePhrases named `densephrases-multi`, which can also be downloaded from [here](#2-pre-trained-models).
Now, you can generate phrase vectors for a large-scale corpus like Wikipedia using `gen-vecs-parallel`.
Note that you can just download [the phrase index for the full Wikipedia scale](#3-phrase-index) and skip this section.
```bash
# Generate phrase vectors in parallel for a large-scale corpus (default = wiki-dev)
make gen-vecs-parallel MODEL_NAME=densephrases-multi START=0 END=8
```
The default text corpus for creating phrase vectors is `wiki-dev` located in `$DATA_DIR/wikidump`. We have three options for larger text corpora:
- `wiki-dev`: 1/100 Wikipedia scale (sampled), 8 files
- `wiki-dev-noise`: 1/10 Wikipedia scale (sampled), 500 files
- `wiki-20181220`: full Wikipedia (20181220) scale, 5621 files

The `wiki-dev*` corpora also contain passages from the NQ development set, so that you can track the performance of your model witn an increasing size of the text corpus (usually decreases as it gets larger). The phrase vectors will be saved as hdf5 files in `$SAVE_DIR/$(MODEL_NAME)_(data_name)/dump` (e.g., `$SAVE_DIR/densephrases-multi_wiki-dev/dump`), which will be referred to `$DUMP_DIR` below.

#### Parallelization
`START` and `END` specify the file index in the corpus (e.g., `START=0 END=8` for `wiki-dev` and `START=0 END=5621` for `wiki-20181220`).  Each run of `gen-vecs-parallel` only consumes 2GB in a single GPU, and you can distribute the processes with different `START` and `END` using slurm or shell script (e.g., `START=0 END=200`, `START=200 END=400`, ..., `START=5400 END=5621`). Distributing 28 processes on 4 24GB GPUs (each processing about 200 files) can create phrase vectors for `wiki-20181220` in 8 hours. Processing the entire Wikiepdia requires up to 500GB and we recommend using an SSD to store them if possible (a smaller corpus can be stored in a HDD).

After generating the phrase vectors, you need to create a phrase index for the sublinear time search of phrases. Here, we use IVFOPQ for the phrase index.
```bash
# Create IVFOPQ index for a set of phrase vectors
make index-vecs DUMP_DIR=$SAVE_DIR/densephrases-multi_wiki-dev/dump/
```

For `wiki-dev-noise` and `wiki-20181220`, you need to modify the number of clusters to 101,372 and 1,048,576, respectively (simply change `medium1-index` in `ìndex-vecs` to `medium2-index` or `large-index`). For `wiki-20181220` (full Wikipedia), this takes about 1~2 days depending on the specification of your machine and requires about 100GB RAM. For IVFSQ as described in the paper, you can use `index-add` and `index-merge` to distribute the addition of phrase vectors to the index.

You also need to compress the metadata (saved in hdf5 files together with phrase vectors) for a faster inference of DensePhrases. This is mandatory for the IVFOPQ index.
```bash
# Compress metadata of wiki-dev
make compress-meta DUMP_DIR=$SAVE_DIR/densephrases-multi_wiki-dev/dump
```

For evaluating the performance of DensePhrases with your phrase indexes, use `eval-index`.
```bash
# Evaluate on the NQ test set questions
make eval-index MODEL_NAME=densephrases-multi DUMP_DIR=$SAVE_DIR/densephrases-multi_wiki-dev/dump/
```

### 3. Query-side fine-tuning
Query-side fine-tuning makes DensePhrases a versatile tool for retrieving phrase-level knowledge given different types of input queries and answers. Although DensePhrases was trained on QA datasets, it can be adapted to non-QA style inputs such as "subject [SEP] relation" where we expect related object entities to be retrieved. It also significantly improves the performance on QA datasets by reducing the discrepancy of training and inference.

First, you need a phrase index for the full Wikipedia (`wiki-20181220`), which can be simply downloaded [here](#3-phrase-index), or a custom phrase index as described above.
Given your query-answer pairs pre-processed as json files in `$DATA_DIR/open-qa` or `$DATA_DIR/kilt`, you can easily query-side fine-tune your model. For instance, the training set of T-REx (`$DATA_DIR/kilt/trex/trex-train-kilt_open_10000.json`) looks as follows:
```
{
    "data": [
        {
            "id": "111ed80f-0a68-4541-8652-cb414af315c5",
            "question": "Effie Germon [SEP] occupation",
            "answers": [
                "actors",
                "actor",
                "actress",
                "actresses"
            ]
        },
        ...
    ]
}
```
The following command query-side fine-tunes `densephrases-multi` on T-REx.
```bash
# Query-side fine-tune on T-REx (model will be saved as MODEL_NAME)
make train-query MODEL_NAME=densephrases-multi-query-trex DUMP_DIR=$SAVE_DIR/densephrases-multi_wiki-20181220/dump/
```
Note that the pre-trained query encoder is specified in `train-query` as `--query_encoder_path $(SAVE_DIR)/densephrases-multi` and a new model will be saved as `densephrases-multi-query-trex` as specified in `MODEL_NAME`. You can also train on different datasets by changing the dependency `trex-open-data` to `*-open-data` (e.g., `wq-open-data` for WebQuestions).

#### IVFOPQ vs IVFSQ
Currently, `train-query` uses the IVFOPQ index for query-side fine-tuning, and you should apply minor changes in the code to train with an IVFSQ index.
For IVFOPQ, training takes 2 to 3 hours per epoch for large datasets (NQ, TQA, SQuAD), and 3 to 8 minutes for small datasets (WQ, TREC). We recommend using IVFOPQ since it has similar or better performance than IVFSQ while being much faster than IVFSQ. With IVFSQ, the training time will be highly dependent on the File I/O speed, so using SSDs is recommended for IVFSQ.

### 4. Inference
With any DensePhrases query encoders (e.g., `densephrases-multi-query-nq`) and a phrase index (e.g., `densephrases-multi_wiki-20181220`), you can test your queries as follows and the retrieval results will be saved as a json file with the `--save_pred` option:

```bash
# Evaluate on Natural Questions
make eval-index MODEL_NAME=densephrases-multi-query-nq DUMP_DIR=$SAVE_DIR/densephrases-multi_wiki-20181220/dump/

# If the demo is being served on http://localhost:51997
make eval-demo I_PORT=51997
```
For the evaluation on different datasets, simply change the dependency of `eval-index` (or `eval-demo`) accordingly (e.g., `nq-open-data` to `trec-open-data` for the evaluation on CuratedTREC).
Note that the test set evaluation of slot filling tasks requires prediction files to be uploaded on [eval.ai](https://eval.ai/web/challenges/challenge-page/689/overview) (use `strip-kilt` target in `Makefile` for better accuracy).

## Pre-processing
At the bottom of `Makefile`, we list commands that we used for pre-processing the datasets and Wikipedia. For training question generation models (T5-large), we used [https://github.com/patil-suraj/question\_generation](https://github.com/patil-suraj/question_generation) (see also [here](https://github.com/princeton-nlp/DensePhrases/blob/main/scripts/question_generation/generate_squad.py) for QG). Note that all datasets are already pre-processed including the generated questions, so you do not need to run most of these scripts. For creating test sets for custom (open-domain) questions, see `preprocess-openqa` in `Makefile`.

## Questions?
Feel free to email Jinhyuk Lee `(jl5167@princeton.edu)` for any questions related to the code or the paper. You can also open a Github issue. Please try to specify the details so we can better understand and help you solve the problem.

## Reference
Please cite our paper if you use DensePhrases in your work:
```bibtex
@inproceedings{lee2021learning,
   title={Learning Dense Representations of Phrases at Scale},
   author={Lee, Jinhyuk and Sung, Mujeen and Kang, Jaewoo and Chen, Danqi},
   booktitle={Association for Computational Linguistics (ACL)},
   year={2021}
}
```

## License
Please see LICENSE for details.

[demo]: http://densephrases.korea.ac.kr
