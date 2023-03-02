# Scripts

This README provides instruction on how to use the scripts in this directory for
data pre-processing and FastLAS learning tasks.

## Data pre-processing

You will need parse the raw arff files of TMC2007-500 before running the data
pre-processing. To do so, run `arff_data_parsing.sh` which will parse the file
with `arff_parse.py`. We provide an example shell script called `arff_data_parsing_example`.
To reproduce the same train test split, the random seed needs to be 73. 

Once you have the parsed files, you can run `data_preprocess.sh` (we provide an
example `data_process_example`). Please set the arg `-m` (mode) to `mi` so that
the pre-processing is based on mutual information. The other key args are `-nl`
(number of labels), `-mit` (MI value threshold). You can follow the below table
to generate the pre-processed subsets/dataset that we use in our paper. Note
that during pre-processing some samples after will have 0 for all selected
attributes, such samples will be discarded. The output of the script will print
out howe many such examples are skipped.

| Dataset | Attributes (after pre-processing) | Labels | MI threshold | Train | Val  | Test | Total |
|---------|-----------------------------------|--------|--------------|-------|------|------|-------|
| TMC-3   | 59                                | 3      | 0.01         | 10331 | 2582 | 4034 | 16947 |
| TMC-5   | 60                                | 5      | 0.02         | 11301 | 2849 | 4415 | 18565 |
| TMC-10  | 34                                | 10     | 0.04         | 12878 | 3222 | 5354 | 21454 |
| TMC-15  | 80                                | 15     | 0.04         | 14616 | 3649 | 6053 | 24318 |
| TMC-22  | 103                               | 22     | 0.05         | 17185 | 4292 | 7061 | 28538 |

## FastLAS learning tasks

You will need to install [FastLAS](https://github.com/spike-imperial/FastLAS/releases)
and clingo first.
We provide a `las_prep_example` that calls `las_gen.py` to generate mode biases
and learning examples for a LAS task. The script takes pre-processed dataset and
will dump the `.las` files into a directory (you need to create one first before
pass it in as an arg). For example, after running `las_prep.sh` on CUB-3 subset,
the output background file (that has mode biases, but no actual background
knowledge) is `bk_t3.las` and the examples file is `example_t3.las`, to learn
with FastLAS, run:

```
FastLAS --opl bk_t3.las example_t3.las
```

FastLAS may be killed when learning because of memory issue. In our case, 16G of
RAM is not enough for FastLAS to learn in any of the subsets/dataset.
