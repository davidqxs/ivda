# Datasets

The datasets are not redistributed here; place them as follows before running
the experiments. Checksums are of the exact files used for the paper's results.

## NSL-KDD

Source: Canadian Institute for Cybersecurity, University of New Brunswick
(https://www.unb.ca/cic/datasets/nsl.html).

## UNSW-NB15

Source: UNSW Canberra (https://research.unsw.edu.au/projects/unsw-nb15-dataset).
Download `UNSW_NB15_training-set.csv` and save it as `data/unsw/train.csv`.
Only the training-set CSV is used; the script performs its own stratified split.

## Expected layout and SHA-256

```
data/nslkdd/KDDTrain+.txt
  1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95  (NSL-KDD training file)
data/nslkdd/KDDTest+.txt
  fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84  (NSL-KDD test file)
data/unsw/train.csv
  bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa  (UNSW-NB15 training-set CSV (UNSW_NB15_training-set.csv, renamed))
```
