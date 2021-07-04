#!/bin/bash

base_dir=$(cd $(dirname "$0")/.. && pwd)
label="adidas"
dir="2021-07-02"

# copy pickle file
mv ${base_dir}/${data}/${label}_vocab.pkl ${base_dir}/${label}/${dir}/vocab.pkl

# upload all result file
aws s3 sync ${label}/${dir} s3://smartad-dev/seanchuang/model/data/${label}/${dir}
