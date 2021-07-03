create table if not exists z_seanchuang.sasrec_exp (
    user_id string,
    item_id string
)
partitioned by (label string)
row format delimited fields terminated by '\t'
lines terminated by '\n'
stored as textfile
location 's3://smartad-dmp/warehouse/user/seanchuang/sasrec_exp';
