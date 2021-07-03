with data as (
    select 
        *
    from hive_ad.datafeed.daily_i2i_data_raw
    where label='{label}'
        and dt > date_format(date_add('day', -{recency}, date('{dt}')), '%Y-%m-%d')
        and dt <= '{dt}'
), p_data as (
    select 
        user_id,
        count(*) as user_count
    from data
    group by 1
), p_data_stddev as (
    select
        user_id,
        user_count,
        (user_count - avg(user_count) over ())
            / (stddev(user_count) over ()) as zscore
    from p_data
), user_list as (
    select
        user_id
    from p_data_stddev a
    where abs(zscore) <= 10
        and user_count >= 3
), item_list as (
    select 
        content_id
    from data
    inner join user_list using (user_id)
    group by 1
    having count(*) >= 3
)
select 
    user_id,
    content_id as item_id
from data
inner join user_list using (user_id)
inner join item_list using (content_id)
order by user_id, ts
