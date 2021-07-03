with data as (
    select 
        *
    from hive_ad.datafeed.daily_i2i_data_raw
    where label='{label}'
        and dt > date_format(date_add('day', -{recency}, date('{dt}')),'%Y-%m-%d')
        and dt <= '{dt}'
        and length(user_id) = 42
        and content_id != ''
)
select 
    user_id,
    slice(array_agg(content_id order by ts desc), 1, {history_max} * 2) as rtg_item
from data
group by 1