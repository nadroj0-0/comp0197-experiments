# M5 preprocessing cookbook

This cookbook was built from your uploaded preprocessing outputs and the preprocessing script. It is organized in the same folder order as your output directory.

## 0. Big picture

- Dataset name: **validation**
- Bottom-level series: **30,490**
- Observed history window: **d_1 to d_1913**
- Future rows appended: **56** days
- Feature batches written: **30** files
- Static identifiers repeated inside feature batches: **False**
- Pre-release rows dropped: **True**

Important implication: because `keep_static_in_batches = false`, the feature batch files do **not** contain `item_id`, `dept_id`, `cat_id`, `store_id`, or `state_id`. To recover those static identifiers, join a feature batch to `validation_series_info.pkl.gz` on `id`.

Another important implication: all files in `features/validation_features_batch_*.pkl.gz` share the **same 77-column schema**. The ranges below for the `features/` section are shown from the uploaded `validation_features_batch_000.pkl.gz`, which is representative of the schema.

## 1. Folder: `features/`

### What these files are

Each `validation_features_batch_XXX.pkl.gz` file is a chunk of the long daily panel. Each row is one `(id, day)` observation. The batches exist only for memory reasons; the schema is the same across all 30 files.

### 1.1 Key / time columns

| feature | dtype | observed_range_in_batch_000 | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| id | category | 1024 levels; e.g. HOBBIES_1_001_CA_1_validation, HOBBIES_1_002_CA_1_validation, HOBBIES_1_003_CA_1_validation, HOBBIES_1_004_CA_1_validation, HOBBIES_1_005_CA_1_validation | Bottom-level M5 series identifier (item-store pair with validation suffix). | Series key used to group rows into one forecasting stream; join key back to series_info. |
| d | category | 1969 levels; e.g. d_897, d_898, d_899, d_900, d_901 | Original M5 day label (d_1 … d_1969). | Dataset-native time key used for merges and split definitions. |
| d_num | int16 | 1 to 1969 | Numeric day index extracted from d. | Convenient ordered time index for sorting, slicing, and building lags. |
| date | datetime64[ns] | 2011-01-29 to 2016-06-19 | Real calendar date from calendar.csv. | Lets you align external calendars and build time-aware validation windows. |
| wm_yr_wk | int16 | 11101 to 11621 | Walmart retail week identifier. | Needed because prices are stored weekly, not daily. |

### 1.2 Target column

| feature | dtype | observed_range_in_batch_000 | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| sales | float32 | 0 to 294 | Observed unit sales at the bottom level; NaN on future rows. | This is the forecasting target. |

### 1.3 Mask / status columns

| feature | dtype | observed_range_in_batch_000 | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| sales_observed | int8 | 0 to 1 | 1 if sales is historical/observed, 0 if it is a future placeholder row. | Useful to separate supervised history from forecast horizon rows. |
| is_future | int8 | 0 to 1 | 1 if row was appended as a future-known-covariate row, 0 otherwise. | Lets sequence models know which rows are prediction horizon rows. |
| available_for_sale | int8 | 1 to 1 | 1 if the item has already been released in that store by this day. | Prevents treating pre-release periods as real zero demand. |

### 1.4 Lifecycle columns

| feature | dtype | observed_range_in_batch_000 | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| days_since_release | float32 | 0 to 1968 | Days elapsed since the item first appears in sell_prices for that store. | Captures product life-cycle and helps distinguish new launches from mature items. |
| age_since_first_sale | float32 | 0 to 1968 | Days elapsed since the first observed non-zero sale for the series. | Captures adoption stage and maturity of the item-store series. |

### 1.5 Calendar / known-future columns

| feature | dtype | observed_range_in_batch_000 | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| snap | float32 | 0 to 1 | Store-state-specific SNAP indicator merged from the correct state SNAP column. | Captures demand shifts tied to SNAP payment timing, especially useful for food. |
| weekday | category | 7 levels; e.g. Saturday, Sunday, Monday, Tuesday, Wednesday | Weekday name from the calendar. | Encodes weekly retail seasonality. |
| wday | int8 | 1 to 7 | Weekday as integer 1–7. | Compact numeric weekday representation. |
| month | int8 | 1 to 12 | Calendar month number. | Captures within-year seasonality. |
| year | int16 | 2011 to 2016 | Calendar year. | Captures slow-moving drift and trend. |
| week_of_year | int8 | 1 to 53 | ISO week number. | Captures repeating yearly seasonality at weekly frequency. |
| quarter | int8 | 1 to 4 | Calendar quarter number. | Useful for broad seasonal shifts. |
| day_of_month | int8 | 1 to 31 | Day number within month. | Can proxy month-end / pay-cycle effects. |
| day_of_year | int16 | 1 to 366 | Day number within year. | Continuous position in annual seasonality. |
| is_weekend | int8 | 0 to 1 | 1 on Saturday/Sunday, else 0. | Retail demand often differs sharply on weekends. |
| is_month_start | int8 | 0 to 1 | 1 on the first day of a month. | Useful for pay-cycle and stocking patterns. |
| is_month_end | int8 | 0 to 1 | 1 on the last day of a month. | Useful for pay-cycle and end-of-month shopping spikes. |
| is_quarter_start | int8 | 0 to 1 | 1 on the first day of a quarter. | Coarse seasonal / reporting-cycle marker. |
| is_quarter_end | int8 | 0 to 1 | 1 on the last day of a quarter. | Coarse seasonal / reporting-cycle marker. |
| is_year_start | int8 | 0 to 1 | 1 on the first day of a year. | Year boundary indicator. |
| is_year_end | int8 | 0 to 1 | 1 on the last day of a year. | Holiday / year-end spike indicator. |
| is_event_day | int8 | 0 to 1 | 1 if either event slot is non-empty on that date. | Fast holiday / special-event flag. |
| event_name_1_code | int8 | 0 to 30 | Integer code for the primary named event. | Categorical event identity without keeping long strings. |
| event_type_1_code | int8 | 0 to 4 | Integer code for the primary event type. | Coarser event grouping (sporting, cultural, national, religious). |
| event_name_2_code | int8 | 0 to 4 | Integer code for the secondary named event. | Captures overlapping/secondary events. |
| event_type_2_code | int8 | 0 to 2 | Integer code for the secondary event type. | Coarser grouping for secondary events. |
| month_sin | float32 | -1 to 1 | Sine transform of month. | Cyclical encoding avoids fake discontinuity between Dec and Jan. |
| month_cos | float32 | -1 to 1 | Cosine transform of month. | Pairs with month_sin for cyclical month seasonality. |
| wday_sin | float32 | -0.974928 to 0.974928 | Sine transform of weekday. | Cyclical weekly encoding. |
| wday_cos | float32 | -0.900969 to 1 | Cosine transform of weekday. | Pairs with wday_sin for cyclical weekly encoding. |

### 1.6 Price columns

| feature | dtype | observed_range_in_batch_000 | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| sell_price | float32 | 0.01 to 30.98 | Current weekly sell price for the item in that store. | Direct micro-demand driver and main price-response feature. |
| price_lag_1w | float32 | 0.01 to 30.98 | One-week lagged sell price. | Lets models measure recent price changes. |
| price_change_1w | float32 | -15.97 to 15.97 | Absolute price change vs previous week. | Captures discount/repricing shocks. |
| price_pct_change_1w | float32 | -0.995951 to 246 | Percent price change vs previous week. | Scale-free price shock, closer to elasticity intuition. |
| price_roll_mean_4w | float32 | 0.1 to 30.98 | Mean sell price over the previous 4 weeks. | Reference-price anchor for short-run promotions. |
| price_roll_mean_13w | float32 | 0.1 to 30.98 | Mean sell price over the previous 13 weeks. | Medium-run reference price. |
| price_roll_mean_52w | float32 | 0.204038 to 30.98 | Mean sell price over the previous 52 weeks. | Long-run reference price / annual baseline. |
| price_rel_4w | float32 | 0.00404858 to 5.88 | Current price divided by the prior 4-week average price. | Shows whether price is high/low versus recent norm. |
| price_rel_13w | float32 | 0.00404858 to 3 | Current price divided by the prior 13-week average price. | Medium-run relative price signal. |
| price_rel_52w | float32 | 0.0039039 to 2.59574 | Current price divided by the prior 52-week average price. | Long-run relative price signal. |
| price_rel_cat_store | float32 | 0.00183765 to 6.20994 | Current price divided by same week average price in the same category and store. | Proxy for within-category substitution / relative attractiveness. |
| price_rel_dept_store | float32 | 0.00195012 to 6.0366 | Current price divided by same week average price in the same department and store. | Finer within-department substitution proxy. |
| price_rank_dept_store | float32 | 0.00191939 to 1 | Percentile rank of the current price within the same department-store-week. | Shows whether the item is cheap or premium relative to nearby alternatives. |
| price_change_flag_1w | float32 | 0 to 1 | 1 if the weekly price changed from the prior week. | Simple promotion/repricing indicator. |

### 1.7 Own-series history columns

| feature | dtype | observed_range_in_batch_000 | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| sales_lag_1 | float32 | 0 to 294 | Sales one day ago. | Short-memory autoregressive signal. |
| sales_lag_7 | float32 | 0 to 294 | Sales seven days ago. | Same-weekday seasonal memory. |
| sales_lag_14 | float32 | 0 to 294 | Sales fourteen days ago. | Two-week seasonal memory. |
| sales_lag_28 | float32 | 0 to 294 | Sales twenty-eight days ago. | Same-day-in-4-week cycle; very common in retail forecasting. |
| sales_lag_56 | float32 | 0 to 294 | Sales fifty-six days ago. | Longer seasonal memory. |
| sales_roll_mean_7 | float32 | 0 to 54 | Mean sales over the prior 7 days. | Recent local demand level. |
| sales_roll_mean_28 | float32 | 0 to 54 | Mean sales over the prior 28 days. | Monthly-level recent demand baseline. |
| sales_roll_mean_56 | float32 | 0 to 54 | Mean sales over the prior 56 days. | Longer-run demand baseline. |
| sales_roll_std_28 | float32 | 0 to 55.4898 | Standard deviation of sales over the prior 28 days. | Measures demand volatility / uncertainty. |
| sales_roll_nonzero_rate_28 | float32 | 0 to 1 | Share of prior 28 days with positive sales. | Very useful for intermittent demand and sparsity. |
| sale_occurrence | float32 | 0 to 1 | 1 if today’s sales > 0, 0 if observed sale is zero, NaN on future rows. | Occurrence component for intermittent-demand modeling. |

### 1.8 Hierarchy history columns

| feature | dtype | observed_range_in_batch_000 | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| store_sales_lag_7 | float32 | 0 to 6948 | Store-level total sales lagged 7 days. | Adds broad local demand context beyond one item. |
| store_sales_lag_28 | float32 | 0 to 6948 | Store-level total sales lagged 28 days. | Longer local demand context. |
| store_sales_roll_mean_7 | float32 | 2555.29 to 5152.86 | Store-level 7-day rolling mean sales. | Captures store traffic / broad store demand. |
| store_sales_roll_mean_28 | float32 | 2803.18 to 4773.96 | Store-level 28-day rolling mean sales. | Smoother store-level demand baseline. |
| state_sales_lag_7 | float32 | 5 to 25224 | State-level total sales lagged 7 days. | Captures state-wide demand conditions. |
| state_sales_lag_28 | float32 | 5 to 25224 | State-level total sales lagged 28 days. | Longer state-wide demand context. |
| state_sales_roll_mean_7 | float32 | 9572.29 to 19161.4 | State-level 7-day rolling mean sales. | Smooth state demand context. |
| state_sales_roll_mean_28 | float32 | 10258.7 to 18175.7 | State-level 28-day rolling mean sales. | Smoother state trend. |
| cat_store_sales_lag_7 | float32 | 0 to 1665 | Category-store total sales lagged 7 days. | Captures demand in the item’s broader category within the same store. |
| cat_store_sales_lag_28 | float32 | 0 to 1665 | Category-store total sales lagged 28 days. | Longer category-store demand context. |
| cat_store_sales_roll_mean_7 | float32 | 211.571 to 1154.57 | Category-store 7-day rolling mean sales. | Smooth category-store demand baseline. |
| cat_store_sales_roll_mean_28 | float32 | 270.286 to 1096.82 | Category-store 28-day rolling mean sales. | Longer smooth category-store trend. |
| dept_store_sales_lag_7 | float32 | 0 to 1389 | Department-store total sales lagged 7 days. | More specific local hierarchy context than category level. |
| dept_store_sales_lag_28 | float32 | 0 to 1389 | Department-store total sales lagged 28 days. | Longer department-store context. |
| dept_store_sales_roll_mean_7 | float32 | 6.14286 to 918.714 | Department-store 7-day rolling mean sales. | Smooth local department demand baseline. |
| dept_store_sales_roll_mean_28 | float32 | 10.5357 to 866.286 | Department-store 28-day rolling mean sales. | Longer department-store trend. |

### Practical reading notes for `features/`

- `sales` is the target on observed rows and is `NaN` on appended future rows.
- `available_for_sale` is constant `1` in your exported feature batches because you ran the script with `drop_pre_release = true`; rows before release were removed.
- `event_name_*` and `event_type_*` string columns were dropped from the feature batches to save space; only integer codes remain here.
- Explicit lag / rolling features are **past-only** features. They are strongest for tree models and simple direct baselines; sequence models such as DeepAR can often rely more on the raw target history.

## 2. Folder: `metadata/`

### `validation_event_code_maps.json`

This file is the lookup table from integer event codes back to human-readable event labels. Use it whenever you need to interpret `event_name_1_code`, `event_type_1_code`, `event_name_2_code`, or `event_type_2_code`.

| field | n_codes | examples |
|---|---|---|
| event_name_1 | 31 | None→0, SuperBowl→1, ValentinesDay→2, PresidentsDay→3, LentStart→4, LentWeek2→5, StPatricksDay→6, Purim End→7 |
| event_type_1 | 5 | None→0, Sporting→1, Cultural→2, National→3, Religious→4 |
| event_name_2 | 5 | None→0, Easter→1, Cinco De Mayo→2, OrthodoxEaster→3, Father's day→4 |
| event_type_2 | 3 | None→0, Cultural→1, Religious→2 |

### `validation_preprocessing_summary.json`

This file is the run log for the preprocessing job. It tells you which raw files were used, how many series and batches were created, whether future rows were appended, whether pre-release rows were dropped, and the final feature list in the batch files.

| field | value |
|---|---|
| dataset_name | validation |
| n_series | 30490 |
| start_d | 1 |
| end_d | 1913 |
| future_days_added | 56 |
| batch_size | 1024 |
| output_format | pickle |
| drop_pre_release | True |
| keep_static_in_batches | False |
| n_batches | 30 |

### `validation_recommended_splits.json`

This file gives recommended rolling 28-day splits for model validation on the validation dataset.

| split | train_window | validation_window | predict_window |
|---|---|---|---|
| fold_1 | d_1 to d_1857 | d_1858 to d_1885 |  |
| fold_2 | d_1 to d_1885 | d_1886 to d_1913 |  |
| holdout | d_1 to d_1913 |  | d_1914 to d_1941 |

## 3. Folder: `static/`

The `static/` folder contains tables that are either truly static (one row per series/day/week) or shared lookup tables used to build the batches. These tables are also the place to recover information that was intentionally omitted from the feature batches to save memory.

### `validation_calendar_features.pkl.gz`

One row per calendar day. This is the master known-future calendar table used to merge date, event, SNAP, and cyclical features into the daily panel.

| feature | dtype | observed_range_in_file | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| d | object | 1969 levels; e.g. d_1, d_2, d_3, d_4, d_5 | Original M5 day label (d_1 … d_1969). | Dataset-native time key used for merges and split definitions. |
| d_num | int16 | 1 to 1969 | Numeric day index extracted from d. | Convenient ordered time index for sorting, slicing, and building lags. |
| date | datetime64[ns] | 2011-01-29 to 2016-06-19 | Real calendar date from calendar.csv. | Lets you align external calendars and build time-aware validation windows. |
| wm_yr_wk | int16 | 11101 to 11621 | Walmart retail week identifier. | Needed because prices are stored weekly, not daily. |
| weekday | category | 7 levels; e.g. Saturday, Sunday, Monday, Tuesday, Wednesday | Weekday name from the calendar. | Encodes weekly retail seasonality. |
| wday | int8 | 1 to 7 | Weekday as integer 1–7. | Compact numeric weekday representation. |
| month | int8 | 1 to 12 | Calendar month number. | Captures within-year seasonality. |
| year | int16 | 2011 to 2016 | Calendar year. | Captures slow-moving drift and trend. |
| week_of_year | int8 | 1 to 53 | ISO week number. | Captures repeating yearly seasonality at weekly frequency. |
| quarter | int8 | 1 to 4 | Calendar quarter number. | Useful for broad seasonal shifts. |
| day_of_month | int8 | 1 to 31 | Day number within month. | Can proxy month-end / pay-cycle effects. |
| day_of_year | int16 | 1 to 366 | Day number within year. | Continuous position in annual seasonality. |
| is_weekend | int8 | 0 to 1 | 1 on Saturday/Sunday, else 0. | Retail demand often differs sharply on weekends. |
| is_month_start | int8 | 0 to 1 | 1 on the first day of a month. | Useful for pay-cycle and stocking patterns. |
| is_month_end | int8 | 0 to 1 | 1 on the last day of a month. | Useful for pay-cycle and end-of-month shopping spikes. |
| is_quarter_start | int8 | 0 to 1 | 1 on the first day of a quarter. | Coarse seasonal / reporting-cycle marker. |
| is_quarter_end | int8 | 0 to 1 | 1 on the last day of a quarter. | Coarse seasonal / reporting-cycle marker. |
| is_year_start | int8 | 0 to 1 | 1 on the first day of a year. | Year boundary indicator. |
| is_year_end | int8 | 0 to 1 | 1 on the last day of a year. | Holiday / year-end spike indicator. |
| is_event_day | int8 | 0 to 1 | 1 if either event slot is non-empty on that date. | Fast holiday / special-event flag. |
| event_name_1 | category | 31 levels; e.g. None, SuperBowl, ValentinesDay, PresidentsDay, LentStart | Primary event name as string. | Human-readable event label; mostly useful for debugging / EDA. |
| event_type_1 | category | 5 levels; e.g. None, Sporting, Cultural, National, Religious | Primary event type as string. | Human-readable event type. |
| event_name_2 | category | 5 levels; e.g. None, Easter, Cinco De Mayo, OrthodoxEaster, Father's day | Secondary event name as string. | Human-readable secondary event label. |
| event_type_2 | category | 3 levels; e.g. None, Cultural, Religious | Secondary event type as string. | Human-readable secondary event type. |
| event_name_1_code | int8 | 0 to 30 | Integer code for the primary named event. | Categorical event identity without keeping long strings. |
| event_type_1_code | int8 | 0 to 4 | Integer code for the primary event type. | Coarser event grouping (sporting, cultural, national, religious). |
| event_name_2_code | int8 | 0 to 4 | Integer code for the secondary named event. | Captures overlapping/secondary events. |
| event_type_2_code | int8 | 0 to 2 | Integer code for the secondary event type. | Coarser grouping for secondary events. |
| snap_CA | int8 | 0 to 1 | California SNAP flag from calendar. | Used internally to build store-specific snap. |
| snap_TX | int8 | 0 to 1 | Texas SNAP flag from calendar. | Used internally to build store-specific snap. |
| snap_WI | int8 | 0 to 1 | Wisconsin SNAP flag from calendar. | Used internally to build store-specific snap. |
| month_sin | float32 | -1 to 1 | Sine transform of month. | Cyclical encoding avoids fake discontinuity between Dec and Jan. |
| month_cos | float32 | -1 to 1 | Cosine transform of month. | Pairs with month_sin for cyclical month seasonality. |
| wday_sin | float32 | -0.974928 to 0.974928 | Sine transform of weekday. | Cyclical weekly encoding. |
| wday_cos | float32 | -0.900969 to 1 | Cosine transform of weekday. | Pairs with wday_sin for cyclical weekly encoding. |

### `validation_price_features.pkl.gz`

One row per item-store-week. This is the master price feature table that turns weekly sell_prices into interpretable price dynamics and relative-price signals.

| feature | dtype | observed_range_in_file | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| store_id | category | 10 levels; e.g. CA_1, CA_2, CA_3, CA_4, TX_1 | Store identifier. | Static location identity. |
| item_id | category | 3049 levels; e.g. FOODS_1_001, FOODS_1_002, FOODS_1_003, FOODS_1_004, FOODS_1_005 | Anonymized product identifier. | Static categorical identity for embeddings / grouping. |
| wm_yr_wk | int16 | 11101 to 11621 | Walmart retail week identifier. | Needed because prices are stored weekly, not daily. |
| sell_price | float32 | 0.01 to 107.32 | Current weekly sell price for the item in that store. | Direct micro-demand driver and main price-response feature. |
| release_wm_yr_wk | int16 | 11101 to 11603 | First Walmart week in which the item-store pair appears in sell_prices. | Defines product release timing and availability. |
| release_d | float32 | 1 to 1842 | First daily d_num associated with release_wm_yr_wk. | Daily version of release timing used in the batches. |
| price_lag_1w | float32 | 0.01 to 107.32 | One-week lagged sell price. | Lets models measure recent price changes. |
| price_change_1w | float32 | -94.86 to 94.86 | Absolute price change vs previous week. | Captures discount/repricing shocks. |
| price_pct_change_1w | float32 | -0.998886 to 897 | Percent price change vs previous week. | Scale-free price shock, closer to elasticity intuition. |
| price_roll_mean_4w | float32 | 0.07 to 83.605 | Mean sell price over the previous 4 weeks. | Reference-price anchor for short-run promotions. |
| price_roll_mean_13w | float32 | 0.1 to 61.46 | Mean sell price over the previous 13 weeks. | Medium-run reference price. |
| price_roll_mean_52w | float32 | 0.1 to 30.98 | Mean sell price over the previous 52 weeks. | Long-run reference price / annual baseline. |
| price_rel_4w | float32 | 0.00111359 to 71 | Current price divided by the prior 4-week average price. | Shows whether price is high/low versus recent norm. |
| price_rel_13w | float32 | 0.00111359 to 10.28 | Current price divided by the prior 13-week average price. | Medium-run relative price signal. |
| price_rel_52w | float32 | 0.00113781 to 10.28 | Current price divided by the prior 52-week average price. | Long-run relative price signal. |
| price_rel_cat_store | float32 | 0.00183642 to 19.4635 | Current price divided by same week average price in the same category and store. | Proxy for within-category substitution / relative attractiveness. |
| price_rel_dept_store | float32 | 0.00157258 to 17.7013 | Current price divided by same week average price in the same department and store. | Finer within-department substitution proxy. |
| price_rank_dept_store | float32 | 0.00121507 to 1 | Percentile rank of the current price within the same department-store-week. | Shows whether the item is cheap or premium relative to nearby alternatives. |
| price_change_flag_1w | int8 | 0 to 1 | 1 if the weekly price changed from the prior week. | Simple promotion/repricing indicator. |

### `validation_series_info.pkl.gz`

One row per bottom-level series. This is the static identity table for joining item/store/category/state IDs and numeric indices back onto the feature batches.

| feature | dtype | observed_range_in_file | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| id | object | 30490 levels; e.g. HOBBIES_1_001_CA_1_validation, HOBBIES_1_002_CA_1_validation, HOBBIES_1_003_CA_1_validation, HOBBIES_1_004_CA_1_validation, HOBBIES_1_005_CA_1_validation | Bottom-level M5 series identifier (item-store pair with validation suffix). | Series key used to group rows into one forecasting stream; join key back to series_info. |
| item_id | category | 3049 levels; e.g. HOBBIES_1_001, HOBBIES_1_002, HOBBIES_1_003, HOBBIES_1_004, HOBBIES_1_005 | Anonymized product identifier. | Static categorical identity for embeddings / grouping. |
| dept_id | category | 7 levels; e.g. HOBBIES_1, HOBBIES_2, HOUSEHOLD_1, HOUSEHOLD_2, FOODS_1 | Department identifier. | Useful static hierarchy level and grouping. |
| cat_id | category | 3 levels; e.g. HOBBIES, HOUSEHOLD, FOODS | Category identifier. | Useful broad product family identifier. |
| store_id | category | 10 levels; e.g. CA_1, CA_2, CA_3, CA_4, TX_1 | Store identifier. | Static location identity. |
| state_id | category | 3 levels; e.g. CA, TX, WI | State identifier (CA, TX, WI). | Static geography identity. |
| id_idx | int16 | 0 to 30489 | Integer index for id. | Convenient numeric mapping for embeddings or joins. |
| item_idx | int16 | 0 to 3048 | Integer index for item_id. | Embedding-ready item code. |
| dept_idx | int8 | 0 to 6 | Integer index for dept_id. | Embedding-ready department code. |
| cat_idx | int8 | 0 to 2 | Integer index for cat_id. | Embedding-ready category code. |
| store_idx | int8 | 0 to 9 | Integer index for store_id. | Embedding-ready store code. |
| state_idx | int8 | 0 to 2 | Integer index for state_id. | Embedding-ready state code. |
| first_sale_d | float32 | 1 to 1846 | First day with positive historical sales. | Series launch/adoption timing. |

### `validation_series_stats.pkl.gz`

One row per bottom-level series. This is a series summary table with scale and intermittency statistics computed from the history window.

| feature | dtype | observed_range_in_file | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| id | object | 30490 levels; e.g. HOBBIES_1_001_CA_1_validation, HOBBIES_1_002_CA_1_validation, HOBBIES_1_003_CA_1_validation, HOBBIES_1_004_CA_1_validation, HOBBIES_1_005_CA_1_validation | Bottom-level M5 series identifier (item-store pair with validation suffix). | Series key used to group rows into one forecasting stream; join key back to series_info. |
| series_total_sales | float32 | 10 to 250502 | Sum of historical sales over the observed window. | Overall scale of the series; useful for normalization and filtering. |
| series_mean_sales | float32 | 0.00522739 to 130.947 | Average daily sales over the observed window. | Series-level scale summary. |
| series_std_sales | float32 | 0.0821549 to 108.583 | Standard deviation of daily sales over the observed window. | Series-level volatility summary. |
| series_zero_rate | float32 | 0.00156822 to 0.996341 | Fraction of observed days with zero sales. | Key intermittency summary. |
| series_nonzero_count | int16 | 7 to 1910 | Number of days with positive sales. | Shows how sparse or active the series is. |
| series_mean_nonzero_sales | float32 | 1 to 161.198 | Average sales conditional on sales being positive. | More informative scale summary for intermittent series. |
| first_sale_d | float32 | 1 to 1846 | First day with positive historical sales. | Series launch/adoption timing. |
| last_sale_d | float32 | 251 to 1913 | Last day with positive historical sales. | Shows how recently the series sold. |

### `validation_store_aggregate_features.pkl.gz`

One row per store-day. Contains lagged and rolling store-total demand features.

| feature | dtype | observed_range_in_file | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| store_id | category | 10 levels; e.g. CA_1, CA_2, CA_3, CA_4, TX_1 | Store identifier. | Static location identity. |
| d | category | 1913 levels; e.g. d_1, d_2, d_3, d_4, d_5 | Original M5 day label (d_1 … d_1969). | Dataset-native time key used for merges and split definitions. |
| store_sales_lag_7 | float32 | 0 to 9338 | Store-level total sales lagged 7 days. | Adds broad local demand context beyond one item. |
| store_sales_lag_28 | float32 | 0 to 9338 | Store-level total sales lagged 28 days. | Longer local demand context. |
| store_sales_roll_mean_7 | float32 | 777.714 to 7673.29 | Store-level 7-day rolling mean sales. | Captures store traffic / broad store demand. |
| store_sales_roll_mean_28 | float32 | 1327.18 to 7011.07 | Store-level 28-day rolling mean sales. | Smoother store-level demand baseline. |

### `validation_state_aggregate_features.pkl.gz`

One row per state-day. Contains lagged and rolling state-total demand features.

| feature | dtype | observed_range_in_file | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| state_id | category | 3 levels; e.g. CA, TX, WI | State identifier (CA, TX, WI). | Static geography identity. |
| d | category | 1913 levels; e.g. d_1, d_2, d_3, d_4, d_5 | Original M5 day label (d_1 … d_1969). | Dataset-native time key used for merges and split definitions. |
| state_sales_lag_7 | float32 | 1 to 25224 | State-level total sales lagged 7 days. | Captures state-wide demand conditions. |
| state_sales_lag_28 | float32 | 1 to 25224 | State-level total sales lagged 28 days. | Longer state-wide demand context. |
| state_sales_roll_mean_7 | float32 | 5112 to 19161.4 | State-level 7-day rolling mean sales. | Smooth state demand context. |
| state_sales_roll_mean_28 | float32 | 5622.96 to 18175.7 | State-level 28-day rolling mean sales. | Smoother state trend. |

### `validation_cat_store_aggregate_features.pkl.gz`

One row per category-store-day. Contains lagged and rolling category-store demand features.

| feature | dtype | observed_range_in_file | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| cat_id | category | 3 levels; e.g. HOBBIES, HOUSEHOLD, FOODS | Category identifier. | Useful broad product family identifier. |
| store_id | category | 10 levels; e.g. CA_1, CA_2, CA_3, CA_4, TX_1 | Store identifier. | Static location identity. |
| d | category | 1913 levels; e.g. d_1, d_2, d_3, d_4, d_5 | Original M5 day label (d_1 … d_1969). | Dataset-native time key used for merges and split definitions. |
| cat_store_sales_lag_7 | float32 | 0 to 6488 | Category-store total sales lagged 7 days. | Captures demand in the item’s broader category within the same store. |
| cat_store_sales_lag_28 | float32 | 0 to 6488 | Category-store total sales lagged 28 days. | Longer category-store demand context. |
| cat_store_sales_roll_mean_7 | float32 | 66.7143 to 5353.14 | Category-store 7-day rolling mean sales. | Smooth category-store demand baseline. |
| cat_store_sales_roll_mean_28 | float32 | 108.821 to 4766.39 | Category-store 28-day rolling mean sales. | Longer smooth category-store trend. |

### `validation_dept_store_aggregate_features.pkl.gz`

One row per department-store-day. Contains lagged and rolling department-store demand features.

| feature | dtype | observed_range_in_file | what_it_is | why_it_is_useful |
|---|---|---|---|---|
| dept_id | category | 7 levels; e.g. HOBBIES_1, HOBBIES_2, HOUSEHOLD_1, HOUSEHOLD_2, FOODS_1 | Department identifier. | Useful static hierarchy level and grouping. |
| store_id | category | 10 levels; e.g. CA_1, CA_2, CA_3, CA_4, TX_1 | Store identifier. | Static location identity. |
| d | category | 1913 levels; e.g. d_1, d_2, d_3, d_4, d_5 | Original M5 day label (d_1 … d_1969). | Dataset-native time key used for merges and split definitions. |
| dept_store_sales_lag_7 | float32 | 0 to 5118 | Department-store total sales lagged 7 days. | More specific local hierarchy context than category level. |
| dept_store_sales_lag_28 | float32 | 0 to 5118 | Department-store total sales lagged 28 days. | Longer department-store context. |
| dept_store_sales_roll_mean_7 | float32 | 0 to 4364.86 | Department-store 7-day rolling mean sales. | Smooth local department demand baseline. |
| dept_store_sales_roll_mean_28 | float32 | 1.71429 to 3803.54 | Department-store 28-day rolling mean sales. | Longer department-store trend. |

## 4. Which features are useful for which model?

### 4.1 Quick rule of thumb

- **Tree / tabular models** want explicit lag, rolling, price, and hierarchy features.
- **Sequence models** want a clean separation into static features, known-future features, and observed-past features.
- **DeepAR-style models** need fewer hand-engineered lag features than LightGBM because the model is already autoregressive.

### DeepAR

**Use these features explicitly:**
- Target/history: sales, id, d_num/date
- Static categorical IDs from series_info: item_idx, dept_idx, cat_idx, store_idx, state_idx
- Known-future calendar: wday/month/week_of_year, event codes, is_weekend, SNAP, cyclical encodings
- Known-future price block if future prices are trusted: sell_price, price_change_1w, price_rel_*

**Optional / lower priority:**
- Explicit sales_lag_* and sales_roll_* are optional because DeepAR already uses autoregressive target history
- Hierarchy lag features are optional extras rather than core inputs

**Why:** DeepAR learns directly from the past target sequence plus static IDs and dynamic covariates; it does not need a large tabular lag feature set to work well.

### TFT

**Use these features explicitly:**
- Static covariates from series_info: item/store/category/state IDs and optionally series_stats
- Known-future covariates: calendar block, SNAP, event codes, future sell_price and price-relative features, availability/lifecycle if known
- Observed-past covariates: sales, sales_lag_*, sales_roll_*, sale_occurrence, hierarchy demand features

**Optional / lower priority:**
- Raw string columns are not needed; use codes/indices instead

**Why:** TFT explicitly separates static, known-future, and observed-past inputs, so this preprocessing layout matches TFT especially well.

### Deterministic LSTM/GRU

**Use these features explicitly:**
- Core sequence: sales history
- Small exogenous block: wday/month/event/SNAP/sell_price
- A few hand-picked lags or rolling means if you use a simpler direct architecture
- Static IDs via embeddings or merged indices from series_info

**Optional / lower priority:**
- Using every engineered feature may overcomplicate a simple recurrent baseline
- Hierarchy aggregates are optional

**Why:** Plain recurrent baselines usually work best with a compact input set; too many engineered columns can make training noisy without big gains.

### Probabilistic LSTM

**Use these features explicitly:**
- Same feature set as deterministic LSTM/GRU
- Plus series_stats for scaling/normalization if useful

**Optional / lower priority:**
- Large tree-style feature blocks are optional

**Why:** Only the output head/loss changes; the covariate logic is the same as for standard recurrent baselines.

### LightGBM / Quantile LightGBM

**Use these features explicitly:**
- Almost all engineered tabular features in feature batches
- Especially: sales_lag_*, sales_roll_*, price features, event codes, SNAP, hierarchy lag/roll features
- Static IDs or encoded versions from series_info

**Optional / lower priority:**
- Raw datetime or raw strings should be encoded numerically

**Why:** Tree models do not infer sequence memory automatically, so explicit lag, rolling, and relative-price features are exactly what they need.

### Econometric weekly regression / PPML / NB

**Use these features explicitly:**
- Weekly target built from daily sales
- sell_price, price_change_1w, price_rel_cat_store or price_rel_dept_store
- SNAP, event codes, store/state/category/dept fixed effects or dummies

**Optional / lower priority:**
- Daily lag stacks and deep-learning-specific hierarchy roll features are less central

**Why:** Econometric models need an interpretable weekly panel with price and calendar drivers, not the full deep-learning feature store.

## 5. Minimal feature registry you can hand to teammates

### Static covariates
- From `validation_series_info.pkl.gz`: `item_idx`, `dept_idx`, `cat_idx`, `store_idx`, `state_idx` (or the string IDs if a library can handle them).
- Optional static summaries from `validation_series_stats.pkl.gz`: `series_mean_sales`, `series_std_sales`, `series_zero_rate`, `series_mean_nonzero_sales`.

### Known-future covariates
- Calendar block: `wday`, `month`, `week_of_year`, `quarter`, `day_of_month`, `is_weekend`, `is_event_day`, event codes, cyclical encodings.
- SNAP and release/lifecycle features that are known ahead of time: `snap`, `available_for_sale`, `days_since_release`.
- Price block if the future weekly prices are treated as known: `sell_price`, `price_change_1w`, `price_pct_change_1w`, `price_rel_*`, `price_rank_dept_store`.

### Observed-past covariates
- Own demand history: `sales`, `sales_lag_*`, `sales_roll_*`, `sale_occurrence`.
- Hierarchy demand context: `store_*`, `state_*`, `cat_store_*`, `dept_store_*`.

## 6. One very important join you will probably need

Because the feature batches do not carry the static IDs, you will often do:

```python
feat = pd.read_pickle('validation_features_batch_000.pkl.gz', compression='gzip')
series_info = pd.read_pickle('validation_series_info.pkl.gz', compression='gzip')
feat = feat.merge(series_info[['id','item_idx','dept_idx','cat_idx','store_idx','state_idx']], on='id', how='left')
```

This is the standard way to give DeepAR / TFT / LSTM / LightGBM access to static series identity.







One key join you will almost certainly need is:

feat = pd.read_pickle("validation_features_batch_000.pkl.gz", compression="gzip")
series_info = pd.read_pickle("validation_series_info.pkl.gz", compression="gzip")

feat = feat.merge(
    series_info[["id", "item_idx", "dept_idx", "cat_idx", "store_idx", "state_idx"]],
    on="id",
    how="left",
)

That is the clean way to give DeepAR, TFT, LSTM/GRU, or LightGBM access to the static identity information that was intentionally left out of the batched feature files to save memory.