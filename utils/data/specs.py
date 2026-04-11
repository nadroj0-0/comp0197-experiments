FEATURE_SETS = {
    "sales_only": ["sales"],
    "sales_hierarchy": [
        "sales", "state_id_int", "store_id_int", "cat_id_int", "dept_id_int",
    ],
    "sales_hierarchy_dow": [
        "sales", "state_id_int", "store_id_int", "cat_id_int", "dept_id_int", "day_of_week",
    ],
    "sales_yen": [
        "sales", "sell_price", "is_available", "wday",
        "month", "year", "snap_CA", "snap_TX", "snap_WI", "has_event",
    ],
    "sales_yen_hierarchy": [
        "sales", "sell_price", "is_available", "wday",
        "month", "year", "snap_CA", "snap_TX", "snap_WI",
        "has_event", "state_id_int", "store_id_int", "cat_id_int", "dept_id_int",
    ],
}

TARGET_COL = "sales"


def get_feature_cols(feature_set: str) -> list:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. "
                         f"Options: {list(FEATURE_SETS)}")
    return FEATURE_SETS[feature_set]
