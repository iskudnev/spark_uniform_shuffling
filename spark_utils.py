from typing import List

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window


KEY_TYPE = "integer"

TYPE_MAPPING = {
    "long": "int64",
    "integer": "int32",
    "short": "int16",
}


def get_key_list(
        spark: SparkSession,
        num_keys: int,
        key_type: str = "INTEGER",
        cardinality: int = 1_000_000,
    ) -> List[int]:
    """
    Generate list of integers to shuffle on
    to separate data into different partitions
    Генерация списка ключей, распределяющихся
    в разные партиции

    Arguments
    _________
    spark: SparkSession object
    num_keys: Number of desired keys for shuffle
    key_type: Any type of integer containing values of result list
    cardinality: Range size of integers to choose keys from

    Returns
    _______
    keys: List of generated keys
    """
    if not isinstance(num_keys, int):
        raise TypeError(
            f"Type of parameter `num_keys` is {type(num_keys)} ,"
            "should be class<int>"
        )
    value_error_condition = (
        key_type.lower() == "byte" and not (num_keys < cardinality < 2 ** 7)
        or key_type.lower() == "short" and not (num_keys < cardinality < 2 ** 15)
        or key_type.lower() in ["int", "integer"] and not (num_keys < cardinality < 2 ** 31)
        or key_type.lower() == "long" and not (num_keys < cardinality < 2 ** 63)
    )
    if value_error_condition:
        raise ValueError(
            "Check num_keys, key_type and cardinality params"
        )

    win_rn = (Window
        .partitionBy("mod")
        .orderBy("id")
    )
    keys = (spark
        .range(
            cardinality,
            numPartitions=20
        )
        .select(
            F.col("id").cast(key_type)
        )
        .select(
            F.col("id"),

            F.when(
                F.hash("id") % num_keys >= 0,                  
                F.hash("id") % num_keys
            ).otherwise(
                F.hash("id") % num_keys + num_keys
            ).alias("mod"),
        )
        .select(
            F.col("id"),
            F.row_number().over(win_rn).alias("rn"),
        )
        .where(
            F.col("rn") == 1
        )
        .select(
            F.col("id"),
        )
        .distinct()
        .rdd.map(
            lambda row: row["id"]
        )
        .collect()
    )
    assert len(keys) == num_keys, (
        f"Not enough keys (requested keys = {num_keys}, "
        f"generated keys = {len(keys)}), increase cardinality "
        "or change key_type params"
    )
    return keys
