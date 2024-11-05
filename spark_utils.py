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


def get_key_list(spark: SparkSession,
                 num_keys: int,
                 key_type: str = KEY_TYPE) -> List[int]:
    """
    Generating key list for uniform shuffle
    Генерация списка ключей для равномерного перемешивания

    Arguments
    _________
    spark: SparkSession object
    num_keys: Number of desired keys for shuffle
    key_type: Any type of integer containing values of result list

    Returns
    _______
    key_list: List of generated keys
    """
    win_spec = (Window
        .partitionBy("mod")
        .orderBy("id")
    )
    
    key_list = (spark
        .range(1_000_000, numPartitions=2)
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
            F.row_number().over(win_spec).alias("rn"),
        )
        .where(
            F.col("rn") == 1
        )
        .rdd.map(
            lambda row: row["id"]
        )
        .collect()
    )

    return key_list