from enum import Enum


class TimeUnit(Enum):
    """时间单元

    Args:
        Enum (_type_): 用于计算时间戳
    """
    SECOND = 1
    MINUTE = 60
    HOUR = 60 * 60
    DAY = 60*60*24
    WEEK = 60 * 60 * 24 * 7
    MONTH = 60 * 60 * 24 * 30
    TWO_MONTH = 60 * 60 * 24 * 60
    YEARS = 60 * 60 * 24 * 30 * 12