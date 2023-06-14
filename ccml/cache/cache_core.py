import functools
import logging
import os
import time
import traceback
import pickle
from typing import Any
from ccml.cache.time_unit import TimeUnit


def cacheable(
    cache_name: str = None,
    cache_key: str = None,
    duration: float = 1.0,
    time_unit: TimeUnit = TimeUnit.HOUR,
    project: str = "default",
    cache_path: str = None,
    cache_extra_param: str = None,
):
    """缓存
    基于pickle序列化, 缓存数据, 建议数据为python基本类型 tuple, dict, str等.
    
    Args:
        cache_name (str, optional): 缓存名字. Defaults to None.
        cache_key (str, optional): 缓存key, 根据函数参数构造cache_name, 优先级比cache_name高
            如果要使用cache_key, 函数调用必须显式声明入参名, 例如:
                fun(ccml='xxx')
                @cacheable(cache_key='ccml')
                def fun(ccml:str):
                    pass
        duration (float, optional): 缓存时长. Defaults to 1.0.
        time_unit (TimeUnit, optional): 缓存时间单元, 秒还是小时等. Defaults to TimeUnit.HOUR.
        project (str, optional): 项目名称. Defaults to "default".
        cache_path (str, optional): 缓存根路径. Defaults to None.

    Returns:
        _type_: decorator
    
    Example:
    >>> from ccml.cache import cacheable
    >>> @cacheable(cache_name="test", duration=1., time_unit=TimeUnit.HOUR)
    >>> def get_data():
    >>>     return {"data1": 1, "data2": 2}
    """
    # 创建缓存根目录
    if cache_path is None:
        cache_path = _get_cache_path(cache_path=cache_path)
    if not os.path.exists(cache_path):
        logging.warning(f"缓存路径不存在, 创建文件夹{cache_path}...")
        os.makedirs(cache_path)
    # 创建缓存项目目录
    cache_path = os.path.join(cache_path, project)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    
    if cache_name is None and cache_key is None:
        logging.warning(f"cache_name and cache_key must be not None")
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if cache_key in kwargs.keys():
                cache_name = kwargs[cache_key]
            if cache_extra_param in kwargs.keys():
                cache_name += str(kwargs[cache_extra_param])
            cache_name = __check_cache_name(cache_name)
            # 缓存失效判断, 以文件是否存在为标准
            # 1. 时间
            file_path = os.path.join(cache_path, cache_name)
            logging.info(f"缓存路径: {file_path}")
            last_modified_time = _get_file_mtime(file_path)
            validity_time = last_modified_time + time_unit.value * duration
            if validity_time < time.time() and os.path.exists(file_path):  # 不在有效期间删除
                os.remove(file_path)
            # 2. ...
            # 存在缓存文件
            if os.path.exists(file_path):
                data = _deserialization(file_path)
                if data is not None:
                    logging.debug(f"using cached data {file_path}")
                    return data
            data = func(*args, **kwargs)
            _serialization(data, file_path)
            return data

        return wrapper

    return decorator


def _get_file_mtime(file_name: str) -> float:
    """获取文件最近修改时间
    如果文件不存在, 返回当前时间

    Args:
        file_name (str): 文件路径

    Returns:
        float: 时间xxx秒
    """
    if not os.path.exists(file_name):
        return time.time()
    return os.path.getmtime(file_name)


def _deserialization(file_path: str):
    if not os.path.exists(file_path):
        logging.warning(f"cache_path {file_path} not exist")
        return None
    data = None
    try:
        data = pickle.load(open(file_path, "rb"))
    except Exception as e:
        logging.error(f"获取{file_path}缓存失败")
        logging.error(e)
        logging.error("\n" + traceback.format_exc())
    return data


def _serialization(data: Any, file_path: str):
    pickle.dump(data, open(file_path, "wb"), protocol=4)


def _get_cache_path(cache_path: str = None) -> None:
    if cache_path is None:
        cache_path = os.path.join(os.environ["HOME"], ".cache", "ccml")
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    return cache_path


def __check_cache_name(cache_name: str):
    if cache_name is None:
        cache_name = ""
    # 替换非法字符
    for c in ["/", "@", " ", "."]:
        cache_name = cache_name.replace(c, "_")
    # 限制长度
    if len(cache_name) >= 255:
        logging.warning(f"cache_name 长度超过255为{len(cache_name)}, 将被截断为{cache_name[:255]}")
        return cache_name[:255]
    if len(cache_name) == 0:
        logging.warning("cache_name 长度为0, 默认使用ccml作为缓存key")
        return "ccml"
    return cache_name
