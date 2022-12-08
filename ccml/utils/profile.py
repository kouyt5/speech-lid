from collections import defaultdict
import functools
import threading
import logging
import time


class TimeCostRecoder(object):
    _instance_lock = threading.Lock()

    def __init__(self):
        self.values_map = defaultdict(float)
        self.count_map = defaultdict(int)

    def __new__(cls, *args, **kwargs):
        if not hasattr(TimeCostRecoder, "_instance"):
            with TimeCostRecoder._instance_lock:
                if not hasattr(TimeCostRecoder, "_instance"):
                    TimeCostRecoder._instance = object.__new__(cls)
        return TimeCostRecoder._instance

    def format_print(self):
        sorted_values_list = sorted(self.values_map.items(), key = lambda x:x[1], reverse=True)
        for item in sorted_values_list:
            key = item[0]
            total_time = self.values_map[key]
            avg_time = total_time / self.count_map[key]
            logging.info(
                f"函数名:{key} total time -> {total_time:.2f}s, avg -> {avg_time*1000:.2f}ms, count -> {self.count_map[key]}"
            )
            self.values_map[key] = 0.0
        self._clear()
    
    def _clear(self):
        self.values_map = defaultdict(float)
        self.count_map = defaultdict(int)
        
    def recoder(self, key:str, duration:float):
        self.values_map[key] += duration
        self.count_map[key] += 1


_time_cost_recoder = TimeCostRecoder()


def register_cost_statistic(
    need_return: bool = False,
):
    def decorator(func):
        key = func.__name__
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            if need_return:
                data = func(*args, **kwargs)
                end_time = time.time()
                _time_cost_recoder.values_map[key] += end_time - start_time
                _time_cost_recoder.count_map[key] += 1
                return data
            else:
                func(*args, **kwargs)
                end_time = time.time()
                _time_cost_recoder.values_map[key] += end_time - start_time
                _time_cost_recoder.count_map[key] += 1

        return wrapper

    return decorator

@register_cost_statistic(need_return=False)
def test():
    time.sleep(0.2)
    
@register_cost_statistic(need_return=True)
def test2():
    time.sleep(0.3)
    
if __name__ == "__main__":
    logging.warning("ggg")
    test()
    test2()
    test()
    test()
    _time_cost_recoder.format_print()
    _time_cost_recoder.format_print()
    