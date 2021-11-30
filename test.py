from tqdm import tqdm
import random, time
from lid.loggers.logger import Logger


logger = Logger()
with tqdm(enumerate(range(100)), total=100, desc="训练中") as tbar:
    # for i, batch in tbar:
    #     tbar.set_postfix({
    #         'loss': random.randint(1,4),
    #         'acc': random.randint(1,2)
    #     })
    #     time.sleep(0.5)
    #     tbar.set_postfix({
    #         'loss2': random.randint(1,4),
    #         'acc2': random.randint(1,2)
    #     })
    #     time.sleep(0.5)
    for i, batch in tbar:
        logger.log({"loss": random.randint(1,4),'acc': random.randint(1,2)},
                progress=True, tbar=tbar)
        time.sleep(0.5)
        logger.log({"loss2": random.randint(1,4),'acc2': random.randint(1,2)},
                progress=True, tbar=tbar)