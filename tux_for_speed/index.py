from load_dataset import load_dataset
from params import get_params, get_possible_configurations

import traceback

import tux

import time

from threading import Thread,BoundedSemaphore


params, hyperparams_list = get_params()

configs = get_possible_configurations(hyperparams_list)

df = load_dataset(params["nb_yes"])
params["dataset"] = df
params["semaphore"] = None

if not params["algo"] == "rf" :
    limit = os.sysconf('SC_NPROCESSORS_ONLN')
    if "n_jobs" in params["hyperparams"]:
        limit = params["hyperparams"]["n_jobs"]
        del(params["hyperparams"]["n_jobs"])
    params["semaphore"] = BoundedSemaphore(limit)

    
list_ml = []
time_start = time.time()
print(time_start)
if len(configs) > 0:
    for config in configs:
        
        for k,v in config.items():
            params["hyperparams"][k] = v
        
        print("Starting")
        try:
            ml = tux.TuxML(**params)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
        
        if params["semaphore"] is None:
            try:
                ml.start()
            except Exception as e:
                print(traceback.format_exc())
                print(e)
                print("Fails")
        else:
            list_ml.append(Thread(target=ml.start))
            
    for i in list_ml:
        i.start()
    for i in list_ml:
        i.join()
else:
    
        print("Starting")
        try:
             ml = tux.TuxML(**params)
        except Exception as e:
            print(traceback.format_exc())
            print(e)

        try:
            ml.start()
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            print("Fails")
print("End")

time_end = time.time()
print(time_end)
print("Total time : ",time_end - time_start)