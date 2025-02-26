import os, sys, json, copy


size_methods = ["vmlinux", "GZIP-bzImage", "GZIP-vmlinux", "GZIP", "BZIP2-bzImage", 
              "BZIP2-vmlinux", "BZIP2", "LZMA-bzImage", "LZMA-vmlinux", "LZMA", "XZ-bzImage", "XZ-vmlinux", "XZ", 
              "LZO-bzImage", "LZO-vmlinux", "LZO", "LZ4-bzImage", "LZ4-vmlinux", "LZ4"]

def get_possible_configurations(hyperparams_list):
    configs = []
    
    #Check if there is still something to iterate
    if any(hyperparams_list):
        
        #Get the first key in the list of hyperparams
        key = next(iter(hyperparams_list))
        
        #Get all the possible configs without considering the current key
        next_possible_configs = get_possible_configurations({k:v for k,v in hyperparams_list.items() if not k == key})
        
        #Combine every value of the current key to all known possible configurations
        for value in hyperparams_list[key]:
            
            if len(next_possible_configs) > 0:
                for config in next_possible_configs:
                    config[key] = value
                    configs.append(copy.deepcopy(config))
            
            else:
                configs.append({key:value})
    
    return configs

def get_params():
    
    params = {
        "resultsPath":"results/",
        "perf":"vmlinux",
        "nbFolds":10,
        "minSampleSize":100,
        "maxSampleSize":None,
        "paceSampleSize":None,
        "nb_bins":10,
        "nb_yes":1,
        "columns_to_drop":["cid"]+size_methods,
        "algo":"rf"
    }
    
    
    #List all possible hyperparams, in order to avoid error with scikit
    possible_hyperparams = [
        "criterion","splitter","max_features","max_depth","min_samples_split","min_samples_leaf", "min_weight_fraction_leaf","max_leaf_nodes","random_state","min_impurity_decrease","n_estimators",
        "loss","learning_rate","subsample","presort"
    ]

    hyperparams = {}

    hyperparams_list = {

    }

    if os.path.isfile('./config/config.json'):
        with open("./config/config.json","r") as f:
            data = json.load(f)
            for i in data:
                if i in params:
                    params[i] = data[i]
                elif i in possible_hyperparams:
                    if type(data[i]) == list:
                        hyperparams_list[i] = data[i]
                    else:
                        hyperparams[i] = data[i]
    #Get input params
    for k,v in enumerate(sys.argv):
        if v.startswith("--"):
            key = v[2:]
            #json loads handle list and int from input string
            try:
                value = json.loads(sys.argv[k+1])
            except:
                value = sys.argv[k+1]

            if key in params:
                params[key] = value
            elif key in possible_hyperparams:
                if type(value) == list:
                    hyperparams_list[key] = value
                else:
                    hyperparams[key] = value
                    
            else:
                print("param",params,"unknown")

    params["hyperparams"] = hyperparams
    
    return params, hyperparams_list