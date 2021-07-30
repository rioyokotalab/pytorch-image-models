"""
save results to wandb
"""
import json
import pprint
def main():
    json_open = open('test_run_2021_07_27_02_01_03_result.json', 'r')
    json_load = json.load(json_open)
    dic = {}
    for k_name, v in json_load.items():
        dic[k_name] = {}
        for data_name, vv in v.items():
            dic[k_name][data_name] = {}
            for sample_num, vvv in vv.items():
                ave = 0
                repeats = 0
                for val in vvv:
                    if val != 0:
                        repeats += 1
                        ave += 1
                if repeats:
                    ave = ave/repeats
                dic[k_name][data_name][sample_num] = ave
    pprint.pprint(dic)
if __name__ == "__main__":
    main()