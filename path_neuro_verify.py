#!/usr/bin/python
import os.path
import os
import sys
import subprocess
import time
import torch.multiprocessing as multiprocessing
from functools import lru_cache
sys.path.insert(0, '/extra/ali/agents/')
from neurouse.models.gat.network import GNN 
from neurouse.data.graph.generator import DataGraphGenerator
import torch
import pprint

TRAIN_WARNINGS = 500
# usage
# python main_remote.py file.json

# out put
# time_out_file_name: all link files and jsons of time out
# memory_out_file_name: all link files and jsons of memory out
# klee_result_file_name: all jsons of klee
# klee_log_file_name: all log of klee. I do not mv those log into one file.

# those variables need you change
home_path = "/extra/ali/IncreLux"
klee_path = home_path+"/KLEE/klee/build/bin/klee"

total_cpu = multiprocessing.cpu_count()
# total_cpu = 10
klee_log_file_name = "confirm_result.log"
klee_result_file_name = "confirm_result.json"

log_file_name = "log.json"

schedule_time = 1  # second
time_out = 15 # second
time_out_file_name = "time_out.json"

# notice: for the reason that python can not kill the klee quickly, it is better to set this small.
memory_out = 2 * 1024 * 1024 * 1024  # byte
memory_out_file_name = "memory_out.json"

right_return_code = 0
klee_error_result_file_name = "error.json"
klee_right_result_file_name = "tested.json"
# if you need change the path in link file
linux_kernel_path_in_json = "/data2/yizhuo/inc-experiment/experiment"
linux_kernel_path_in_this_pc = "/extra/ali/kernel_ir"

klee_right_result = "KLEE: done: generated tests ="

training_data_dir = home_path + "/training_data"

path_enumerator_pass_path = "/extra/ali/kernel_ir/passes/build/path_enumerator/libpath_enumerator.so"

infeasible_warnings = 0

K = 10
top_K = [0 for _ in range(K)]
found = 0

class WarningVerifyProcess(multiprocessing.Process):
    def __init__(self, gnn, json, graph, link_file, index):
        super(WarningVerifyProcess, self).__init__()
        self.gnn = gnn
        self.json = json
        self.graph = graph
        self.link_file = link_file
        self.index = index

    # @classmethod
    def run(self):
        # global found
        verify_warning(self.gnn, self.json, self.graph, self.link_file, self.index)

def verify_warning(gnn=None, json=None, graph=None, link_file=None, index=None):
        if index is not None:
            os.chdir(str(index))
        if (graph is None):
            print("Unable to generate graph")
            return
        error, stdout, stderr = link_files(link_file)
        if (error != 0): 
            print(f"{index if index else ''} Error in linking: Output and errors below...")
            print("--BEGIN-OUTPUT--")
            print(stdout, stderr)
            print("--END-OUTPUT--") 
            return
        error, stdout, stderr = enumerate_paths(json)
        if (error != 0): 
            print(f"{index if index else ''}: Error in enumerating paths. Output and errors below...")
            print("--BEGIN-OUTPUT--")
            print(stdout, stderr)
            print("--END-OUTPUT--")
            return
        # pprint.pprint(json)
        with open("paths.txt") as f:
            paths = f.read()
        # print(paths)
        paths = eval(paths)
        if not paths:
            return
        # link_files.append(link_file)
        items = rank_paths(gnn, paths, graph)
        
        for i in range(min(len(items), K)):
            item = items[i]
            path, rank = item
            # print(rank)
            r, _, _ = verify_single_path(json, path)
            if (r == 69):
                top_K[i] += 1
                # found += 1
                # print(f"Found: {found}")
                break

                
            execute_shell("rm -rf klee-out")
        if index is not None:
            os.chdir("../")
class GraphGeneratorProcess(multiprocessing.Process):
    def __init__(self, link_file, index):
        super(GraphGeneratorProcess, self).__init__()
        self.link_file = link_file
        self.index = index

    def run(self):
        os.chdir(str(self.index))
        DataGraphGenerator.ensure_data_graph_prepared(self.link_file)
        os.chdir("../")


def execute_shell(cmd, debug=False):
    # print(cmd)
    cmd_process = None
    if debug:
        # print(cmd)
        cmd_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = cmd_process.communicate()
        return (cmd_process.returncode, stdout, stderr)
    else:
        # cmd_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cmd_process = subprocess.Popen(cmd, shell=True)
        return (cmd_process.wait(), None, None)


def link_files(link_file):
    link_file = link_file.replace(linux_kernel_path_in_json, linux_kernel_path_in_this_pc)
    bc_list = link_file.replace(":\n", "")
    bc_list = bc_list.split(":")
    link_cmd = home_path + "/llvm/build/bin/llvm-link --only-needed -o " + "./built-in.bc"
    for bc in bc_list:
        link_cmd = link_cmd + " " + bc
    
    print(bc_list)
    return execute_shell(link_cmd)

def enumerate_paths(warning):
    cmd = f"{home_path}/llvm/build/bin/opt -load {path_enumerator_pass_path} -enumerate-paths -cg='cg-414.json' -warning '{warning}' ./built-in.bc"
    return execute_shell(cmd, debug=False)


def read_all_json(file_name):
    f = open(file_name, "a")
    for i in range(total_cpu):
        path_file_name = str(i) + "/" + file_name
        #print(path_file_name)
        if os.path.isfile(path_file_name):
            tf = open(path_file_name, "r")
            f.write(tf.read())
            tf.close()
    f.close()


def rank_paths(gnn, paths, graph):
    global infeasible_warnings
    ranks = torch.sigmoid(gnn(paths, graph))
    # # print(ranks)
    # infeasible = ranks <= 0.5
    # no_feasible = torch.all(infeasible)

    # if no_feasible:
    #     infeasible_warnings += 1
    ranks = ranks.tolist()
    # print(ranks)
    if isinstance(ranks, float):
        ranks = [ranks]
    items = zip(paths, ranks)
    items = sorted(items, key=lambda x: x[1], reverse=True)
    # ranks = 
    return items



@lru_cache(maxsize=None)
def create_data_graph(link_file):
    g = DataGraphGenerator.prepare_or_load_data_graph(link_file)
    return g


def verify_singleton_path(json, path_file):
    cmd = f"{klee_path} -json='{json}' -fjson='cg-414.json' -single-path -single-path-file='{path_file}' -output-dir=klee-out ./built-in.bc"
    # cmd = f"{klee_path} -json='{json}' -fjson='cg-414.json' ./built-in.bc"
    return execute_shell(cmd, debug=False)


def verify_baseline(json):
    cmd = f"{klee_path} -json='{json}' -fjson='cg-414.json' -output-dir=klee-out --max-time=15 ./built-in.bc"
    return execute_shell(cmd, debug=False)

def verify_single_path(json, path):
    singleton_path_file = "singleton_path.txt"
    with open(singleton_path_file, "w") as f:
        f.write('\n'.join(path))
    
    return verify_singleton_path(json, singleton_path_file)


def main():
    # os.chdir("temp")
    # for i in range(total_cpu):
        # execute_shell(f"rm -rf {i}")
    # for i in range(total_cpu):
        # execute_shell(f"mkdir -p {i}")


    # gnn = GNN([32, 16 , 32])
    # sd = torch.load("/extra/ali/agents/state.dict")
    # gnn.load_state_dict(sd)
    # gnn.eval()

    filename = sys.argv[1]
    file = open(filename)

    index = 0
    json_index = 0
    line = file.readline()
    # for _ in range(TRAIN_WARNINGS):
    # counter = {}
    warn_count = 0
    # while line:
    # while warn_count < 400:
    SKIP_WARNINGS = 0
    skipped = 0
    TOTAL_WARNINGS = SKIP_WARNINGS + 100000
    total = 0
    processes = [None for _ in range(total_cpu)]
    # link_files = []
    idx = 0
    # with multiprocessing.Pool() as p:
    t = time.perf_counter()
    # for i in range(TOTAL_WARNINGS):
    i = 0
    # pool = multiprocessing.Pool(32)
    # done = set()
    skipped = 0
    for i in range(TOTAL_WARNINGS):
    # while line:
        # i += 1
        print(f"Warning: {i}")
        json_index = json_index + 1
        link_file = line
        json = file.readline()
        json = json.replace("\n", "")
        line = file.readline()
        x = link_file.strip().split(":")
        if (len(x) > 1750):
            print(len(x))
            link_files(link_file)
            verify_baseline(json)
            break
        # parsed = JSON.loads(json)
        
        # if (skipped < SKIP_WARNINGS):
        #     skipped += 1
        #     continue
        # # g = create_data_graph(link_file)
        # if link_file in done:
        #     continue
        
        # while processes[idx] and processes[idx].is_alive():
            # idx = (idx + 1) % total_cpu
        
        # output, stdout, stderr = link_files(link_file)
        # if (output != 0):
            # print(link_file)
            # print(stdout)
            # print(stderr)

        # p.apply_async(WarningVerifyProcess.run, (gnn, json, link_file, idx))
        # # processes[idx].terminate()
        # if processes[idx]:
            # processes[idx].join()
            # processes[idx].close()
        # WarningVerifyProcess.run(gnn, json, g, link_file)
        # processes[idx] = WarningVerifyProcess(gnn, json, g, link_file, idx)
        # processes[idx] = GraphGeneratorProcess(link_file, index)
        # processes[idx].start()
        # DataGraphGenerator.ensure_data_graph_prepared(link_file, "temp")
        # pool.apply_async(DataGraphGenerator.ensure_data_graph_prepared, (link_file,"temp"))
        # done.add(link_file)
        # p.close()
        # p.join()
    # pool.close()
    # pool.join()
    
    # os.chdir("..")

    # for p in processes:
    #     if p:
    #         p.join()
    #         p.close()

    print(f"Elapsed time: {time.perf_counter()-t} seconds")
    # read_all_json("confirm_result.json")

    print(top_K)
    print(total)
    # for i in range(total_cpu):
        # execute_shell(f"rm -rf {i}")
    

if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")
    main()