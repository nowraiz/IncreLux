#!/usr/bin/python
import os.path
import os
import sys
import subprocess
import time
import shutil
import signal
import json as JSON
import numpy as np
import psutil
import pickle
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

total_cpu = 48
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

# linux_kernel_path_in_json = "/data2/yizhuo/inc-experiment/experiment/lll-v4.14"
# linux_kernel_path_in_this_pc = "/extra/ali/kernel_ir/lll-v4.14_original"
klee_right_result = "KLEE: done: generated tests ="

training_data_dir = home_path + "/training_data"

path_enumerator_pass_path = "/extra/ali/kernel_ir/passes/build/path_enumerator/libpath_enumerator.so"

infeasible_warnings = 0

affinity = list(range(24)) + list(range(48, 72))
# affinity = [1 if 0 <= i < 24 or 48 <= i < 72 else 0 for i in range(multiprocessing.cpu_count())]

K = 10

def kill_proc_tree(pid, sig=signal.SIGKILL, include_parent=True,
                   timeout=None, on_terminate=None):
    """Kill a process tree (including grandchildren) with signal
    "sig" and return a (gone, still_alive) tuple.
    "on_terminate", if specified, is a callback function which is
    called as soon as a child terminates.
    """
    assert pid != os.getpid(), "won't kill myself"
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    if include_parent:
        children.append(parent)
    for p in children:
        try:
            p.send_signal(sig)
        except psutil.NoSuchProcess:
            pass
    gone, alive = psutil.wait_procs(children, timeout=timeout,
                                    callback=on_terminate)
    return (gone, alive)   

class WarningVerifyProcess(multiprocessing.Process):
    def __init__(self, gnn, json, graph, link_file, index, json_idx):
        super(WarningVerifyProcess, self).__init__()
        self.gnn = gnn
        self.json = json
        self.graph = graph
        self.link_file = link_file
        self.index = index
        self.json_idx = json_idx
        self.success = multiprocessing.Value('i', False) 
        self.path_found = multiprocessing.Value('i', False)
        self.rank = multiprocessing.Value('i', 0)
        self.p = multiprocessing.Value('i', 0)
        self.min_path = multiprocessing.Value('i', 0)
        self.avg_paths = multiprocessing.Value('d', 0)
        self.num_paths = multiprocessing.Value('i', 0)

    # @classmethod
    def run(self):
        # global found
        # self.success.value, self.path_found.value, self.rank.value, self.p.value = \
            # verify_warning(self.gnn, self.json, self.graph, self.link_file, self.index, self.json_idx)
        self.success.value, self.path_found.value, self.rank.value, self.p.value, self.min_path.value, self.avg_paths.value, self.num_paths.value = \
            verify_warning(self.gnn, self.json, None, self.link_file, self.index, None)

def verify_baseline_warning(gnn=None, json=None, graph=None, link_file=None, index=None, json_idx=None):
    if index is not None:
        os.chdir(str(index))

    (error, stdout, stderr), bc_list = link_files(link_file)
    if (error != 0):
        print("error in linking")
        return False, False, 0, 0

    r, _, _ = verify_baseline(json)
    if r == 69:
        shutil.rmtree("klee-out")
        return True, True, 0, 0
    shutil.rmtree("klee-out")
    os.remove("./built-in.bc")
    return True, False, 0, 0

def verify_warning(gnn=None, json=None, graph=None, link_file=None, index=None, json_idx=None):
        if index is not None:
            os.chdir(str(index))

        (error, stdout, stderr), bc_list = link_files(link_file)
        if (error != 0): 
            print(f"{json_idx} Error in linking: Output and errors below...")
            print(f"--BEGIN-OUTPUT--")
            print(stdout.decode("utf-8"), stderr.decode("utf-8"))
            print("--END-OUTPUT--") 
            return False, False
        graph = DataGraphGenerator.prepare_or_load_data_graph_from_bc("./built-in.bc", bc_list)
        if (graph is None):
            print("Unable to generate graph")
            return False, False
        error, stdout, stderr = enumerate_paths(json)
        if (error != 0): 
            print(f"{json_idx}: Error in enumerating paths. Output and errors below...")
            print("--BEGIN-OUTPUT--")
            print(stdout.decode("utf-8"), stderr.decode("utf-8"))
            print("--END-OUTPUT--")
            return False, False
        # pprint.pprint(json)
        # with open("paths.txt") as f:
        #     paths = f.read()
        stdout = stdout.decode("utf-8")
        # print(paths)
        paths = eval(stdout)
        if not paths:
            return False, False
        # link_files.append(link_file)
        items = rank_paths(gnn, paths, graph)
        
        found = False
        feasible = False
        top_rank = 0
        num_paths = 0
        min_path = len(min(paths, key=len))
        num_paths = len(paths)
        avg_paths = sum(list(map(len, paths)))/num_paths
        for i in range(min(len(items), K)):
            item = items[i]
            path, rank = item
            # print(rank)
            r, _, _ = verify_single_path(json, path)
            if (r == 69):
                if not found:
                    top_rank = i + 1
                    feasible = rank >= 0.5
                    

                found = True
                # found += 1
                # print(f"Found: {found}")
                # break
                
            shutil.rmtree("klee-out")

        os.remove("./built-in.bc")
        return True, found, top_rank, feasible, min_path, avg_paths, num_paths

class GraphGeneratorProcess(multiprocessing.Process):
    def __init__(self, link_file, index):
        super(GraphGeneratorProcess, self).__init__()
        self.link_file = link_file
        self.index = index

    def run(self):
        os.chdir(str(self.index))
        DataGraphGenerator.ensure_data_graph_prepared(self.link_file)
        os.chdir("../")


def execute_shell(cmd, debug=True):
    # print(cmd)
    cmd_process = None
    if debug:
        # print(cmd)
        cmd_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.sched_setaffinity(cmd_process.pid, affinity)
        stdout, stderr = cmd_process.communicate()
        return (cmd_process.returncode, stdout, stderr)
    else:
        cmd_process = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # cmd_process = subprocess.Popen(cmd, shell=True)
        os.sched_setaffinity(cmd_process.pid, affinity)
        # cmd_process = subprocess.Popen(cmd, shell=True)
        return (cmd_process.wait(), None, None)


def link_files(link_file):
    link_file = link_file.replace(linux_kernel_path_in_json, linux_kernel_path_in_this_pc)
    bc_list = link_file.replace(":\n", "")
    bc_list = bc_list.split(":")
    link_cmd = home_path + "/llvm/build/bin/llvm-link -o " + "./built-in.bc"
    for bc in bc_list:
        link_cmd = link_cmd + " " + bc
    
    return execute_shell(link_cmd), bc_list 

def enumerate_paths(warning):
    cmd = f"{home_path}/llvm/build/bin/opt -load {path_enumerator_pass_path} -o /dev/null -enumerate-paths -cg='../cg-414.json' -warning '{warning}' ./built-in.bc"
    return execute_shell(cmd, True)


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
    cmd = f"{klee_path} -json='{json}' -fjson='../cg-414.json' -single-path -single-path-file='{path_file}' -output-dir=klee-out ./built-in.bc"
    # cmd = f"{klee_path} -json='{json}' -fjson='cg-414.json' ./built-in.bc"
    return execute_shell(cmd, debug=False)


def verify_baseline(json):
    cmd = f"{klee_path} -json='{json}' -fjson='../cg-414.json' -output-dir=klee-out --max-time=10min ./built-in.bc"
    return execute_shell(cmd, debug=False)

def verify_single_path(json, path):
    singleton_path_file = "singleton_path.txt"
    with open(singleton_path_file, "w") as f:
        f.write('\n'.join(path))
    
    return verify_singleton_path(json, singleton_path_file)

TIMEOUT = 10*60
start_times = [None for _ in range(total_cpu)]
processes = [None for _ in range(total_cpu)]
stats = {"min_path": [], "avg_paths": [], "num_paths": []}

def main():
    global stats
    # os.chdir("temp")
    for i in range(total_cpu):
        execute_shell(f"rm -rf {i}")
    for i in range(total_cpu):
        execute_shell(f"mkdir -p {i}")


    gnn = GNN([32, 16 , 32])
    sd = torch.load("/extra/ali/agents/saved_models/state.dict")
    gnn.load_state_dict(sd)
    gnn.eval()

    ids = set()
    with open("ids.json") as f:
        line = f.readline()
        while line:
            json = JSON.loads(line)
            warn_id = json["id"]
            ids.add(warn_id)
            line = f.readline()
         
    top_ranks = []
    print(len(ids))



    filename = sys.argv[1]
    file = open(filename)

    json_index = 0
    line = file.readline()
    SKIP_WARNINGS = 0
    idx = 0
    t = time.perf_counter()
    i = 0 
    infeasible = 0
  
    skipped = 0
    completed = 0
    paths_found = 0
    failed = 0
    timed_out_warnings = 0
    while line:
        i += 1
        json_index = json_index + 1
        link_file = line
        json = file.readline()
        json = json.replace("\n", "")
        line = file.readline()
        parsed = JSON.loads(json)
        if (parsed["id"] not in ids):
            skipped += 1
            continue 

        # if (len(x) > 1900):
        #     print(len(x))
        #     link_files(link_file)
        #     print("Linked", flush=True)
        #     enumerate_paths(json)
            # verify_baseline(json)
        # parsed = JSON.loads(json)
        
        # if (skipped < SKIP_WARNINGS):
        #     skipped += 1
        #     continue
        # # g = create_data_graph(link_file)
        # if link_file in done:
        #     continue
        
        timed_out = False
        while processes[idx] and processes[idx].is_alive():
            if start_times[idx] and time.perf_counter() - start_times[idx] > TIMEOUT:
                timed_out = True
                break 
            idx = (idx + 1) % total_cpu
        
        # output, stdout, stderr = link_files(link_file)
        # if (output != 0):
            # print(link_file)
            # print(stdout)
            # print(stderr)

        if timed_out:
            if processes[idx].is_alive():
                kill_proc_tree(processes[idx].pid, include_parent=False)
                processes[idx].kill()
            while processes[idx].is_alive():
                processes[idx].kill()
            processes[idx].close()
            timed_out_warnings += 1
        # p.apply_async(WarningVerifyProcess.run, (gnn, json, link_file, idx))
        # # processes[idx].terminate()
        if processes[idx] and not timed_out:
            processes[idx].join()
            processes[idx].close()
            success, path_found, rank, feasible = processes[idx].success.value, processes[idx].path_found.value, processes[idx].rank.value, processes[idx].p.value
            min_path, avg_path, num_path = processes[idx].min_path.value, processes[idx].avg_paths.value, processes[idx].num_paths.value
            stats["min_path"].append(min_path)
            stats["avg_paths"].append(avg_path)
            stats["num_paths"].append(num_path)
            if (success):
                completed += 1
            else:
                failed += 1
            if path_found:
                paths_found += 1
            top_ranks.append(rank)
            if not feasible:
                infeasible += 1
            processes[idx] = None
        # WarningVerifyProcess.run(gnn, json, g, link_file)
        print(f"Completed: {completed} Processed: {i-skipped}, Found_paths: {paths_found}, Total: {i}, Avg-Num-paths: {sum(stats['num_paths'])/max(len(stats['num_paths']), 1)}, Failed: {failed} Timed-out: {timed_out_warnings}, Skipped: {skipped}")
        # print(f"Completed: {completed} Processed: {i-skipped}, Found_paths: {paths_found}, Total: {i}, Failed: {failed} Timed-out: {timed_out_warnings}, Skipped: {skipped}")
        processes[idx] = WarningVerifyProcess(gnn, json, None, link_file, idx, json_index)
        # processes[idx] = GraphGeneratorProcess(link_file, index)
        processes[idx].start()
        os.sched_setaffinity(processes[idx].pid, affinity)
        start_times[idx] = time.perf_counter()


    for p in processes:
        if p:
            p.join()
            p.close()
            success, path_found, rank, feasible = p.success.value, p.path_found.value, p.rank.value, p.p.value
            min_path, avg_path, num_path = p.min_path.value, p.avg_paths.value, p.num_paths.value
            stats["min_path"].append(min_path)
            stats["avg_paths"].append(avg_path)
            stats["num_paths"].append(num_path)
            if (success):
                completed += 1
            else:
                failed += 1
            if path_found:
                paths_found += 1
            top_ranks.append(rank)
            if not feasible:
                infeasible += 1

    print(f"Elapsed time: {time.perf_counter()-t} seconds")
    print(f"Completed: {completed} Processed: {i-skipped}, Found_paths: {paths_found}, Total: {i}, Avg-Num-paths: {sum(stats['num_paths'])/len(stats['num_paths'])}, Failed: {failed} Timed-out: {timed_out_warnings}, Skipped: {skipped}")
    print(f"Avg rank: {sum(top_ranks)/len(top_ranks)}, 90th percentile rank: {np.percentile(top_ranks, 90)}, Infeasible: {infeasible}, Accuracy: {(completed-infeasible)*100}")
    

    with open("stats.pickle", "wb") as f:
        pickle.dump(stats, f)
    
    read_all_json("confirm_result.json")

    for i in range(total_cpu):
        shutil.rmtree(str(i))
    for i in range(total_cpu):
        execute_shell(f"rm -rf {i}")


if __name__ == "__main__":
    try:
    # multiprocessing.set_start_method("spawn")
        main()
    except KeyboardInterrupt:
        for p in processes:
            if p:
                kill_proc_tree(p.pid)
        raise KeyboardInterrupt
