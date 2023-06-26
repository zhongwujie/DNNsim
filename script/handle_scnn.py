import numpy as np
import os
import subprocess
from typing import Dict, Tuple

def gen_fc_test(m = 30, n = 20, k = 40):
  weight_path = "./net_traces/fc/weight-layer0.npy"
  act_path = "./net_traces/fc/act-layer0-0.npy"
  act = np.random.rand(m, k)
  weight = np.random.rand(k, n)
  np.save(act_path, act)
  np.save(weight_path, weight)

def gen_conv_test():
  K = 8; R = 4; S = 4; C = 32; H = 4; W = 7; N = 1
  weight_path = "./net_traces/cnn/wgt-layer0.npy"
  act_path = "./net_traces/cnn/act-layer0-0.npy"
  act = np.random.randint(1, 128, size=(N, C, H, W), dtype = np.uint8)
  weight = np.random.randint(1, 128, size=(K, C, S, R), dtype = np.uint8)
  np.save(act_path, act)
  np.save(weight_path, weight)
  

# read config file
# return a dict includes {R, S, C, K, H, W, N} or {M, K, N}
def read_config(cfg_file: str) -> Dict[str, int]:
  param = {}
  with open(cfg_file, "r") as f:
    line = f.readline().strip()
    for item in line.split(","):
      key, value = item.split("=")
      key = key.strip()
      value = value.strip()
      param[key] = int(value)
  return param

def read_4d_array(file_path: str, dim: tuple) -> np.ndarray:
  arr_4d = np.zeros(dim, dtype = int)
  (d1, d2, d3, d4) = dim
  with open(file_path, 'r') as f:
    for i in range(d1):
        for j in range(d2):
            for k in range(d3):
              line = f.readline().strip()
              while(line == ""):
                line = f.readline().strip()
              items = line.split()
              for l in range(d4):
                arr_4d[i, j, k, l] = int(items[l])
  return arr_4d

def read_2d_array(file_path: str, dim: tuple) -> np.ndarray:
  arr_2d = np.zeros(dim, dtype = int)
  (d1, d2) = dim
  with open(file_path, 'r') as f:
    for i in range(d1):
      line = f.readline().strip()
      items = line.split()
      for j in range(d2):
        arr_2d[i, j] = int(items[j])
  return arr_2d

# An example
# input: ./models/benchmark/sparsity_0.20/conv 0
# output: ./dataset/data/benchmark/conv/data0/sparsity_0.20
def get_data_path(model_folder: str, layer_id: int) -> str:
  layer_type = model_folder.split("/")[-1]
  sparsity = model_folder.split("/")[-2]
  datai = "data" + str(layer_id)
  ret_str = "./dataset/data/benchmark/{}/{}/{}".format(layer_type, datai, sparsity)
  return ret_str

# An example
# input: ./models/benchmark/sparsity_0.20/conv 0
# output: ./dataset/config/benchmark/conv/config0.txt
def get_cfg_path(model_folder: str, layer_id: int) -> str:
  layer_type = model_folder.split("/")[-1]
  cfgi = "config" + str(layer_id)
  ret_str = "./dataset/config/benchmark/{}/{}.txt".format(layer_type, cfgi)
  return ret_str

def get_data(model_folder: str, layer_id: int) -> Tuple[np.ndarray, np.ndarray]:
  cfg_path = get_cfg_path(model_folder, layer_id)
  data_folder = get_data_path(model_folder, layer_id)
  param = read_config(cfg_path)
  # check if model_folder contains "conv"
  act_path = os.path.join(data_folder, "activation.txt")
  wei_path = os.path.join(data_folder, "weight.txt")
  if "conv" in model_folder:
    # read 4d array
    act_arr = read_4d_array(act_path, (param["N"], param["C"], param["H"], param["W"]))
    wei_arr = read_4d_array(wei_path, (param["C"], param["K"], param["S"], param["R"]))
    wei_arr = wei_arr.transpose((1, 0, 2, 3)) # (K, C, S, R)
  elif "fc" in model_folder:
    act_arr = read_2d_array(act_path, (param["M"], param["K"]))
    wei_arr = read_2d_array(wei_path, (param["K"], param["N"]))
  return (act_arr, wei_arr)

def get_layer_list(model_folder: str) -> list:
  model_path = os.path.join(model_folder, "model.csv")
  layer_list = []
  with open(model_path, "r") as f:
    for line in f.readlines():
      layer_id = line.split(",")[0].strip()
      layer_list.append(layer_id)
  return layer_list

# generate net_traces for the benchmark
def gen_net_trace(model_folder: str):
  layer_list = get_layer_list(model_folder)
  net_trace_folder = model_folder.replace("models", "net_traces")
  if not os.path.exists(net_trace_folder):
    os.makedirs(net_trace_folder)
  for layer_str in layer_list:
    print(layer_str)
    layer_id = int(layer_str[-1])
    act_arr, wei_arr = get_data(model_folder, layer_id)
    act_path = os.path.join(net_trace_folder, "act-layer{}-0.npy".format(layer_id))
    wei_path = os.path.join(net_trace_folder, "wgt-layer{}.npy".format(layer_id))
    np.save(act_path, act_arr)
    np.save(wei_path, wei_arr)

def get_model_folders(root_folder: str) -> list:
  model_folders = []
  for folder, subfolders, files in os.walk(root_folder):
    if 'model.csv' in files:
      model_folders.append(folder)
  return model_folders

def gen_net_traces(root_folder: str):
  model_folders = get_model_folders(root_folder)
  for model_folder in model_folders:
    print(model_folder)
    gen_net_trace(model_folder)

# execute the SCNN one time
def single_exec(scnn_path: str):
  command = ["cmake", "--build", "cmake-build-release/", "--target", "all"]
  subprocess.run(command)
  command = ["./DNNsim", scnn_path]
  subprocess.run(command)

# execute the SCNN multiple times
def multi_exec(scnn_folder: str):
  command = ["cmake", "--build", "cmake-build-release/", "--target", "all"]
  subprocess.run(command)
  for folder, subfolders, files in os.walk(scnn_folder):
    for file in files:
      file_path = os.path.join(folder, file)
      command = ["./DNNsim", file_path]
      subprocess.run(command)

def create_result_folder(model_root_folder: str):
  model_folders = get_model_folders(model_root_folder)
  for model_folder in model_folders:
    result_folder = model_folder.replace("models", "results")
    if not os.path.exists(result_folder):
      os.makedirs(result_folder)

def read_weight(file_path: str):
  weight = np.load(file_path)
  print(weight[0][0][:][:])
  

def test():
  read_weight("./net_traces/benchmark/sparsity_0.20/conv/wgt-layer0.npy")
  
if __name__ == "__main__":
  single_exec("./examples/SCNN/benchmark/sparsity_0.20/SCNN_Conv")
  # gen_net_traces("./models/benchmark")
  # create_result_folder("./models/benchmark")
  # test()