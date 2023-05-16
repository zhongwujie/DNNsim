import numpy as np

def generate_fc(m = 30, n = 20, k = 40):
  weight_path = "./net_traces/fc/weight-layer0.npy"
  act_path = "./net_traces/fc/act-layer0-0.npy"
  act = np.random.rand(m, k)
  weight = np.random.rand(k, n)
  np.save(act_path, act)
  np.save(weight_path, weight)

def generate_conv():
  K = 8
  R = 4
  S = 4
  C = 32
  H = 4
  W = 7
  N = 1
  weight_path = "./net_traces/cnn/wgt-layer0.npy"
  act_path = "./net_traces/cnn/act-layer0-0.npy"
  act = np.random.randint(1, 128, size=(N, C, H, W), dtype = np.uint8)
  weight = np.random.randint(1, 128, size=(K, C, S, R), dtype = np.uint8)
  np.save(act_path, act)
  np.save(weight_path, weight)

def test():
  filepath = "./net_traces/cnn/wgt-layer0.npy"
  weight = np.load(filepath)
  print(weight[0:2, 0:2, 0:2, 0:2])
  
if __name__ == "__main__":
  generate_conv()