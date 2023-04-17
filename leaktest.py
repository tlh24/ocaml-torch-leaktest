import torch as th
import subprocess

image_count = 4*2048
image_res = 30

torch_device = 0
print("torch cuda devices", th.cuda.device_count())
print("torch device", th.cuda.get_device_name(torch_device))
th.cuda.set_device(torch_device)
th.set_default_tensor_type('torch.cuda.FloatTensor')
th.set_float32_matmul_precision('high') # desktop.

def run_nvidiasmi () : 
	p = subprocess.Popen(["nvidia-smi | grep python"], stdout=subprocess.PIPE, shell=True)
	(output, err) = p.communicate()
	print(output)

dbf = th.randn(image_count, image_res, image_res)
for i in range(30): 
	a = th.randn(image_res, image_res)
	d = th.sum((dbf - a)**2, (1,2))
	mindex = th.argmin(d)
	dist = d[mindex]
	print(f"[{i}] {th.cuda.memory_allocated(torch_device) / 1e6} MB")
	run_nvidiasmi ()
