import time
import torch
import sys
import subprocess
import os

argslist = list(sys.argv)[1:]
num_gpus = torch.cuda.device_count()
print('# of GPUs: {}'.format(num_gpus))
argslist.append('--n_gpus={}'.format(num_gpus))
workers = []
job_id = time.strftime("%Y_%m_%d-%H%M%S")
argslist.append("--group_name=group_{}".format(job_id))

# create log dir if not exist
logdir = 'logs'
if not os.path.isdir(logdir):
  print('creating dir: {}'.format(logdir))
  os.makedirs(logdir)

for i in range(num_gpus):
    argslist.append('--rank={}'.format(i))
    stdout = None if i == 0 else open("{}/{}_GPU_{}.log".format(logdir, job_id, i),
                                      "w")
    print(argslist)
    p = subprocess.Popen([str(sys.executable)]+argslist, stdout=stdout)
    workers.append(p)
    argslist = argslist[:-1]

for p in workers:
    p.wait()
