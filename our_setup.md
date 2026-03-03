1. Open a Terminal
2.
```
gcloud auth login
```
There will be a popup. Sign in to the account with Google Cloud.

3.
```
gcloud config set project uclac147
```
uclac147 is the project name

4. In VS Code, click on "Open a Remote Window" in the bottom left. Then, "Connect to Host". There should be a host starting with "instance-2"

5. At top of train.py add
```
import torch
torch.backends.cudnn.enabled = False
```

6. Activate conda environment, then run whatever
```
conda activate emg
```
