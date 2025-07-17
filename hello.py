import sys
print(sys.executable)

import torch
print("Torch:", torch.__version__)
print("Device:", torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
