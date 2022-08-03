import sys
from pkg_resources import parse_version

import torch, torchvision

assert parse_version(torch.__version__).public == sys.argv[1], \
    f"torch version should be {sys.argv[1]} but found {parse_version(torch.__version__).public}."
assert parse_version(torchvision.__version__).public == sys.argv[2], \
    f"torchvision version should be {sys.argv[2]} but found {parse_version(torchvision.__version__).public}."
assert torch.version.cuda == sys.argv[3], \
    f"torch cuda version should be {sys.argv[3]} but found {torch.version.cuda}."

