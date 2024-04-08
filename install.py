"""Install requirements for WD14-tagger."""
import os
import sys

from launch import run  # pylint: disable=import-error

NAME = "TokenizeAnything"
req_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "requirements.txt")
print(f"loading {NAME} reqs from {req_file}")
run(f'"{sys.executable}" -m pip install -q -r "{req_file}"',
    f"Checking {NAME} requirements.",
    f"Couldn't install {NAME} requirements.")

models = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
concepts = os.path.join(os.path.dirname(os.path.realpath(__file__)), "concepts")

run(f"mkdir {models}")
run(f"cd {models}")
run("wget https://huggingface.co/spaces/BAAI/tokenize-anything/resolve/main/models/tap_vit_l_548184.pkl")

run(f"mkdir {concepts}")
run(f"cd {concepts}")
run("wget https://huggingface.co/spaces/BAAI/tokenize-anything/resolve/main/concepts/merged_2560.pkl")
