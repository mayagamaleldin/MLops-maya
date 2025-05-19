import litserve as ls
from omegaconf import OmegaConf

from api import InferenceAPI

if __name__ == "__main__":
    # read params.yaml file
    api = InferenceAPI()
    server = ls.LitServer(api, accelerator="cpu")
    server.run(port=8000, generate_client_file=False)