from src.deployment.online.api import InferenceAPI
import litserve as ls

if __name__ == "__main__":
    api = InferenceAPI()
    server = ls.LitServer(
        api, 
        accelerator="cpu"
    )
    server.run(port=8000, generate_client_file = False)