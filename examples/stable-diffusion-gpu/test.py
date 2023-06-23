from beam.utils.mock import MockAPI

# Import your Beam app
from app import app

# Load your app into the MockAPI
mocker = MockAPI(app)

# Call the API
mocker.call(prompt="painted portrait of rugged zeus, god of thunder, greek god")

app = beam.App(
    name="test-cuda-toolkit",
    cpu=4,
    memory="8Gi",
    gpu="T4",
    commands=[
        "apt-get update",
        "apt-get install -y wget",
        "wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin",
        "mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600",
        "wget -q https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb",
        "export DEBIAN_FRONTEND=noninteractive && dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.1-530.30.02-1_amd64.deb",
        "cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/",
    ],
)