from beam.utils.mock import MockAPI

# Import your Beam app
from app import app

# Load your app into the MockAPI
mocker = MockAPI(app)

# Call the API
mocker.call(prompt="painted portrait of rugged zeus, god of thunder, greek god")
