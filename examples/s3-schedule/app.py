import beam


app = beam.App(
    name="s3-background-remover",
    cpu=4,
    memory="16Gi",
    python_version="python3.8",
    python_packages=["pillow", "rembg", "boto3"],
)

app.Trigger.Schedule(
    when="every 5m",
    handler="run.py:process_images",
)

app.Mount.PersistentVolume(app_path="./unprocessed_images", name="unprocessed_images")