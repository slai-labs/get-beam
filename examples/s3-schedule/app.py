import beam

# The environment the app will run in
app = beam.App(
    name="s3-background-remover",
    cpu=4,
    memory="16Gi",
    python_version="python3.8",
    python_packages=["pillow", "rembg", "boto3"],
)

# A trigger that sets the app to run on a schedule
app.Trigger.Schedule(
    # The frequency can be denoted in cron or every syntax
    when="every 5m",
    # The handler is the function that will run when the task is invoked
    handler="run.py:process_images",
)

# Mount a Persistent Volume to store the images downloaded from S3
app.Mount.PersistentVolume(app_path="./unprocessed_images", name="unprocessed_images")