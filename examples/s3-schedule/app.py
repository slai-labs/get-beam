from beam import App, Runtime, Image, Volume

import os
import io
import boto3
from rembg import remove
import PIL

# The environment the app will run in
app = App(
    name="s3-background-remover",
    runtime=Runtime(
        cpu=4,
        memory="16Gi",
        image=Image(
            python_version="python3.8",
            python_packages=["pillow", "rembg", "boto3"],
        ),
    ),
    volumes=[Volume(path="./unprocessed_images", name="unprocessed_images")],
)


@app.schedule(when="every 5m")
def process_images():
    """
    This function will:
    - download all files from a bucket
    - remove the background of each image file
    - upload the new image files to a separate S3 bucket
    """
    client = Boto3Client()
    # Download S3 files to a Beam Persistent Volume
    client.download_from_s3(
        bucket_name=os.environ["UNPROCESSED_IMAGES_BUCKET"],
        download_path="./unprocessed_images",
    )

    for f in os.listdir("./unprocessed_images"):
        # Remove the background from each image, one by one
        with open(f"./unprocessed_images/{f}", "rb") as file:
            img = PIL.Image.open(file)
            output = remove(img)
            name = os.path.splitext(f)[0]

            # Convert image to bytes
            img_in_bytes = io.BytesIO()
            output.save(img_in_bytes, format="PNG")

            # Write the processed file back to a separate S3 bucket
            client.upload_to_s3(
                bucket_name=os.environ["PROCESSED_IMAGES_BUCKET"],
                file_body=img_in_bytes.getvalue(),
                key=f"{name}.png",
            )


"""
A basic S3 client. 
AWS credentials are safely retrieved from the Beam Secrets Manager:
getbeam.dev/apps/settings/secrets
"""


class Boto3Client:
    def __init__(self):
        # These environment variables are stored in the Beam Secrets Manager
        self.boto3_client = boto3.session.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name="us-east-1",
        )

    def download_from_s3(self, bucket_name, download_path):
        s3_client = self.boto3_client.resource("s3").Bucket(bucket_name)

        for s3_object in s3_client.objects.all():
            filename = os.path.split(s3_object.key)
            s3_client.download_file(s3_object.key, f"{download_path}/{filename}")

    def upload_to_s3(self, bucket_name, file_body, key):
        s3_client = self.boto3_client.resource("s3").Bucket(bucket_name)
        s3_client.put_object(Body=file_body, Key=key)


if __name__ == "__main__":
    process_images()
