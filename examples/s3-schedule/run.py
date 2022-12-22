import os
import io
import boto3
from PIL import Image
from rembg import remove


class Boto3Client:
    def __init__(self):
        self.boto3_client = boto3.session.Session(
            aws_access_key_id=os.environ["AWS_ACCESS_KEY"],
            aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            region_name="us-east-1",
        )

    def download_from_s3(self, bucket_name, download_path):
        s3_client = self.boto3_client.resource("s3").Bucket(bucket_name)
        all_files = s3_client.objects.all()

        for s3_object in all_files:
            filename = os.path.split(s3_object.key)
            s3_client.download_file(s3_object.key, f"{download_path}/{filename}")

    def upload_to_s3(self, bucket_name, file_body, key):
        s3_client = self.boto3_client.resource("s3").Bucket(bucket_name)
        s3_client.put_object(Body=file_body, Key=key)


def process_images():
    client = Boto3Client()
    # Download S3 files to a Beam Persistent Volume
    client.download_from_s3(
        bucket_name="slai-example-images", download_path="./unprocessed_images"
    )

    for f in os.listdir("./unprocessed_images"):
        with open(f"./unprocessed_images/{f}", "rb") as file:
            img = Image.open(file)
            output = remove(img)
            name = os.path.splitext(f)[0]

            # Convert image to bytes
            img_in_bytes = io.BytesIO()
            output.save(img_in_bytes, format="PNG")

            # Write back to S3 bucket
            client.upload_to_s3(
                bucket_name="slai-processed-images",
                file_body=img_in_bytes.getvalue(),
                key=f"{name}.png",
            )


if __name__ == "__main__":
    process_images()