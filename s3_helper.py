import io
import boto3
from botocore import UNSIGNED
from botocore.client import Config

# Config
REGION_NAME = 'ap-southeast-1'


def get_file_stream_s3(uri):
    # split bucket & key
    BUCKET_NAME, filename = split_s3_bucket_key(uri)

    s3 = boto3.resource('s3', region_name=REGION_NAME,
                        config=Config(signature_version=UNSIGNED))
    img_bucket = s3.Bucket(BUCKET_NAME)
    img_obj = img_bucket.Object(filename)

    # Prepare stream
    file_stream = io.BytesIO()
    img_obj.download_fileobj(file_stream)

    return file_stream


def find_bucket_key(s3_path):
    """
    This is a helper function that given an s3 path such that the path is of
    the form: bucket/key
    It will return the bucket and the key represented by the s3 path
    """
    s3_components = s3_path.split('/')
    bucket = s3_components[0]
    s3_key = ""
    if len(s3_components) > 1:
        s3_key = '/'.join(s3_components[1:])
    return bucket, s3_key


def split_s3_bucket_key(s3_path):
    """Split s3 path into bucket and key prefix.
    This will also handle the s3:// prefix.
    :return: Tuple of ('bucketname', 'keyname')
    """
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]
    return find_bucket_key(s3_path)
