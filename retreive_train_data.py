import argparse

import os

import boto3


s3 = boto3.resource('s3')


parser = argparse.ArgumentParser()
parser.add_argument(
  '--s3_path',
  type=str,
  default=os.environ.get('S3_PATH') or '',
  help='Path to the zip file.'
)
parser.add_argument(
  '--output',
  type=str,
  default=os.environ.get('OUTPUT') or '',
  help='Path to the output directory.'
)


FLAGS, _ = parser.parse_known_args()
access_key = os.environ.get('AWS_ACCESS_KEY_ID')
secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')


s3_path_parts = FLAGS.s3_path.split('/')
bucket_name = s3_path_parts[0]
file_path = '/'.join(s3_path_parts[1:])


s3.Bucket(bucket_name).download_file(file_path, os.path.join(FLAGS.output, s3_path_parts[-1]))
