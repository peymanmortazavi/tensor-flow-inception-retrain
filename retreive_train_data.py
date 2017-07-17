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

import pdb
pdb.set_trace()
