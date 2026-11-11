
from maap.maap import MAAP
import logging
import argparse
import sys
import os
import boto3
import glob
from rasterio.session import AWSSession

S3_BUCKET = "maap-ops-workspace"

logger = logging.getLogger("hls_pipeline")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)s %(message)s"
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def upload_all_outputs_to_s3(tile, results_path, s3_prefix):
    logger.info("Uploading outputs to S3")
    s3 = boto3.client("s3")

    uploaded = []
    search_pattern = os.path.join(results_path, f"*/*/*.tif")
    tif_files = glob.glob(search_pattern)

    for fp in tif_files:
        key = (
            f"{s3_prefix}/"
            f"{tile}/"
            f"{os.path.basename(fp)}"
        )

        logger.info(f"Uploading s3://{S3_BUCKET}/{key}")

        s3.upload_file(
            fp,
            S3_BUCKET,
            key
        )

        uploaded.append(f"s3://{S3_BUCKET}/{key}")

    return uploaded


def get_credentials():
    logger.info("Configuring requester-pays credentials")
    maap = MAAP(maap_host="api.maap-project.org")
    
    credentials = maap.aws.requester_pays_credentials()
    boto3_session = boto3.Session(
        aws_access_key_id=credentials["aws_access_key_id"],
        aws_secret_access_key=credentials["aws_secret_access_key"],
        aws_session_token=credentials["aws_session_token"],
    )
    
    aws_session = AWSSession(
        boto3_session,
        requester_pays=True
    )
    return maap, aws_session


if __name__ == "__main__":
    print("Uploading to s3")
    parser = argparse.ArgumentParser(description="Download and process HLS data to EVI2.")
    parser.add_argument("--tile", required=True, help="MGRS tile ID (e.g., 18SUJ).")
    parser.add_argument("--results_path", required=True, help="Dir where phenometrics are output to from phenometric_algorithm.py")
    parser.add_argument("--output_prefix", required=False, default = None, help="Optional s3 prefix path, defaults to shared/colinquinn/HLS_phenometrics")
    
    args = parser.parse_args()
    mgrs_tile = args.tile    
    results_path = args.results_path
    s3_prefix = args.output_prefix
    
    if not args.output_prefix or not args.output_prefix.strip():
        s3_prefix = "shared/colinquinn/HLS_phenometrics"
    else:
        s3_prefix = args.output_prefix.strip()
        
    maap, aws_session = get_credentials()
    
    uploaded = upload_all_outputs_to_s3(mgrs_tile, results_path, s3_prefix)
    
    logger.info("Uploaded outputs:")
    for u in uploaded:
        logger.info(u)
        
    logger.info("PROCESSING COMPLETE")        

