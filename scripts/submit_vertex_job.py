"""Submit KazLLM training job to Vertex AI.

Usage:
    # Submit 500M training job (1x A100 40GB, spot/preemptible):
    python scripts/submit_vertex_job.py

    # Custom settings:
    python scripts/submit_vertex_job.py \
        --model-config kaz500m \
        --training-config pretrain_500m \
        --machine-type a2-highgpu-1g \
        --accelerator-count 1 \
        --spot

    # Dry run (show config without submitting):
    python scripts/submit_vertex_job.py --dry-run
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PROJECT_ID = "minerva-invisionu"
REGION = "us-central1"
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/kazllm-training/kazllm-v3:latest"
GCS_BUCKET = "gs://kazllm-training-v3"
STAGING_BUCKET = GCS_BUCKET


def submit_job(args):
    from google.cloud import aiplatform

    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=STAGING_BUCKET,
    )

    display_name = f"kazllm-{args.model_config}-{args.training_config}"

    env_vars = {
        "GCS_DATA_DIR": f"{GCS_BUCKET}/data/tokenized",
        "GCS_CHECKPOINT_DIR": f"{GCS_BUCKET}/checkpoints/{args.model_config}",
        "GCS_TOKENIZER_DIR": f"{GCS_BUCKET}/tokenizer",
        "MODEL_CONFIG": args.model_config,
        "TRAINING_CONFIG": args.training_config,
    }

    if args.wandb_key:
        env_vars["WANDB_API_KEY"] = args.wandb_key
        env_vars["WANDB_PROJECT"] = "kazllm"

    machine_spec = {
        "machine_type": args.machine_type,
        "accelerator_type": args.accelerator_type,
        "accelerator_count": args.accelerator_count,
    }

    replica_spec = {
        "machine_spec": machine_spec,
        "replica_count": 1,
        "container_spec": {
            "image_uri": IMAGE_URI,
            "env": [{"name": k, "value": v} for k, v in env_vars.items()],
        },
        "disk_spec": {
            "boot_disk_type": "pd-ssd",
            "boot_disk_size_gb": 200,
        },
    }

    log.info(f"Job: {display_name}")
    log.info(f"Image: {IMAGE_URI}")
    log.info(f"Machine: {args.machine_type} + {args.accelerator_count}x {args.accelerator_type}")
    log.info(f"Spot: {args.spot}")
    log.info(f"Env: {env_vars}")

    if args.dry_run:
        log.info("DRY RUN — not submitting.")
        return

    job = aiplatform.CustomJob(
        display_name=display_name,
        worker_pool_specs=[replica_spec],
    )

    log.info("Submitting job to Vertex AI...")
    job.run(
        service_account=args.service_account,
        restart_job_on_worker_restart=True,  # auto-resume on preemption
        enable_web_access=False,
        scheduling_strategy="SPOT" if args.spot else "STANDARD",
        timeout=args.timeout,
        sync=False,  # don't block — return immediately
    )

    log.info(f"Job submitted: {job.display_name}")
    log.info(f"Resource name: {job.resource_name}")
    log.info(f"Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs"
             f"?project={PROJECT_ID}")
    log.info(f"Billing: https://console.cloud.google.com/billing?project={PROJECT_ID}")


def main():
    parser = argparse.ArgumentParser(description="Submit KazLLM Vertex AI training job")
    parser.add_argument("--model-config", default="kaz500m", help="Model config name")
    parser.add_argument("--training-config", default="pretrain_500m", help="Training config name")
    parser.add_argument(
        "--machine-type", default="a2-highgpu-1g",
        help="GCE machine type (default: a2-highgpu-1g = 1x A100 40GB)",
    )
    parser.add_argument(
        "--accelerator-type", default="NVIDIA_TESLA_A100",
        help="GPU type",
    )
    parser.add_argument("--accelerator-count", type=int, default=1)
    parser.add_argument("--spot", action="store_true", default=True, help="Use spot/preemptible (default: true)")
    parser.add_argument("--no-spot", dest="spot", action="store_false")
    parser.add_argument("--wandb-key", default=None, help="W&B API key for logging")
    parser.add_argument("--service-account", default=None)
    parser.add_argument("--timeout", type=int, default=60 * 60 * 96, help="Max job duration in seconds (default: 96h)")
    parser.add_argument("--dry-run", action="store_true", help="Show config without submitting")
    args = parser.parse_args()

    submit_job(args)


if __name__ == "__main__":
    main()
