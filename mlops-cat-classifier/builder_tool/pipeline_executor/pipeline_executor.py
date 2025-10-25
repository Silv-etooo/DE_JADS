import argparse
import google.cloud.aiplatform as aip
import logging
import sys
import json # You need this to load the parameter_dict

def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    # Arguments passed from Cloud Build Step 4
    parser.add_argument('--name', type=str, required=True, help="Pipeline Display Name")
    parser.add_argument('--pipeline_def', type=str, required=True, help="Path to the compiled Pipeline YAML file")
    parser.add_argument('--pipeline_root', type=str, required=True, help="GCS URI for pipeline root artifacts")
    parser.add_argument('--parameter_dict', type=str, required=True, help="Path to the JSON file containing pipeline parameters")
    # New arguments needed for aip.init()
    parser.add_argument('--project_id', type=str, required=True)
    parser.add_argument('--region', type=str, required=True) 
    
    args = parser.parse_args()
    return vars(args)


def run_pipeline_job(name, pipeline_def, pipeline_root, parameter_dict, project_id, region):
    logging.info(f"Initializing Vertex AI in {project_id}/{region}")

    # 1. Initialize Vertex AI
    aip.init(project=project_id, staging_bucket=pipeline_root, location=region)

    # 2. Load parameters from the JSON file path passed via the argument
    with open(parameter_dict, 'r') as f:
        pipeline_parameters = json.load(f)

    # 3. Create and submit the Pipeline Job
    job = aip.PipelineJob(
        display_name=name,
        enable_caching=False, 
        template_path=pipeline_def, # The compiled YAML file
        pipeline_root=pipeline_root,
        location=region,
        parameter_values=pipeline_parameters # The dictionary of parameters
    )

    logging.info("Submitting pipeline...")
    job.run()
    logging.info("✓ Pipeline execution initiated!")


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    try:
        args = parse_command_line_arguments()
        run_pipeline_job(**args)
    except Exception as e:
        logging.error(f"Failed to submit pipeline: {e}")
        sys.exit(1)