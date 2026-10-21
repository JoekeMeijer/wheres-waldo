import json
import argparse
import google.cloud.aiplatform as aip


def run_pipeline(project_id, location, template_path, parameters_path, pipeline_root):
    # Load parameters from JSON
    with open(parameters_path, 'r') as f:
        pipeline_params = json.load(f)

    # Initialize Vertex AI client
    aip.init(project=project_id, location=location)

    # Prepare and submit the pipeline job
    job = aip.PipelineJob(
        display_name="wheres-waldo-ct-pipeline",
        template_path=template_path,
        pipeline_root=pipeline_root,
        parameter_values=pipeline_params
    )

    job.submit()  # Use .submit() instead of .run() for CI/CD so it doesn't block the build indefinitely


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_id', required=True, type=str)
    parser.add_argument('--location', required=True, type=str)
    parser.add_argument('--template_path', default="wheres-waldo-sliding-window-predictor.yaml", type=str)
    parser.add_argument('--parameters_path', default="parameters.json", type=str)
    parser.add_argument('--pipeline_root', required=True, type=str)
    args = parser.parse_args()

    run_pipeline(**vars(args))