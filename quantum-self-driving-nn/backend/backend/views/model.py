from flask import Blueprint
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from auth import token_required

model_bp = Blueprint('model', __name__)

@model_bp.route("/load", methods=["GET"])
@token_required
# retrieve fine tuned model parameters for specific user
def load():
    return model_params



## need three endpoints, train, load (previously trained models to continue training), evaluate
@model_bp.route("/train", methods=["POST", "GET"])
@token_required
def train():
    # Load data from a bucket (one of three options)
    training_data=# get user choice from front end
    X_train = ...
    y_train = ...

    # Define model architecture
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile model
    # arguments should be customisable
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train model on AI Platform, modify below code depending on cloud provider
    project_id = 'your-project-id'
    job_name = 'your-job-name'
    region = 'your-region'
    bucket_name = 'your-bucket-name'
    train_data_path = 'gs://your-bucket-name/path/to/train/data.csv'
    job_dir = 'gs://your-bucket-name/path/to/job/dir'
    config_file_path = 'path/to/config.yaml'

    config = {
        'trainingInput': {
            'scaleTier': 'BASIC_GPU',
            'region': region,
            'jobDir': job_dir,
            'runtimeVersion': tf.__version__,
            'pythonVersion': '3.7',
            'masterConfig': {
                'imageUri': 'gcr.io/cloud-ml-algos/image_classification:latest',
                'acceleratorConfig': {
                    'count': 1,
                    'type': 'NVIDIA_TESLA_K80'
                }
            },
            'jobName': job_name,
            'packageUris': ['gs://your-bucket-name/path/to/trainer/package.tar.gz'],
            'pythonModule': 'trainer.task',
            'args': [
                '--train-data-path', train_data_path,
                '--job-dir', job_dir
            ]
        }
    }

    job_client = aiplatform.gapic.JobServiceClient(client_options={
        "api_endpoint": f"{region}-aiplatform.googleapis.com"
    })

    # print the epoch number during training
    # return a message saying model training finished
    parent = f"projects/{project_id}/locations/{region}"
    response = job_client.create_custom_job(parent=parent, custom_job=config)

