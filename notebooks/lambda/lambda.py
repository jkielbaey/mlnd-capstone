import os
import boto3

def lambda_handler(event, context):
    
    data = event['body']

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
    response = runtime.invoke_endpoint(EndpointName=os.environ['SM_ENDPOINT'],
                                       ContentType='text/csv',
                                       Accept='text/csv',
                                       Body=data)

    # The response is an HTTP response whose body contains the result of our inference
    result = response['Body'].read().decode('utf-8').split()

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/csv' },
        'body' : str(','.join(result))
    }