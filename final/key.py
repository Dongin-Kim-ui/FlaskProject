import boto3

def list_objects_in_bucket(bucket_name, prefix, access_key_id, secret_access_key):
    # 직접 액세스 키 ID와 시크릿 액세스 키를 사용하여 Boto3를 설정합니다.
    s3 = boto3.client('s3', aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

    try:
        # 지정된 버킷에서 해당 접두사(prefix)를 가진 객체들의 목록을 가져옵니다.
        response = s3.list_objects(Bucket=bucket_name, Prefix=prefix)
        
        # 가져온 객체 목록을 출력합니다.
        for file in response.get('Contents', []):
            print(file['Key'])
    
    except Exception as e:
        print(f"An error occurred: {e}")

# 실행 예시
if __name__ == "__main__":
    bucket_name = 'completesite'
    prefix = 'uploaded'
    access_key_id = ''
    secret_access_key = ''
    list_objects_in_bucket(bucket_name, prefix, access_key_id, secret_access_key)
