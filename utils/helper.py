import csv
import io
import os
from urllib.parse import urlparse
import boto3
from dotenv import load_dotenv
from docx import Document
from pypdf import PdfReader

def fetch_docx_as_json(bucket:str , key:str):
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = response["Body"].read()

    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]

    return {
        "file_name": key,
        "content": paragraphs
    }
    
def fetch_json_from_s3(bucket: str, key: str):
    """Fetch and parse JSON file from S3"""
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = response["Body"].read()
    
    return {
        "file_name": key,
        "content": json.loads(file_bytes.decode('utf-8'))
    }


def fetch_csv_from_s3(bucket: str, key: str):
    """Fetch and parse CSV file from S3"""
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = response["Body"].read()
    
    csv_content = file_bytes.decode('utf-8')
    csv_reader = csv.DictReader(io.StringIO(csv_content))
    rows = list(csv_reader)
    
    return {
        "file_name": key,
        "rows": rows
    }


def fetch_pdf_from_s3(bucket: str, key: str):
    """Fetch and extract text content from PDF file in S3"""
    
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = response["Body"].read()
    
    # Extract text from PDF
    pdf_reader = PdfReader(io.BytesIO(file_bytes))
    
    text_content = []
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text.strip():
            text_content.append(page_text.strip())
    
    return {
        "file_name": key,
        "content": "\n\n".join(text_content)
    }


def fetch_text_from_s3(bucket: str, key: str):
    """Fetch text file from S3"""
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    file_bytes = response["Body"].read()
    
    return {
        "file_name": key,
        "content": file_bytes.decode('utf-8')
    }


def extract_bucket_and_key(s3_url):
    parsed = urlparse(s3_url)
    
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    _, ext = os.path.splitext(key)
    file_type = ext.lower().replace(".", "")
    
    return bucket, key, file_type


def fetch_document_by_type(bucket: str, key: str, file_type: str):
    """Fetch document based on file type"""
    if file_type == "docx":
        return fetch_docx_as_json(bucket, key)
    elif file_type == "json":
        return fetch_json_from_s3(bucket, key)
    elif file_type == "csv":
        return fetch_csv_from_s3(bucket, key)
    elif file_type == "pdf":
        return fetch_pdf_from_s3(bucket, key)
    elif file_type in ["txt", "md"]:
        return fetch_text_from_s3(bucket, key)
    else:
        # Unsupported file type, just store metadata
        return {
            "file_path": f"s3://{bucket}/{key}",
            "file_type": file_type,
            "status": "unsupported_type"
        }
