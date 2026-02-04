"""
This script detects anomalies in payments data using several machine learning techniques.
It reads input data from S3 bucket and transforms into embeddings using Standard Scaler
and Sentence Transformer, then compares them to model data stored in PostgreSQL. Finally,
enriched dataset with cosine similarity as anomaly signal (between 0 and 1) is stored as
output data into S3 bucket.
"""

from os import environ, path, makedirs
from timeit import default_timer
from concurrent.futures import ProcessPoolExecutor
from boto3 import client as boto3_client
from pandas import get_dummies, to_datetime, concat, read_csv
from numpy import concatenate, array, ndarray, pad
from psycopg import connect
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from kubernetes import client as k8s_client, config as k8s_config
from botocore.exceptions import ClientError
from warnings import warn


# -------------------- CONFIG MAP --------------------

def load_kubernetes_config():
    try:
        k8s_config.load_incluster_config()
    except k8s_config.ConfigException:
        try:
            k8s_config.load_kube_config()
        except k8s_config.ConfigException:
            return False
    return True


def get_namespace(prefix):
    if not load_kubernetes_config():
        return None

    v1 = k8s_client.CoreV1Api()
    try:
        namespaces = v1.list_namespace()
        for ns in namespaces.items:
            if ns.metadata.name.startswith(prefix):
                return ns.metadata.name
    except Exception:
        pass
    return None


def get_config_map_values(config_map_name="config-map"):
    if not load_kubernetes_config():
        return {}

    namespace = get_namespace("spf-app-anomaly-detector")
    if not namespace:
        return {}

    v1 = k8s_client.CoreV1Api()
    try:
        config_map = v1.read_namespaced_config_map(config_map_name, namespace)
        return config_map.data or {}
    except Exception:
        return {}


# -------------------- S3 HELPERS --------------------

def list_s3_files(s3_client, bucket, prefix):
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return response.get("Contents", [])
    except ClientError as e:
        raise e


def download_s3_file(s3_client, bucket, key, local_path):
    makedirs(path.dirname(local_path), exist_ok=True)
    s3_client.download_file(bucket, key, local_path)


def upload_s3_file(s3_client, bucket, local_path, key):
    s3_client.upload_file(local_path, bucket, key)


# -------------------- EMBEDDINGS --------------------

def encode_batch(batch):
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    return model.encode(batch)


def create_embeddings(df):
    numerical_features = [
        "billing_zip", "billing_latitude",
        "billing_longitude", "order_price"
    ]

    categorical_features = [
        "billing_state", "billing_country", "product_category"
    ]

    textual_features = [
        "customer_name", "billing_city", "billing_street",
        "customer_email", "billing_phone", "ip_address"
    ]

    timestamp_features = ["EVENT_TIMESTAMP", "LABEL_TIMESTAMP"]

    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    encoded_cat = get_dummies(df[categorical_features], drop_first=True).astype(float)

    for col in timestamp_features:
        df[col] = to_datetime(df[col])
        for unit in ["year", "month", "day", "hour", "minute", "second"]:
            df[f"{col}_{unit}"] = getattr(df[col].dt, unit)

    ts_cols = [c for c in df.columns if c.startswith("EVENT_TIMESTAMP_") or c.startswith("LABEL_TIMESTAMP_")]

    combined_features = concat(
        [df[numerical_features], encoded_cat, df[ts_cols]],
        axis=1
    )

    batch_size = 1000
    batches = [
        df[textual_features].fillna("").sum(axis=1).iloc[i:i+batch_size].tolist()
        for i in range(0, len(df), batch_size)
    ]

    text_embeddings = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        for result in executor.map(encode_batch, batches):
            text_embeddings.append(result)

    text_embeddings = concatenate(text_embeddings)

    embeddings = concatenate(
        [combined_features.values, text_embeddings],
        axis=1
    )

    return embeddings


# -------------------- DATABASE --------------------

def connect_to_database(dbname, user, password, host, port):
    return connect(
        f"host={host} port={port} dbname={dbname} user={user} password={password}"
    )


def is_transaction_anomaly(conn, df, embeddings, target_dim=847):
    cursor = conn.cursor()
    scores = []

    if embeddings.shape[1] != target_dim:
        if embeddings.shape[1] < target_dim:
            embeddings = pad(
                embeddings,
                ((0, 0), (0, target_dim - embeddings.shape[1])),
                mode="constant"
            )
        else:
            embeddings = embeddings[:, :target_dim]

    query = "SELECT MAX(1 - (embedding <=> %s::vector)) FROM transaction_anomalies"

    for emb in embeddings:
        cursor.execute(query, (emb.tolist(),))
        scores.append(cursor.fetchone()[0])

    cursor.close()
    df["anomaly_score"] = scores
    return df


# -------------------- MAIN --------------------

def main():
    config = get_config_map_values()

    s3_bucket = config.get("S3_BUCKET_NAME")
    s3_input = config.get("S3_PATH_PAYMENT")
    s3_output = config.get("S3_PATH_MODEL")

    s3 = boto3_client("s3")

    files = list_s3_files(s3, s3_bucket, s3_input)
    csv_files = [f["Key"] for f in files if f["Key"].endswith(".csv")]

    if "KUBERNETES_SERVICE_HOST" in environ:
        host = f'{config.get("SERVICE_NAME")}.{config.get("NAMESPACE")}'
        port = config.get("SERVICE_PORT")
    else:
        host = config.get("DBHOST")
        port = config.get("DBPORT")

    conn = connect_to_database(
        config.get("DBNAME"),
        config.get("DBUSER"),
        config.get("DBPASS"),
        host,
        port
    )

    for key in csv_files:
        local_path = path.join("data", key)
        download_s3_file(s3, s3_bucket, key, local_path)

        df = read_csv(local_path)
        embeddings = create_embeddings(df)
        df_scored = is_transaction_anomaly(conn, df, array(embeddings, dtype=float))

        out_local = local_path.replace(".csv", "_scored.csv")
        df_scored.to_csv(out_local, index=False)

        out_key = key.replace(s3_input, s3_output).replace(".csv", "_scored.csv")
        upload_s3_file(s3, s3_bucket, out_local, out_key)

    conn.close()


if __name__ == "__main__":
    start = default_timer()
    main()
    print(f"Total execution time: {default_timer() - start:.2f}s")
