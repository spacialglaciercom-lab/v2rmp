use anyhow::{Context, Result};
use futures_util::StreamExt;
use object_store::aws::AmazonS3Builder;
use object_store::ObjectStore;
use std::sync::Arc;

pub struct R2Storage {
    pub store: Arc<dyn ObjectStore>,
}

impl R2Storage {
    pub fn new(
        bucket: &str,
        access_key_id: &str,
        secret_access_key: &str,
        endpoint: &str,
    ) -> Result<Self> {
        let store = AmazonS3Builder::new()
            .with_bucket_name(bucket)
            .with_access_key_id(access_key_id)
            .with_secret_access_key(secret_access_key)
            .with_endpoint(endpoint)
            .with_region("auto") // R2 uses "auto" for region
            .with_allow_http(true) // R2 might use https, but object_store handles it via endpoint URL
            .build()
            .context("Failed to build R2 object store")?;

        Ok(Self {
            store: Arc::new(store),
        })
    }

    pub fn from_env(bucket: &str) -> Result<Self> {
        dotenvy::dotenv().ok();

        let access_key_id = std::env::var("R2_ACCESS_KEY_ID")
            .context("R2_ACCESS_KEY_ID not found in environment")?;
        let secret_access_key = std::env::var("R2_SECRET_ACCESS_KEY")
            .context("R2_SECRET_ACCESS_KEY not found in environment")?;
        let endpoint =
            std::env::var("R2_ENDPOINT").context("R2_ENDPOINT not found in environment")?;

        Self::new(bucket, &access_key_id, &secret_access_key, &endpoint)
    }

    pub async fn list_objects(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        let prefix_path = prefix.map(object_store::path::Path::from);
        let mut list_stream = self.store.list(prefix_path.as_ref());
        let mut objects = Vec::new();

        while let Some(meta) = list_stream.next().await {
            let meta = meta.context("Failed to list object from R2")?;
            objects.push(meta.location.to_string());
        }

        Ok(objects)
    }

    pub async fn upload_object(&self, path: &str, data: Vec<u8>) -> Result<()> {
        let location = object_store::path::Path::from(path);
        self.store
            .put(&location, data.into())
            .await
            .context("Failed to upload object to R2")?;
        Ok(())
    }

    pub async fn download_object(&self, path: &str) -> Result<Vec<u8>> {
        let location = object_store::path::Path::from(path);
        let result = self
            .store
            .get(&location)
            .await
            .context("Failed to download object from R2")?;

        let bytes = result
            .bytes()
            .await
            .context("Failed to read bytes from R2 get result")?;
        Ok(bytes.to_vec())
    }

    pub async fn delete_object(&self, path: &str) -> Result<()> {
        let location = object_store::path::Path::from(path);
        self.store
            .delete(&location)
            .await
            .context("Failed to delete object from R2")?;
        Ok(())
    }

    pub async fn get_object_info(&self, path: &str) -> Result<serde_json::Value> {
        let location = object_store::path::Path::from(path);
        let meta = self
            .store
            .head(&location)
            .await
            .context("Failed to get object info from R2")?;

        Ok(serde_json::json!({
            "path": meta.location.to_string(),
            "size": meta.size,
            "last_modified": meta.last_modified.to_rfc3339(),
            "e_tag": meta.e_tag,
        }))
    }
}
