#[cfg(test)]
mod overture_s3_integration {
    use futures_util::StreamExt;
    use object_store::aws::AmazonS3Builder;
    use object_store::path::Path;
    use object_store::ObjectStore;
    use std::sync::Arc;

    const BUCKET: &str = "overturemaps-us-west-2";
    const REGION: &str = "us-west-2";
    const RELEASE: &str = "2026-04-15.0";

    fn build_store() -> Arc<dyn ObjectStore> {
        let opts =
            object_store::ClientOptions::new().with_timeout(std::time::Duration::from_secs(120));
        let store = AmazonS3Builder::new()
            .with_bucket_name(BUCKET)
            .with_region(REGION)
            .with_skip_signature(true)
            .with_client_options(opts)
            .build()
            .expect("Failed to build S3 client");
        Arc::new(store)
    }

    #[tokio::test]
    async fn test_s3_list_segment_parquet_files() {
        let store = build_store();
        let prefix = Path::from(format!(
            "release/{}/theme=transportation/type=segment/",
            RELEASE
        ));

        let mut stream = store.list(Some(&prefix));
        let mut count = 0;
        let mut found_parquet = false;

        while let Some(item) = stream.next().await {
            let meta = item.expect("Failed to list S3 objects");
            if meta.location.to_string().ends_with(".parquet") {
                found_parquet = true;
                count += 1;
            }
            if count >= 5 {
                break;
            }
        }

        assert!(found_parquet, "No .parquet files found in listing");
        assert!(count > 0, "Expected at least 1 parquet file, got {count}");
        eprintln!("Found {count} parquet files in first few listing results");
    }

    #[tokio::test]
    async fn test_s3_download_single_parquet_file() {
        let store = build_store();
        let prefix = Path::from(format!(
            "release/{}/theme=transportation/type=segment/",
            RELEASE
        ));

        // Get first parquet file
        let mut stream = store.list(Some(&prefix));
        let first_parquet = loop {
            let meta = stream
                .next()
                .await
                .expect("Stream ended")
                .expect("Failed to list");
            if meta.location.to_string().ends_with(".parquet") {
                break meta.location;
            }
        };

        eprintln!("Downloading: {first_parquet}");

        let result = store.get(&first_parquet).await;
        assert!(
            result.is_ok(),
            "Failed to GET parquet file: {:?}",
            result.err()
        );

        let bytes = result.unwrap().bytes().await;
        assert!(
            bytes.is_ok(),
            "Failed to read parquet file bytes: {:?}",
            bytes.err()
        );

        let data = bytes.unwrap();
        assert!(
            data.len() > 1024,
            "Parquet file too small: {} bytes",
            data.len()
        );
        eprintln!("Downloaded {} bytes from {first_parquet}", data.len());

        // Verify it's a valid parquet file (magic bytes)
        assert_eq!(&data[0..4], b"PAR1", "Not a valid Parquet file");
    }

    #[tokio::test]
    async fn test_s3_list_finds_multiple_partitions() {
        let store = build_store();
        let prefix = Path::from(format!(
            "release/{}/theme=transportation/type=segment/",
            RELEASE
        ));

        let mut stream = store.list(Some(&prefix));
        let mut unique_files = std::collections::HashSet::new();

        while let Some(item) = stream.next().await {
            let meta = item.expect("Failed to list");
            let name = meta.location.to_string();
            if name.ends_with(".parquet") && name.contains("part-") {
                unique_files.insert(name);
            }
            if unique_files.len() >= 20 {
                break;
            }
        }

        eprintln!("Found {} unique partition files", unique_files.len());
        assert!(
            unique_files.len() >= 3,
            "Expected at least 3 partitions, got {}",
            unique_files.len()
        );
    }
}
