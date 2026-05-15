// Unit tests for optimize module (RMP binary read / metadata).
// Run with: cargo test --test unit_optimize --features extract

fn build_minimal_rmp(nodes: &[(f64, f64)], edges: &[(u32, u32, f64, u8)]) -> Vec<u8> {
    use std::io::Write;
    let mut buf = Vec::new();
    buf.extend_from_slice(b"RMP1");
    buf.write_all(&(nodes.len() as u32).to_le_bytes()).unwrap();
    buf.write_all(&(edges.len() as u32).to_le_bytes()).unwrap();
    for (lat, lon) in nodes {
        buf.write_all(&lat.to_le_bytes()).unwrap();
        buf.write_all(&lon.to_le_bytes()).unwrap();
    }
    for (f, t, w, ow) in edges {
        buf.write_all(&f.to_le_bytes()).unwrap();
        buf.write_all(&t.to_le_bytes()).unwrap();
        buf.write_all(&w.to_le_bytes()).unwrap();
        buf.push(*ow);
    }
    let crc = crc32fast::hash(&buf);
    buf.write_all(&crc.to_le_bytes()).unwrap();
    buf
}

#[test]
fn test_read_rmp_empty() {
    let nodes = vec![(45.0, 0.0), (45.001, 0.001)];
    let edges = vec![(0, 1, 150.0, 0)];
    let buf = build_minimal_rmp(&nodes, &edges);

    let (read_nodes, read_edges) = v2rmp::core::optimize::read_rmp_file(&buf).unwrap();
    assert_eq!(read_nodes.len(), 2);
    assert_eq!(read_edges.len(), 1);
    assert!((read_nodes[0].lat - 45.0).abs() < 1e-9);
    assert!((read_nodes[0].lon - 0.0).abs() < 1e-9);
}

#[test]
fn test_read_rmp_multiple_edges() {
    let nodes = vec![
        (45.0, 0.0),
        (45.001, 0.001),
        (45.002, 0.002),
    ];
    let edges = vec![
        (0, 1, 157.0, 0),
        (1, 2, 157.0, 0),
        (0, 2, 314.0, 1), // oneway
    ];
    let buf = build_minimal_rmp(&nodes, &edges);

    let (read_nodes, read_edges) = v2rmp::core::optimize::read_rmp_file(&buf).unwrap();
    assert_eq!(read_nodes.len(), 3);
    assert_eq!(read_edges.len(), 3);
    assert_eq!(read_edges[2].oneway, 1);
    assert!((read_edges[0].weight_m - 157.0).abs() < 0.01);
}

#[test]
fn test_read_rmp_bad_magic() {
    let bad = b"BAD!".to_vec();
    let res = v2rmp::core::optimize::read_rmp_file(&bad);
    assert!(res.is_err(), "should reject bad magic");
}

#[test]
fn test_read_rmp_crc_mismatch() {
    let mut buf = build_minimal_rmp(&[(45.0, 0.0)], &[]);
    // Corrupt the last CRC byte
    let last = buf.len() - 1;
    buf[last] = buf[last].wrapping_add(1);
    let res = v2rmp::core::optimize::read_rmp_file(&buf);
    assert!(res.is_err(), "should reject CRC mismatch");
}
