use super::identity_digest;

// FIPS 180-4 vectors: the digest contract the artifact directory names
// ride on, pinned beyond what the ported report tests observe.
#[test]
fn digests_like_node_crypto_sha256() {
    assert_eq!(
        identity_digest(""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        identity_digest("abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
