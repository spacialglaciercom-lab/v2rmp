# Publishing Checklist for v2rmp

This document outlines the steps to publish v2rmp to crates.io.

## Pre-Publishing Checklist

### 1. Code Quality
- [x] Code compiles without errors (`cargo check`)
- [x] All clippy warnings addressed or documented (`cargo clippy`)
- [x] Code formatted (`cargo fmt`)
- [ ] Tests pass (when implemented: `cargo test`)
- [x] No security vulnerabilities (`cargo audit` - install with `cargo install cargo-audit`)

### 2. Documentation
- [x] README.md complete with:
  - [x] Project description
  - [x] Installation instructions
  - [x] Usage examples
  - [x] Feature list
  - [x] License information
- [x] CHANGELOG.md created and up-to-date
- [x] CONTRIBUTING.md with contribution guidelines
- [x] LICENSE files (MIT and Apache-2.0)
- [ ] API documentation complete (`cargo doc --no-deps --open`)
- [ ] Examples directory with working examples (optional)

### 3. Cargo.toml Metadata
- [x] Package name: `v2rmp`
- [x] Version: `0.1.0`
- [x] Authors field populated
- [x] Description (max 200 chars)
- [x] License: `MIT OR Apache-2.0`
- [x] Repository URL
- [x] Homepage URL
- [x] Documentation URL
- [x] Keywords (max 5)
- [x] Categories (max 5)
- [x] README path specified
- [x] Exclude patterns for unnecessary files

### 4. Version Control
- [ ] All changes committed to git
- [ ] Repository pushed to GitHub/GitLab
- [ ] Tag created for version: `git tag -a v0.1.0 -m "Release v0.1.0"`
- [ ] Tag pushed: `git push origin v0.1.0`

### 5. Crates.io Account
- [ ] Account created at https://crates.io
- [ ] API token generated
- [ ] Token saved: `cargo login <your-token>`

## Publishing Steps

### Step 1: Final Verification
```bash
cd /home/rmp/Downloads/rmp.ca-main/v2rmp

# Clean build
cargo clean

# Full check
cargo check --all-targets

# Run clippy
cargo clippy -- -D warnings

# Format check
cargo fmt -- --check

# Build documentation
cargo doc --no-deps

# Dry run publish
cargo publish --dry-run
```

### Step 2: Update Version (if needed)
```bash
# Edit Cargo.toml version field
# Update CHANGELOG.md with release date
# Commit changes
git add Cargo.toml CHANGELOG.md
git commit -m "chore: Prepare v0.1.0 release"
```

### Step 3: Create Git Tag
```bash
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin main
git push origin v0.1.0
```

### Step 4: Publish to Crates.io
```bash
# Login (first time only)
cargo login <your-api-token>

# Publish
cargo publish
```

### Step 5: Verify Publication
- Visit https://crates.io/crates/v2rmp
- Check documentation at https://docs.rs/v2rmp
- Test installation: `cargo install v2rmp`

## Post-Publishing

### Announce Release
- [ ] Create GitHub release with changelog
- [ ] Post on Reddit r/rust
- [ ] Tweet/social media announcement
- [ ] Update project website (if applicable)

### Monitor
- [ ] Watch for issues on GitHub
- [ ] Respond to crates.io feedback
- [ ] Monitor download statistics

## Troubleshooting

### Common Issues

**"failed to verify package tarball"**
- Check `.gitignore` and `Cargo.toml` exclude patterns
- Ensure all source files are included

**"missing required field"**
- Verify all required Cargo.toml fields are present
- Check license file exists

**"crate name already taken"**
- Choose a different name in Cargo.toml
- Check availability: https://crates.io/crates/<name>

**"documentation failed to build"**
- Run `cargo doc --no-deps` locally
- Fix any doc comment errors
- Ensure all dependencies are available

## Before Next Release

1. Update version in `Cargo.toml`
2. Update `CHANGELOG.md` with new changes
3. Run full test suite
4. Update documentation
5. Create new git tag
6. Publish with `cargo publish`

## Useful Commands

```bash
# Check what will be published
cargo package --list

# Build and inspect package
cargo package
tar -tzf target/package/v2rmp-0.1.0.crate

# Yank a version (if needed)
cargo yank --vers 0.1.0

# Un-yank a version
cargo yank --vers 0.1.0 --undo
```

## Resources

- [Cargo Book - Publishing](https://doc.rust-lang.org/cargo/reference/publishing.html)
- [Crates.io Policies](https://crates.io/policies)
- [API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Semantic Versioning](https://semver.org/)
