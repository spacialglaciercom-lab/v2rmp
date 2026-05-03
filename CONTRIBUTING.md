# Contributing to v2rmp

Thank you for your interest in contributing to v2rmp! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/v2rmp.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes: `cargo test` (when tests are available)
6. Commit with clear messages: `git commit -m "Add feature: description"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

### Prerequisites
- Rust 1.70 or later
- Cargo
- Git

### Building
```bash
cargo build
```

### Running
```bash
cargo run --bin rmpca
```

### Testing
```bash
cargo test
cargo clippy
cargo fmt --check
```

## Code Style

- Follow Rust standard formatting (`cargo fmt`)
- Pass all clippy lints (`cargo clippy`)
- Write clear, self-documenting code
- Add comments for complex logic
- Use meaningful variable and function names

## Commit Messages

Use clear, descriptive commit messages:
- `feat: Add new feature`
- `fix: Fix bug in optimizer`
- `docs: Update README`
- `refactor: Improve code structure`
- `test: Add tests for extraction`
- `chore: Update dependencies`

## Pull Request Process

1. Update documentation if needed
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md with your changes
5. Request review from maintainers
6. Address review feedback
7. Squash commits if requested

## Areas for Contribution

### High Priority
- [ ] OSM PBF extraction implementation
- [ ] Async/threaded operations for non-blocking UI
- [ ] Comprehensive test suite
- [ ] CI/CD pipeline setup

### Medium Priority
- [ ] Multi-depot support
- [ ] Time windows and capacity constraints
- [ ] Performance optimizations
- [ ] Additional road class filters

### Documentation
- [ ] API documentation improvements
- [ ] Usage examples
- [ ] Tutorial videos
- [ ] Architecture diagrams

## Testing Guidelines

When tests are implemented:
- Write unit tests for core logic
- Add integration tests for workflows
- Test edge cases and error conditions
- Aim for >80% code coverage

## Documentation

- Document all public APIs with rustdoc comments
- Include examples in documentation
- Update README.md for user-facing changes
- Keep CHANGELOG.md current

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Tag maintainers for urgent matters

## License

By contributing, you agree that your contributions will be licensed under the same terms as the project (MIT OR Apache-2.0).
