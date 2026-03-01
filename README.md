# colossus-rs

Shared Rust library workspace for Colossus applications.

## Crates

| Crate | Description |
|-------|-------------|
| `colossus-auth` | Authentik + Axum authentication integration |

## Usage

Add as a git dependency in your application's `Cargo.toml`:

```toml
[dependencies]
colossus-auth = { git = "https://github.com/your-org/colossus-rs.git" }
```

## Development

```bash
cargo build            # Build all crates
cargo test --workspace # Run all tests
cargo clippy --workspace # Lint all crates
```
