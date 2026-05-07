# polars-vsa

`polars-vsa` provides a strongly-typed API for building and querying high-dimensional symbolic representations. It is designed for workflows where symbolic composition, similarity search, and tabular data processing need to work together. It supports queries at runtime by a own small expression engine.

## Installation

Add the crate to your project:

```toml
[dependencies]
polars-vsa = { git = "https://github.com/Christopher22/polars-vsa.git"}
```

## Quick Start

The following code reproduces the famous "What is the Dollar in Mexico"? example:

```rust
use polars_vsa::{architectures::BinarySpatterCode, Expression, Fixed, Storage};

fn main() {
    // Create the storage
    let mut storage = Storage::new(BinarySpatterCode::<u8>::new(42), Fixed::<10_000>);
    storage.extend([
        "NAM", "MON", "CAP", // Features
        "USA", "DOL", "WDC", // USA
        "MEX", "PES", "MXC", // Mexico
    ]);

    // Build the expression and run it
    let expression: Expression =
        "((NAM * USA) + (CAP * WDC) + (MON * DOL)) * ((NAM * MEX) + (CAP * MXC) + (MON * PES))"
            .parse()
            .expect("valid expression");
    let relation = storage.execute(&expression).expect("expression can be evaluated");

    // Query the dollar
    let query = &storage["DOL"];
    let best = storage
        .find(&(query * relation.as_ref()), &())
        .expect("at least one candidate");

    assert_eq!(&storage[best], &storage["PES"]);
}
```

## Project Status

The crate is under active development. APIs may evolve before a 1.0 release.

## Development

Run checks locally:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
