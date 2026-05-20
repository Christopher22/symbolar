# symbolar

`symbolar` provides a strongly-typed API for building and querying high-dimensional symbolic representations. It is designed for workflows where symbolic composition, similarity search, and tabular data processing need to work together. It supports queries at runtime by a own small expression engine.

## Installation

Add the crate to your project:

```toml
[dependencies]
symbolar = { git = "https://github.com/Christopher22/symbolar.git"}
```

Additionally, you may want to build the Python binding. The development container already contains everything to build the package:

```bash
cd python
maturin develop
```

## Quick Start

The following code reproduces the famous "What is the Dollar in Mexico"? example:

```rust
use symbolar::{architectures::BinarySpatterCode, Expression, Fixed, Storage};

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

In Python, the same example looks like this:

```python
from symbolar import MultiplyAddPermute

# Create and seed the architecture.
vsa = MultiplyAddPermute(42)

# Create the storage and extend it with the symbols.
storage = vsa.create_storage(10_000)
storage.extend(["NAM", "MON",  "CAP", "USA", "DOL", "WDC", "MEX", "PES", "MXC"])

# Compute the dataset vector and query vector, then find the solution and compare it to the expected result.
dataset_vector = storage.execute(
    "((NAM * USA) + (CAP * WDC) + (MON * DOL)) * ((NAM * MEX) + (CAP * MXC) + (MON * PES))"
)
query_vector = storage.get("DOL")

# Find the solution and compare it to the expected result.
solution = storage.find(query_vector * dataset_vector)
expected = storage.get("PES")

assert solution.equals(expected)
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

To run the Python tests (including the Dollar-in-Mexico test), you can just run:

```bash
cd python
pytest -q
```

## Web Demo

The repository also contains a browser demo in [web/README.md](web/README.md). It builds the Rust crate without the `polars` feature to WebAssembly and exposes an interactive storage list where queries color-code elements by similarity.


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
