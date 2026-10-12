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

### Performance
For best performance, compile `symbolar` with CPU-specific optimizations enabled, for example like this: `RUSTFLAGS="-C target-cpu=native"`, `RUSTFLAGS="-C target-cpu=native -C target-feature=-prefer-256-bit"` (to enable AVX512), or `RUSTFLAGS="-C target-cpu=haswell"` (to enable all features supported by CPUs more recent than Haswell (2013)).

For a full list of supported CPU targets and features, run:

```sh
rustc --print target-cpus
rustc --print target-features
```

## Quick Start

The following code reproduces the famous "What is the Dollar in Mexico"? example. The documentation is available [here](https://christopher22.github.io/symbolar/docs/symbolar/).

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

## Encoding tabular data

The Python package encodes the rows of a Polars DataFrame as role-filler bindings, e.g. for linear readouts, decoding, symbolic cohort queries and explanations of model weights. A `Schema` describes how each variable is encoded. It can be written explicitly or inferred from the dtypes (`Enum`/`Categorical`/`String` → nominal, `Boolean` → binary, numbers → level vectors), with per-column overrides:

```python
import symbolar as sb

missing = sb.MissingCodes({7775: "structural", 8888: "refused", 9999: "dont_know"},
                          policy={"refused": "encode", "dont_know": "encode"})
schema = sb.Schema.infer(
    train,
    overrides={
        "isced": sb.Ordinal("isced"),                             # ordered levels
        "bmi": sb.Levels(64, quantiles=(0.005, 0.995)),            # level vectors on [P0.5, P99.5]
        "mi": sb.TimedEvent("mi", indicator="mi_ever", delta="mi_years_ago"),
        "icd": sb.HierarchicalCode("icd", hierarchy=sb.icd10_hierarchy, pool="icd10"),
        "diabetes": {"pool": "yes_no"},                           # share fillers with other yes/no questions
    },
    modules={"bmi": "anthropometry", "isced": "sociodemographics"},
    missing=missing,
)

encoder = sb.Encoder(sb.MultiplyAddPermute(42), 10_000, schema, structure="hierarchical").fit(train)
X_train, X_test = encoder.transform(train), encoder.transform(test)  # NumPy, int8 for normalized MAP/BSC

encoder.decode(X_test, ["bmi", "isced"])                            # nearest full keys
tau = encoder.null_threshold(X_train)                              # similarity of random vectors
encoder.query(X_test, ["smoking:current", "bmi>=30"], tau)         # conjunction, term by term
encoder.explain(weights)                                           # ⟨w, M ⊛ R ⊛ Φ(k)⟩ per variable and value
```

`fit` only derives the data-dependent parameters (categories, level ranges, scaling of fractional power encodings), so it can be applied to each training fold separately; the codebook only depends on the names and the seed. Codebook names follow a fixed scheme and can be used in expressions: roles `<variable>`, modules `@<module>`, fillers `<pool>=<value>`, levels `<pool>#<index>` and missing values `MISSING=<category>`. Full keys are written `<variable>:<label>`, and backticks quote names with whitespace.

Since version 0.14, named vectors are derived from the seed and the name only (`random_named`), independent of the order of creation. Vectors created with earlier versions differ.

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
maturin develop --release
pytest -q
```

## Web Demo

The repository also contains a [browser demo](https://christopher22.github.io/symbolar/). It builds the Rust crate without the `polars` feature to WebAssembly and exposes an interactive storage list where queries color-code elements by similarity.


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
