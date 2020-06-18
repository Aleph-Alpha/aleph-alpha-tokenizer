# aleph-alpha-tokenizer

![Rust](https://github.com/Aleph-Alpha/aleph-alpha-tokenizer/workflows/Rust/badge.svg)
[![docs.rs](https://docs.rs/aleph-alpha-tokenizer/badge.svg)](https://docs.rs/aleph-alpha-tokenizer)
[![License: MIT/Apache](https://img.shields.io/crates/l/aleph-alpha-tokenizer.svg)](#license)

We at Aleph Alpha are big fans of huggingface's [tokenizers] crate. Kudos for
this great library. There is only one downside: The interface is optimized for
the bindings, not for working with it from within Rust.

[tokenizers]: https://github.com/huggingface/tokenizers

So we took it as an inspiration and tried to improve on some things. First we
wanted to see how fast we could make it while implementing the same `Model`
trait. We based our implementation on the very good [fst](https://docs.rs/fst)
crate. Then we added our own interface to play to Rust's strengths (mainly
avoiding needless allocation, re-using data, generics).

We are very happy with the improved performance. In our tests, we found our
tokenizer performed mostly linearly with whatever data was thrown at it, while
the huggingface `wordpiece` tokenizer performs quadratically worse with longer
multi-token words. The following single-core runtimes in µs were measured for
a set of benchmarks:

| # |AlephAlphaTokenizer | ~ as Model | wordpiece |
|---|--------------------|------------|-----------|
| 0 |            749.950 |   1274.923 |  2025.289 |
| 1 |           1010.120 |   1511.214 |  1900.441 |
| 2 |           1775.973 |   2648.909 |  2995.574 |
| 3 |           2263.436 |   3598.771 | 12978.049 |
| 4 |           2262.490 |   3403.918 |  4864.752 |
| 5 |           2808.373 |   4456.960 | 18623.648 |
| 6 |           2783.996 |   4015.472 |  5362.356 |
| 7 |           3160.517 |   5048.136 |  9946.745 |
| 8 |           3016.781 |   4742.037 |  8066.818 |
| 9 |           3497.266 |   5626.896 |  8662.281 |
|10 |           4446.626 |   6679.859 | 10584.524 |

(This was measured on an Intel(R) Core(TM) i7-7600U CPU @ 2.80GHz running on
a Fedora kernel 5.6.15-300.fc32.x86_64 with all mitigations enabled)

As you can see, using our tokenizer as a model is faster than huggingface's 
wordpiece tokenizer by at least 13%, often more. Using the rustic interface, we 
can omit a lot of allocation and memory copying, so we are at least 60% faster.

To re-run the benchmark, call `cargo bench --all-features`. Otherwise only the
`AlephAlphaTokenizer` will be benchmarked.

# License

This package is licensed under MIT or Apache License Version 2, at your 
discretion.
