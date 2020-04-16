# aleph-alpha-tokenizer

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
words. The following single-core runtimes in µs were measured for a set of
benchmarks:

| # |AlephAlphaTokenizer | ~ as Model | wordpiece |
| 0 |  780.927           | 1251.911   |  2134.343 |
| 1 | 1000.583           | 1452.931   |  1870.142 |
| 2 | 3240.754           | 4659.755   |  9540.037 |
| 3 | 1772.298           | 2504.794   |  3001.438 |
| 4 | 2376.060           | 3437.649   | 13287.062 |
| 5 | 3270.748           | 4530.615   |  7985.852 |
| 6 | 3813.023           | 5549.230   |  9130.947 |
| 7 | 3173.984           | 4259.019   | 18975.838 |
| 8 | 4811.540           | 6590.099   | 11097.789 |
| 9 | 2896.895           | 3771.085   |  5378.582 |

(This was measured on an Intel(R) Core(TM) i7-7600U CPU @ 2.80GHz running on
a Fedora kernel 5.5.13-200.fc31.x86_64 with all mitigations enabled)

As you can see, using our tokenizer as a model is faster than huggingface's 
wordpiece tokenizer by at least 20%, often more. Using the rustic interface, we 
can omit a lot of allocation and memory copying, so we are at least 45% faster.

# License

This package is licensed under MIT or Apache 2 License, at your discretion.
