Version 0.3.0

* Breaking: fix a typo where the unknown token would be returned as `[UNK±`
  instead of `[UNK]`
* Breaking: fix offsets returned when used as `Model` which were previously
  falsely given in bytes instead of characters
* add fuzzing check against wordpiece
* fix benchmark run without `huggingface` feature

Version 0.2.0

* Breaking: fix a bug where the tokenizer would select the wrong token ID if
  one token was a prefix of another one which was off by the last character
* Breaking: Introduce the `huggingface` feature so this crate no longer depends
  on `tokenizers` by default

Version 0.1.0

* The initial public release
