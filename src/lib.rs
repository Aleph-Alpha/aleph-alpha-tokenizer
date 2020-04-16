//! aleph-alpha-tokenizer is a fast word-piece-like tokenizer based on fst
//!
//! This can be used as a `Model` in huggingface's tokenizers, or standalone.
//!
//! By default, this library builds only the code to be used standalone. Add it
//! to your `Cargo.toml` with the following `[dependencies]` entry:
//!
//! ```toml
//! [dependencies]
//! aleph-alpha-tokenizers = "0.1"
//! ```
//!
//! If you want to use it together with `tokenizers`, you need to enable the
//! `huggingface` feature, so the dependency entry becomes:
//!
//! ```toml
//! [dependencies]
//! aleph-alpha-tokenizers = { version = "0.1", features = ["huggingface"] }
//! ```
//!
//! # Examples
//!
//! To use as a [`Model`](../tokenizers/tokenizer/trait.Model.html), you need
//! to box it:
//!
//! ```
//!# use std::error::Error;
//!
//!# #[cfg(feature = "huggingface")] {
//! use tokenizers::{
//!     tokenizer::{EncodeInput, Model, Tokenizer},
//!     pre_tokenizers::bert::BertPreTokenizer,
//! };
//! use aleph_alpha_tokenizer::AlephAlphaTokenizer;
//!
//! let mut tokenizer = Tokenizer::new(
//!     Box::new(AlephAlphaTokenizer::from_vocab("vocab.txt")?));
//! tokenizer.with_pre_tokenizer(Box::new(BertPreTokenizer));
//! let _result = tokenizer.encode(
//!     EncodeInput::Single("Some Test".to_string()), true)?;
//!# }
//!# Ok::<_, Box<dyn Error + Send + Sync>>(())
//! ```
//!
//! Remember this depends on the `huggingface` feature. Otherwise, you can use
//! it directly:
//!
//! ```
//!# use std::error::Error;
//! use aleph_alpha_tokenizer::AlephAlphaTokenizer;
//!
//! let source_text = "Ein interessantes Beispiel";
//! let tokenizer = AlephAlphaTokenizer::from_vocab("vocab.txt")?;
//! let mut ids: Vec<i64> = Vec::new();
//! let mut ranges = Vec::new();
//! tokenizer.tokens_into(source_text, &mut ids, &mut ranges, None);
//! for (id, range) in ids.iter().zip(ranges.iter()) {
//!      let _token_source = &source_text[range.clone()];
//!      let _token_text = tokenizer.text_of(*id);
//!      let _is_special = tokenizer.is_special(*id);
//!      // etc.
//! }
//!# Ok::<_, Box<dyn Error + Send + Sync>>(())
//! ```

use fst::raw::{Fst, Output};
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::mem::replace;
use std::ops::Range;
use std::path::PathBuf;

#[cfg(feature = "huggingface")]
use tokenizers::tokenizer::{Model, Token as HfToken};

// TODO: this should be upstreamed into fst
//
// For now, we'll keep it here.
#[inline]
fn find_longest_prefix<D: AsRef<[u8]>>(fst: &Fst<D>, input: &[u8]) -> Option<(usize, u64)> {
    let mut node = fst.root();
    let mut out = Output::zero();
    let mut last_match: Option<(usize, u64)> = None;
    for (i, &b) in input.iter().enumerate() {
        if let Some(trans_index) = node.find_input(b) {
            let t = node.transition(trans_index);
            node = fst.node(t.addr);
            if node.is_final() {
                last_match = Some((i + 1, out.cat(node.final_output()).value()));
            }
            out = out.cat(t.out);
        } else {
            return last_match;
        }
    }
    last_match
}

/// A trait to be able to convert token IDs on the fly
pub trait TokenID: PartialEq + Clone {
	/// Get a zero value
	fn zero() -> Self;
	
	/// Convert a `u64` to `Self`
	fn coerce(t: u64) -> Self;
	
	/// Convert back into `u64`
	fn restore(self) -> u64;
}

impl TokenID for u64 {
	fn zero() -> Self { 0 }
	
	#[inline(always)]
	fn coerce(t: u64) -> Self { t }
	
	#[inline(always)]
	fn restore(self) -> u64 { self }
}

// This can be used in torch Tensors
impl TokenID for i64 {
	fn zero() -> Self { 0 }
	
	#[inline(always)]
	fn coerce(t: u64) -> Self { t as i64 }

	#[inline(always)]
	fn restore(self) -> u64 { self as u64 }
}

// This can be used in torch Tensors
impl TokenID for i32 {
	fn zero() -> Self { 0 }
	
	#[inline(always)]
	fn coerce(t: u64) -> Self { t as i32 }

	#[inline(always)]
	fn restore(self) -> u64 { self as u64 }
}

// This can be used in torch Tensors
impl TokenID for f64 {
	fn zero() -> Self { 0.0 }
	
	#[inline(always)]
	fn coerce(t: u64) -> Self { t as f64 }

	#[inline(always)]
	fn restore(self) -> u64 { self as u64 }
}

/// The Tokenizer. Use [`AlephAlphaTokenizer::from_vocab`] to create an
/// instance.
pub struct AlephAlphaTokenizer {
    tokens: Vec<String>,
    starters: Fst<Vec<u8>>,
    followers: Fst<Vec<u8>>,
    //TODO: perhaps use a SmallVec here
    special_tokens: Vec<u64>,
    unk_id: u32,
    prefix: Option<u32>,
    suffix: Option<u32>,
}

impl AlephAlphaTokenizer {
    /// Creates a tokenizer from the vocabulary.
    ///
    /// For now, we assume the following tokens / IDs:
    ///
    /// * `[CLS]` is classification (and if present is used as prefix)
    /// * `[SEP]` is separator (and if present is used as suffix)
    /// * `[PAD]` is padding and is in position `0`
    /// * `[UNK]` is the *unknonw* token specifier
    pub fn from_vocab(path: &str) -> Result<Self, Box<dyn Error + Send + Sync>> {
        let vocab = File::open(path)?;
        let tokens = BufReader::new(vocab)
            .lines()
            .collect::<Result<Vec<String>, std::io::Error>>()?;
        let mut starter: Vec<(Vec<u8>, u64)> = Vec::new();
        let mut follower: Vec<(Vec<u8>, u64)> = Vec::new();
        let mut special_tokens = Vec::new();
        let mut unk_id = None;
        let mut prefix = None;
        let mut suffix = None;
        for (i, tok) in tokens.iter().enumerate() {
            let token = tok.as_bytes();
            if token.starts_with(b"[") && token.ends_with(b"]") {
                if token.starts_with(b"[unused") {
                    continue;
                }
                if token == b"[UNK]" {
                    unk_id = Some(i as u32);
                } else if token == b"[CLS]" {
                    prefix = Some(i as u32);
                } else if token == b"[SEP]" {
                    suffix = Some(i as u32);
                }
				special_tokens.push(i as u64);
            }
            if token.starts_with(b"##") {
                follower.push((token[2..].to_vec(), i as u64));
            } else {
                starter.push((token.to_vec(), i as u64));
            }
        }
        starter.sort_by(|(k, _), (j, _)| k.cmp(j));
        follower.sort_by(|(k, _), (j, _)| k.cmp(j));
        Ok(AlephAlphaTokenizer {
            tokens,
            starters: Fst::from_iter_map(starter)?,
            followers: Fst::from_iter_map(follower)?,
            special_tokens,
            unk_id: unk_id.ok_or(Box::new(std::env::VarError::NotPresent))?,
            prefix,
            suffix,
        })
    }

	#[inline]
    fn add_prefix<T: TokenID>(&self, token_ids: &mut Vec<T>, token_ranges: &mut Vec<Range<usize>>) {
        if let Some(id) = self.prefix {
            token_ids.push(T::coerce(u64::from(id)));
            token_ranges.push(0..0);
        }
    }

	#[inline]
    fn add_suffix<T: TokenID>(&self, token_ids: &mut Vec<T>, token_ranges: &mut Vec<Range<usize>>) {
        if let Some(id) = self.suffix {
            let pos = token_ranges.last().map_or(0, |range| range.end);
            token_ids.push(T::coerce(u64::from(id)));
            token_ranges.push(pos..pos);
        }
    }

    fn tokenize_word<T: TokenID>(
        &self,
        text: &str,
        range: Range<usize>,
        token_ids: &mut Vec<T>,
        token_ranges: &mut Vec<Range<usize>>,
    ) {
        let (start, end) = (range.start, range.end);
        let word_index = token_ids.len();
        let mut last_index = start;
        if let Some((len, id)) = find_longest_prefix(&self.starters, text[start..end].as_bytes()) {
            last_index = start + len;
            token_ids.push(T::coerce(id));
            token_ranges.push(start..last_index);
            while last_index < end {
                if let Some((len, id)) =
                    find_longest_prefix(&self.followers, &text[last_index..end].as_bytes())
                {
                    let next_index = last_index + len;
                    token_ids.push(T::coerce(id));
                    token_ranges.push(last_index..replace(&mut last_index, next_index));
                } else {
                    break;
                }
            }
        }
        if last_index < end {
            assert!(word_index <= token_ids.len());
            token_ids.truncate(word_index);
            token_ids.push(T::coerce(u64::from(self.unk_id)));
            token_ranges.truncate(word_index);
            token_ranges.push(range);
        }
    }

    /// tokenize the given text into a `&mut Vec<u64>` for ids and
    /// `&mut Vec<Range<usize>>` for source ranges respectively, optionally 
    /// filling a `words` `&mut Vec<Range>` with ranges into the tokens array
    /// with the words' token indices.
    ///
    /// This works by first splitting by whitespace, then gathering the longest
    /// prefix in our token tree (first the starters, then the followers) until
    /// the word is complete, or inserting a `[UNK]` token if the word couldn't
    /// fully be tokenized. This is what wordpiece does, too.
    ///
    /// Note: The output `Vec`s will be cleared before appending tokens.
    ///
    /// # Examples
    ///
    /// ```
    /// use aleph_alpha_tokenizer::AlephAlphaTokenizer;
    ///
    /// let source_text = "Ein interessantes Beispiel";
    /// let tokenizer = AlephAlphaTokenizer::from_vocab("vocab.txt").unwrap();
    /// let mut ids: Vec<i32> = Vec::new();
    /// let mut ranges = Vec::new();
    /// tokenizer.tokens_into(source_text, &mut ids, &mut ranges, None);
    /// assert_eq!(&[3, 198, 19168, 26889, 2249, 4], &ids[..]);
    /// ```
    pub fn tokens_into<T: TokenID>(
        &self,
        text: &str,
        token_ids: &mut Vec<T>,
        token_ranges: &mut Vec<Range<usize>>,
        words: Option<&mut Vec<Range<usize>>>,
    ) {
		token_ids.clear();
		token_ranges.clear();
		let text_len = text.len();
        let mut words = words;
        if let Some(w) = words.as_mut() {
			w.clear();
		}
        let mut last_offs = 0;
        self.add_prefix(token_ids, token_ranges);
        let mut last_token = token_ids.len();
        //TODO: there may be a faster version of this using SIMD
        while let Some(next_ws) = text[last_offs..].find(char::is_whitespace) {
            if next_ws != 0 {
                self.tokenize_word(text, last_offs..last_offs + next_ws, token_ids, token_ranges);
                if let Some(w) = words.as_mut() {
                    w.push(last_token..replace(&mut last_token, token_ids.len()));
                }
            }
            last_offs += next_ws + 1;
        }
        if last_offs < text_len {
            self.tokenize_word(text, last_offs..text_len, token_ids, token_ranges);
        }
        self.add_suffix(token_ids, token_ranges);
    }

    /// Gets the text of this token.
    ///
    /// # Examples
    ///
    /// ```
    /// use aleph_alpha_tokenizer::AlephAlphaTokenizer;
    /// let tokenizer = AlephAlphaTokenizer::from_vocab("vocab.txt").unwrap();
    ///
    /// assert_eq!("[PAD]", tokenizer.text_of(0));
    /// ```
    #[inline]
    pub fn text_of<T: TokenID>(&self, token_id: T) -> &str {
        &self.tokens[token_id.restore() as usize]
    }

    /// Gets the texts of the tokens.
    ///
    /// # Examples
    ///
    /// ```
    /// use aleph_alpha_tokenizer::AlephAlphaTokenizer;
    /// let tokenizer = AlephAlphaTokenizer::from_vocab("vocab.txt").unwrap();
    ///
    /// assert_eq!(
    ///     vec!["[CLS]", "Super", "[SEP]"], 
    ///     tokenizer.texts_of(&[3, 4285, 4])
    /// );
    /// ```
    pub fn texts_of<'t, T: TokenID>(&'t self, token_ids: &[T]) -> Vec<&'t str> {
        token_ids.iter().cloned().map(|id| self.text_of(id)).collect()
    }

    /// Determines whether this token is a special token.
    ///
    /// Special tokens are e.g. `[CLS]`, `[SEP]`, `[PAD]` or `[UNK]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aleph_alpha_tokenizer::AlephAlphaTokenizer;
    /// let tokenizer = AlephAlphaTokenizer::from_vocab("vocab.txt").unwrap();
    ///
    /// assert!(tokenizer.is_special(0i32)); // [PAD]
    /// assert!(tokenizer.is_special(3i32));  // [CLS]
    /// assert!(tokenizer.is_special(4i32));  // [SEP]
    /// assert!(!tokenizer.is_special(42i32));
    /// ```
    #[inline]
    pub fn is_special<T: TokenID>(&self, token_id: T) -> bool {
        self.special_tokens.contains(&token_id.restore())
    }

    /// Calculates the required attention for this token.
    ///
    /// # Examples
    ///
    /// ```
    /// use aleph_alpha_tokenizer::AlephAlphaTokenizer;
    ///
    /// let pad_attention: i64 = AlephAlphaTokenizer::attention(0u64);
    /// let token_attention: f64 = AlephAlphaTokenizer::attention(99i32);
    /// assert_eq!(pad_attention, 0);
    /// assert_eq!(token_attention, 1.0f64);
    /// ```
    #[inline]
    pub fn attention<T: TokenID, U: TokenID>(token_id: T) -> U {
        if token_id == T::zero() {
            U::zero()
        } else {
            U::coerce(1)
        }
    }

    /// Given a slice of `[u64]`s, appends the attentions to the given `Vec`.
    ///
    /// # Examples
    ///
    /// ```
    /// use aleph_alpha_tokenizer::AlephAlphaTokenizer;
    ///
    /// let mut attns: Vec<i32> = Vec::new();
    /// AlephAlphaTokenizer::attentions_into(&[3, 4285, 4, 0, 0], &mut attns);
    /// assert_eq!(&attns[..], &[1, 1, 1, 0, 0]);
    /// ```
    pub fn attentions_into<T: TokenID, U: TokenID>(token_ids: &[T], attns: &mut Vec<U>) {
		attns.clear();
        attns.extend(token_ids.iter().cloned().map(AlephAlphaTokenizer::attention));
    }
    
    /// Save the vocabulary back to a file
    pub fn save_vocab(&self, vocab_path: PathBuf) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
		let vocab = File::create(&vocab_path)?;
        let mut vocab_writer = BufWriter::new(vocab);
        for token in &self.tokens {
            writeln!(vocab_writer, "{}", token)?;
        }
        //TODO: write out FSTs to reduce load time
        Ok(vocab_path)
    }
}

#[cfg(feature = "huggingface")]
use std::{borrow::Cow, path::Path};

/// This type implements the [`Model`] trait so you can use it within
/// huggingface's tokenizers framework.
#[cfg(feature = "huggingface")]
impl Model for AlephAlphaTokenizer {
    fn tokenize(
        &self,
        tokens: Vec<(String, (usize, usize))>,
    ) -> Result<Vec<HfToken>, Box<dyn Error + Send + Sync>> {
        // we expect at least one token per word.
        let mut result = Vec::with_capacity(tokens.len());
        for (index, (word_str, offsets)) in tokens.into_iter().enumerate() {
            let word = index as u32;
            let word_index = result.len();
            let word_bytes = word_str.as_bytes();
            let word_len = word_bytes.len();
            let mut last_index = 0;
            if let Some((start_index, id)) = find_longest_prefix(&self.starters, word_bytes) {
                result.push(HfToken {
                    id: id as u32,
                    value: word_str[..start_index].to_string(),
                    offsets: (offsets.0, offsets.0 + start_index),
                    word,
                });
                last_index = start_index;
                while last_index < word_len {
                    if let Some((len, id)) =
                        find_longest_prefix(&self.followers, &word_bytes[last_index..])
                    {
                        let start = offsets.0 + last_index;
                        result.push(HfToken {
                            id: id as u32,
                            value: "##".to_string() + &word_str[last_index..last_index + len],
                            offsets: (start, start + len),
                            word,
                        });
                        last_index += len;
                    } else {
                        break;
                    }
                }
            }
            // in case we couldn't match the whole word, replace all we have so far with an [UNK] token
            if last_index < word_len {
                assert!(word_index <= result.len());
                result.truncate(word_index);
                result.push(HfToken {
                    id: self.unk_id,
                    value: "[UNK±".to_string(),
                    offsets: (offsets.0, offsets.1),
                    word,
                });
            }
        }
        Ok(result)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        if token.starts_with("##") {
            self.followers.get(&token[2..])
        } else {
            self.starters.get(token)
        }
        .map(|x| x.value() as u32)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokens.get(id as usize).cloned()
    }

    fn get_vocab_size(&self) -> usize {
        self.tokens.len()
    }

    fn save(
        &self,
        folder: &Path,
        name: Option<&str>,
    ) -> Result<Vec<PathBuf>, Box<dyn Error + Send + Sync>> {
        let vocab_name = name.map_or(Cow::Borrowed("vocab.txt"), |n| {
            Cow::Borrowed(n) + "-vocab.txt"
        });
        let mut vocab_path = folder.to_path_buf();
        vocab_path.push(&Path::new(vocab_name.as_ref()));
        self.save_vocab(vocab_path).map(|p| vec![p])
    }
}
