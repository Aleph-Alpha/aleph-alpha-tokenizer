#![no_main]

use std::sync::{Arc, RwLock};
use aleph_alpha_tokenizer::AlephAlphaTokenizer;
use libfuzzer_sys::fuzz_target;
use once_cell::sync::Lazy;
use tokenizers::{
    tokenizer::{EncodeInput, Tokenizer},
    models::wordpiece::WordPiece,
};

static ALEPH: Lazy<Arc<RwLock<Tokenizer>>> = Lazy::new(|| Arc::new(RwLock::new(Tokenizer::new(
    Box::new(AlephAlphaTokenizer::from_vocab("vocab.txt").unwrap())))));
        
static WORDPIECE: Lazy<Arc<RwLock<Tokenizer>>> = Lazy::new(|| Arc::new(RwLock::new(Tokenizer::new(
        Box::new(WordPiece::from_files("vocab.txt").build().unwrap())))));

// check if the string contains words larger than the character limit
fn too_long_word(s: &str) -> bool {
    s.split(char::is_whitespace).any(|w| w.chars().count() >= 100)
}

fuzz_target!(|s: String| {
    // wordpiece starts with a follower token if a word starts with '##'.
    // Also we don't store `[unusedX]` tokens, so those don't get matched.
    // Finally we don't share wordpiece's character limit
    if s.contains("##") || s.contains("[unused") || too_long_word(&s) { return; }
    let input = EncodeInput::Single(s);
    let aleph = ALEPH.read().unwrap().encode(input.clone(), true).ok();
    let wordpiece = WORDPIECE.read().unwrap().encode(input, true).ok();
    assert_eq!(aleph, wordpiece);
});
