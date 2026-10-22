//! The `NGramHash` op: character n-grams of a text, hashed into vocabulary buckets.

use crate::murmur;
use crate::unicode::{self, Decoded};

/// Start-of-text marker added before the first character.
const PREFIX: u8 = b'^';
/// End-of-text marker added after the last character.
const SUFFIX: u8 = b'$';
/// Token that replaces every non-alphabetic character.
const SEPARATOR: u8 = b' ';

/// Parameters of an `NGramHash` op, read from its custom options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NgramHash {
    pub(crate) seed: u64,
    /// N-gram length (in characters) and vocabulary size of every output row.
    pub(crate) heads: Box<[(usize, u64)]>,
    /// Maximum number of tokens, including the prefix and suffix markers.
    pub(crate) max_splits: usize,
    pub(crate) lowercase: bool,
}

impl NgramHash {
    /// Returns one row of bucket ids per head, each with one id per token.
    ///
    /// Ids are in `1..=vocab_size`; 0 is reserved for padding.
    pub(crate) fn hash(&self, text: &str) -> Vec<Vec<u32>> {
        let input = if self.lowercase {
            lowercase(text)
        } else {
            text.as_bytes().to_vec()
        };
        let tokens = tokenize(&input, text.len(), self.max_splits);
        self.heads
            .iter()
            .map(|&(n, vocab)| {
                (0..tokens.len())
                    .map(|start| {
                        let end = (start + n).min(tokens.len());
                        let bytes = &tokens.text[tokens.offsets[start]..tokens.offsets[end]];
                        let bucket = murmur::hash64(bytes, self.seed) % vocab;
                        u32::try_from(bucket + 1).expect("vocabulary sizes are validated to fit in i32")
                    })
                    .collect()
            })
            .collect()
    }
}

/// `LowercaseUnicodeStr`: lowercases alphabetic characters with the Plan 9 tables.
fn lowercase(text: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    for c in text.chars() {
        let c = u32::from(c);
        unicode::encode(if unicode::is_alpha(c) { unicode::to_lower(c) } else { c }, &mut out);
    }
    out
}

/// Tokens of a text: their concatenation and the byte offset where each one starts.
struct Tokens {
    text: Vec<u8>,
    /// `offsets[i]..offsets[i + 1]` is token `i`; the last entry is `text.len()`.
    offsets: Vec<usize>,
}

impl Tokens {
    fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    fn push(&mut self, bytes: &[u8]) {
        self.text.extend_from_slice(bytes);
        self.offsets.push(self.text.len());
    }
}

/// `Tokenize(input, len, max_tokens, exclude_nonalphaspace_tokens = true)`.
///
/// One token per character between `^` and `$`, non-alphabetic characters replaced by a
/// space. MediaPipe passes the length of the original text while reading the lowercased
/// one; lowercasing can shorten it (e.g. `İ` -> `i`), and the bytes read past its end are
/// the string's NUL terminator and whatever follows it. Those are modeled as NUL bytes,
/// which become separators.
fn tokenize(input: &[u8], len: usize, max_tokens: usize) -> Tokens {
    let mut window = input[..len.min(input.len())].to_vec();
    window.resize(len, 0);

    let mut tokens = Tokens {
        text: Vec::with_capacity(len + 2),
        offsets: vec![0],
    };
    tokens.push(&[PREFIX]);
    let mut i = 0;
    while i < window.len() && tokens.len() + 1 < max_tokens {
        let Decoded::Rune(rune, read) = unicode::decode(&window[i..]) else {
            break;
        };
        if unicode::is_alpha(rune) {
            tokens.push(&window[i..i + read]);
        } else {
            tokens.push(&[SEPARATOR]);
        }
        i += read;
    }
    tokens.push(&[SUFFIX]);
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(text: &str, max_tokens: usize) -> String {
        let t = tokenize(text.as_bytes(), text.len(), max_tokens);
        assert_eq!(t.offsets.len(), t.len() + 1);
        String::from_utf8(t.text).unwrap()
    }

    #[test]
    fn tokenizes_like_mediapipe() {
        // Cases from MediaPipe's ngram_hash_ops_utils_test.cc.
        assert_eq!(tokens("hi!", 100), "^hi $");
        assert_eq!(tokens("har ki", 100), "^har ki$");
        assert_eq!(tokens("hi!", 3), "^h$");
        assert_eq!(tokens("ありがと", 100), "^ありがと$");
        assert_eq!(tokens("12 😀", 100), "^    $");
    }

    #[test]
    fn every_character_is_one_token() {
        let t = tokenize("aж中".as_bytes(), "aж中".len(), 100);
        let parts: Vec<&[u8]> = t.offsets.windows(2).map(|w| &t.text[w[0]..w[1]]).collect();
        assert_eq!(parts, [&b"^"[..], b"a", "ж".as_bytes(), "中".as_bytes(), b"$"]);
    }

    #[test]
    fn lowercases_letters_only() {
        assert_eq!(lowercase("hElLo БЙп 1!"), "hello бйп 1!".as_bytes());
        assert_eq!(lowercase("ありがと"), "ありがと".as_bytes());
    }

    #[test]
    fn shortened_lowercase_text_is_padded_with_separators() {
        // "İ" is two bytes, its lowercase "i" one: the window gains a NUL separator.
        let text = "İz";
        let lower = lowercase(text);
        assert_eq!(lower, b"iz");
        assert_eq!(
            String::from_utf8(tokenize(&lower, text.len(), 100).text).unwrap(),
            "^iz $"
        );
    }

    #[test]
    fn hashes_every_ngram_into_its_vocabulary() {
        let op = NgramHash {
            seed: 10_911,
            heads: Box::new([(1, 50), (3, 7)]),
            max_splits: 128,
            lowercase: true,
        };
        let rows = op.hash("Hi");
        assert_eq!(rows.len(), 2);
        // "^", "h", "i", "$".
        assert!(rows.iter().all(|row| row.len() == 4));
        assert!(rows[0].iter().all(|&id| (1..=50).contains(&id)));
        assert!(rows[1].iter().all(|&id| (1..=7).contains(&id)));
        // Lowercasing happens before hashing.
        assert_eq!(op.hash("HI"), op.hash("hi"));
    }
}
