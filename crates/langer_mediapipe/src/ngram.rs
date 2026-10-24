//! The `NGramHash` op: character n-grams of a text, hashed into vocabulary buckets.

use crate::PredictError;
use crate::murmur;
use crate::unicode::{self, Decoded};

/// Start-of-text marker added before the first character.
const PREFIX: u8 = b'^';
/// End-of-text marker added after the last character.
const SUFFIX: u8 = b'$';
/// Token that replaces every non-alphabetic character.
const SEPARATOR: u8 = b' ';

/// Characters of the text kept per token, see [`NgramHash::hash`].
const CHARS_PER_TOKEN: usize = 2;

/// Parameters of an `NGramHash` op, read from its custom options.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct NgramHash {
    pub(crate) seed: u64,
    /// N-gram length (in characters) and vocabulary size of every output row.
    pub(crate) heads: Box<[(usize, u32)]>,
    /// Maximum number of tokens, including the prefix and suffix markers.
    pub(crate) max_splits: usize,
    pub(crate) lowercase: bool,
}

impl NgramHash {
    /// Returns one row of bucket ids per head, each with one id per token.
    ///
    /// Ids are in `1..=vocab_size`; 0 is reserved for padding.
    ///
    /// Only the first `max_splits` characters of the text become tokens, so only a prefix of
    /// [`CHARS_PER_TOKEN`] characters per token is read: hashing a long text then costs as much
    /// as hashing a short one. The prefix is long enough for any text, as lowercasing changes a
    /// character's UTF-8 length by at most half (e.g. "Ⱥ", two bytes, becomes three).
    pub(crate) fn hash(&self, text: &str) -> Result<Vec<Vec<u32>>, PredictError> {
        let prefix = text
            .char_indices()
            .nth(CHARS_PER_TOKEN.saturating_mul(self.max_splits))
            .and_then(|(end, _)| text.get(..end))
            .unwrap_or(text);

        self.hash_all(prefix)
    }

    /// [`Self::hash`] over the whole text.
    fn hash_all(&self, text: &str) -> Result<Vec<Vec<u32>>, PredictError> {
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
                        let bytes = tokens
                            .span(start, (start + n).min(tokens.len()))
                            .ok_or_else(|| PredictError::new("n-gram outside the tokens"))?;

                        let bucket = murmur::hash64(bytes, self.seed) % u64::from(vocab);
                        #[expect(
                            clippy::cast_possible_truncation,
                            reason = "the remainder is smaller than the vocabulary size, a u32"
                        )]
                        let bucket = bucket as u32;
                        // `bucket < vocab`, so this cannot overflow.
                        Ok(bucket + 1)
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

    /// Bytes of tokens `start..end`.
    fn span(&self, start: usize, end: usize) -> Option<&[u8]> {
        self.text.get(*self.offsets.get(start)?..*self.offsets.get(end)?)
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
    let mut window = input.get(..len).unwrap_or(input).to_vec();
    window.resize(len, 0);

    let mut tokens = Tokens {
        text: Vec::with_capacity(len + 2),
        offsets: vec![0],
    };
    tokens.push(&[PREFIX]);

    let mut i = 0;
    while let Some(rest) = window.get(i..).filter(|rest| !rest.is_empty())
        && tokens.len() + 1 < max_tokens
    {
        let Decoded::Rune(rune, read) = unicode::decode(rest) else {
            break;
        };
        if unicode::is_alpha(rune)
            && let Some(character) = rest.get(..read)
        {
            tokens.push(character);
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
    use proptest::prelude::*;
    use rstest::rstest;

    use super::*;

    fn tokens(text: &str, max_tokens: usize) -> Vec<String> {
        let t = tokenize(text.as_bytes(), text.len(), max_tokens);
        t.offsets
            .windows(2)
            .map(|w| String::from_utf8(t.text[w[0]..w[1]].to_vec()).expect("tokens of valid UTF-8 are valid"))
            .collect()
    }

    fn op() -> NgramHash {
        NgramHash {
            seed: 10_911,
            heads: Box::new([(1, 50), (3, 7)]),
            max_splits: 128,
            lowercase: true,
        }
    }

    // Cases from MediaPipe's ngram_hash_ops_utils_test.cc.
    #[rstest]
    #[case::punctuation_becomes_separator("hi!", 100, "^hi $")]
    #[case::space_is_kept_as_separator("har ki", 100, "^har ki$")]
    #[case::cut_at_max_tokens("hi!", 3, "^h$")]
    #[case::non_latin_letters("ありがと", 100, "^ありがと$")]
    #[case::no_letters("12 😀", 100, "^    $")]
    fn tokenize_matches_mediapipe(#[case] text: &str, #[case] max_tokens: usize, #[case] expected: &str) {
        assert_eq!(tokens(text, max_tokens).concat(), expected);
    }

    #[test]
    fn tokenize_makes_one_token_per_character() {
        assert_eq!(tokens("aж中", 100), ["^", "a", "ж", "中", "$"]);
    }

    #[test]
    fn tokenize_pads_shortened_lowercase_text_with_separators() {
        // "İ" is two bytes and its lowercase "i" one: the window gains a NUL, a separator.
        let text = "İz";
        let lower = lowercase(text);
        assert_eq!(lower, b"iz");
        let t = tokenize(&lower, text.len(), 100);
        assert_eq!(t.text, b"^iz $");
    }

    #[rstest]
    #[case::latin_and_cyrillic("hElLo БЙп", "hello бйп")]
    #[case::digits_and_punctuation("A1!", "a1!")]
    #[case::caseless("ありがと", "ありがと")]
    fn lowercase_changes_letters_only(#[case] text: &str, #[case] expected: &str) {
        assert_eq!(lowercase(text), expected.as_bytes());
    }

    #[test]
    fn hash_ignores_letter_case() {
        assert_eq!(op().hash("HI").unwrap(), op().hash("hi").unwrap());
    }

    proptest! {
        #[test]
        fn tokenize_wraps_characters_in_markers(text in any::<String>(), max_tokens in 2_usize..200) {
            let tokens = tokens(&text, max_tokens);
            let characters = text.chars().count().min(max_tokens - 2);
            prop_assert_eq!(tokens.len(), characters + 2);
            prop_assert_eq!(tokens.first().map(String::as_str), Some("^"));
            prop_assert_eq!(tokens.last().map(String::as_str), Some("$"));
            for (token, c) in tokens[1..=characters].iter().zip(text.chars()) {
                let expected = if unicode::is_alpha(u32::from(c)) { c.to_string() } else { " ".to_owned() };
                prop_assert_eq!(token, &expected);
            }
        }

        #[test]
        fn hash_of_a_prefix_equals_hash_of_the_whole_text(
            // Characters whose lowercase form is longer or shorter, among any others.
            chars in proptest::collection::vec(
                prop_oneof![any::<char>(), Just('Ⱥ'), Just('\u{212A}'), Just('İ'), Just('a'), Just(' ')],
                0..200,
            ),
            max_splits in 3_usize..40,
        ) {
            check_prefix_hash(&chars.into_iter().collect::<String>(), max_splits)?;
        }

        #[test]
        fn hash_puts_every_ngram_into_its_vocabulary(text in any::<String>()) {
            check_hash_rows(&text)?;
        }
    }

    /// The prefix [`NgramHash::hash`] reads gives the same ids as the whole text.
    fn check_prefix_hash(text: &str, max_splits: usize) -> Result<(), TestCaseError> {
        let op = NgramHash { max_splits, ..op() };
        prop_assert_eq!(op.hash(text).unwrap(), op.hash_all(text).unwrap());

        Ok(())
    }

    /// Every head has one id per token, in its vocabulary.
    fn check_hash_rows(text: &str) -> Result<(), TestCaseError> {
        let rows = op().hash(text).unwrap();

        // Lowercasing changes byte lengths and MediaPipe reads the original byte length of the
        // lowercased text: a longer character (e.g. "Ⱥ" -> "ⱥ") is cut off, a shorter one
        // (e.g. Kelvin sign -> "k") leaves NUL separators. Only the byte length bounds tokens.
        let max_tokens = text.len().min(126) + 2;
        prop_assert_eq!(rows.len(), 2);
        prop_assert_eq!(rows[0].len(), rows[1].len());
        for (row, &(_, vocab)) in rows.iter().zip(op().heads.iter()) {
            prop_assert!((2..=max_tokens).contains(&row.len()));
            prop_assert!(row.iter().all(|&id| (1..=vocab).contains(&id)));
        }

        Ok(())
    }

    // Cases proptest once found: characters whose lowercase form has another UTF-8 length.
    #[rstest]
    #[case::lowercase_is_longer("Ⱥ")]
    #[case::lowercase_is_shorter("\u{212A}")]
    fn hash_puts_every_ngram_into_its_vocabulary_when_lowercase_changes_length(#[case] text: &str) {
        check_hash_rows(text).unwrap();
    }

    #[test]
    fn hash_of_a_prefix_equals_hash_of_the_whole_text_with_lengths_changing_past_the_prefix() {
        // Found by proptest: with fewer than two characters per token kept, the tokens of the
        // prefix differ once lowercasing lengthens and shortens characters around its end.
        check_prefix_hash("ȺaaaaȺȺȺȺaȺaaaȺ\0ȺİaaaȺİȺ¡aİȺa¡00 00¡ ¡", 10).unwrap();
    }
}
