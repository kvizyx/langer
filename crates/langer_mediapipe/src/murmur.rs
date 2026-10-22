//! MediaPipe's `MurmurHash64WithSeed` (`custom_ops/utils/hash/murmur.cc`).
//!
//! A MurmurHash2-64A variant: the tail is mixed in as one little-endian word and the result
//! is finalized with two shift-mixes. It differs from crates implementing MurmurHash2/3, and
//! the model's n-gram buckets depend on it bit for bit.

const MUL: u64 = 0xc6a4_a793_5bd1_e995;

#[inline]
fn shift_mix(v: u64) -> u64 {
    v ^ (v >> 47)
}

/// Hashes `data` with `seed`.
pub(crate) fn hash64(data: &[u8], seed: u64) -> u64 {
    let len = u64::try_from(data.len()).expect("slice length fits in u64");
    let mut hash = seed ^ len.wrapping_mul(MUL);
    let (words, tail) = data.as_chunks::<8>();
    for word in words {
        let v = u64::from_le_bytes(*word);
        hash ^= shift_mix(v.wrapping_mul(MUL)).wrapping_mul(MUL);
        hash = hash.wrapping_mul(MUL);
    }
    if !tail.is_empty() {
        let mut last = [0; 8];
        last[..tail.len()].copy_from_slice(tail);
        hash ^= u64::from_le_bytes(last);
        hash = hash.wrapping_mul(MUL);
    }
    shift_mix(shift_mix(hash).wrapping_mul(MUL))
}

#[cfg(test)]
mod tests {
    use super::hash64;

    #[test]
    fn empty_input_with_zero_seed_is_zero() {
        // From MediaPipe's murmur_test.cc.
        assert_eq!(hash64(&[], 0), 0);
    }

    #[test]
    fn seed_and_every_byte_change_the_hash() {
        let data = b"hello world, this crosses a word boundary";
        assert_ne!(hash64(&data[..1], 100), hash64(&data[..1], 101));
        let base = hash64(data, 7);
        for i in 0..data.len() {
            let mut changed = data.to_vec();
            changed[i] ^= 1;
            assert_ne!(hash64(&changed, 7), base, "byte {i}");
        }
        // Length is part of the hash even when trailing bytes are zero.
        assert_ne!(hash64(&[0; 3], 7), hash64(&[0; 4], 7));
    }
}
