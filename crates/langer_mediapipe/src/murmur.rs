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
    // Lossless: usize is at most 64 bits on every target Rust supports.
    let len = data.len() as u64;
    let mut hash = seed ^ len.wrapping_mul(MUL);

    let (words, tail) = data.as_chunks::<8>();
    for word in words {
        let v = u64::from_le_bytes(*word);
        hash ^= shift_mix(v.wrapping_mul(MUL)).wrapping_mul(MUL);
        hash = hash.wrapping_mul(MUL);
    }

    if !tail.is_empty() {
        let mut last = [0; 8];
        for (byte, &tail_byte) in last.iter_mut().zip(tail) {
            *byte = tail_byte;
        }
        hash ^= u64::from_le_bytes(last);
        hash = hash.wrapping_mul(MUL);
    }

    shift_mix(shift_mix(hash).wrapping_mul(MUL))
}

#[cfg(test)]
mod tests {
    use super::hash64;

    #[test]
    fn hash64_of_empty_input_with_zero_seed_is_zero() {
        // From MediaPipe's murmur_test.cc.
        assert_eq!(hash64(&[], 0), 0);
    }

    #[test]
    fn hash64_depends_on_seed_every_byte_and_length() {
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
