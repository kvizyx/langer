//! Numeric ops of the model, following the TFLite reference kernels.

use crate::PredictError;

/// `KmeansEmbeddingLookup`: averages the embeddings of the ids in a row.
///
/// An embedding is split into `codes_per_id` blocks; each block is a codebook vector
/// selected by one byte of the id's code.
#[derive(Debug)]
pub(crate) struct KmeansEmbedding {
    /// `codes[id * codes_per_id + j]` is the codebook index of block `j` of `id`.
    pub(crate) codes: Box<[u8]>,
    pub(crate) codes_per_id: usize,
    /// `codebook[index * block + k]`.
    pub(crate) codebook: Box<[f32]>,
    pub(crate) block: usize,
}

impl KmeansEmbedding {
    pub(crate) fn width(&self) -> usize {
        self.codes_per_id * self.block
    }

    pub(crate) fn ids(&self) -> usize {
        self.codes.len() / self.codes_per_id
    }

    pub(crate) fn heap_size(&self) -> usize {
        self.codes.len() + size_of_val(&*self.codebook)
    }

    /// Appends the mean embedding of `ids` to `out`. Id 0 ends the row, as padding.
    pub(crate) fn lookup_into(&self, ids: &[u32], out: &mut Vec<f32>) -> Result<(), PredictError> {
        let mut sum = vec![0.0; self.width()];
        let mut count = 0_usize;
        for &id in ids.iter().take_while(|&&id| id != 0) {
            count += 1;
            let codes = self
                .codes
                .chunks_exact(self.codes_per_id)
                .nth(id as usize)
                .ok_or_else(|| PredictError::new("n-gram id outside the embedding table"))?;

            for (block, &code) in sum.chunks_exact_mut(self.block).zip(codes) {
                let centroid = self
                    .codebook
                    .chunks_exact(self.block)
                    .nth(usize::from(code))
                    .ok_or_else(|| PredictError::new("embedding code outside the codebook"))?;

                for (s, c) in block.iter_mut().zip(centroid) {
                    *s += c;
                }
            }
        }

        #[expect(
            clippy::cast_precision_loss,
            reason = "the id count is bounded by max_splits, far below 2^24"
        )]
        let count = count.max(1) as f32;
        out.extend(sum.iter().map(|s| s / count));

        Ok(())
    }
}

/// Most inputs of a [`HybridDense`] layer whose `i32` accumulator cannot overflow.
///
/// Every product of an `i8` weight and an `i8` input is at most 2^14 in magnitude, and so is
/// every term of the zero point correction: both sums stay below 2^14 * 65 535 and their
/// difference within `i32`. MediaPipe's layers have a few hundred inputs.
pub(crate) const MAX_DENSE_INPUTS: usize = 65_535;

/// Fully connected layer with `i8` weights applied to `f32` input ("hybrid" layer).
///
/// MediaPipe runs these layers through TFLite's XNNPACK delegate, which quantizes the input
/// to `i8` per call (`qd8`) and multiplies in integers; this mirrors its kernels
/// (`xnn_qd8_asymmetric_quantization_params`, `f32-qs8-vcvt`, `qd8-f32-qc8w-gemm`). The
/// output is rescaled with a fused multiply-add, as XNNPACK's x86 AVX2 and Arm kernels do.
#[derive(Debug)]
pub(crate) struct HybridDense {
    /// Row-major `outputs x inputs` weights.
    pub(crate) weights: Box<[i8]>,
    pub(crate) inputs: usize,
    pub(crate) weight_scale: f32,
    pub(crate) bias: Box<[f32]>,
    pub(crate) relu: bool,
    /// Sum of every weight row, used to correct for the input zero point.
    pub(crate) row_sums: Box<[i32]>,
}

impl HybridDense {
    pub(crate) fn new(weights: Box<[i8]>, inputs: usize, weight_scale: f32, bias: Box<[f32]>, relu: bool) -> Self {
        let row_sums = weights
            .chunks_exact(inputs)
            .map(|row| row.iter().map(|&w| i32::from(w)).sum())
            .collect();
        Self {
            weights,
            inputs,
            weight_scale,
            bias,
            relu,
            row_sums,
        }
    }

    pub(crate) fn outputs(&self) -> usize {
        self.bias.len()
    }

    pub(crate) fn heap_size(&self) -> usize {
        self.weights.len() + size_of_val(&*self.bias) + size_of_val(&*self.row_sums)
    }

    pub(crate) fn apply(&self, input: &[f32]) -> Vec<f32> {
        let (quantized, input_scale, zero_point) = quantize_dynamic(input);

        let rows = self.weights.chunks_exact(self.inputs).zip(&self.row_sums);
        rows.zip(&self.bias)
            .map(|((row, &row_sum), &bias)| {
                let dot: i32 = row
                    .iter()
                    .zip(&quantized)
                    .map(|(&w, &q)| i32::from(w) * i32::from(q))
                    .sum();

                #[expect(
                    clippy::cast_precision_loss,
                    reason = "XNNPACK converts the i32 accumulator to f32 the same way"
                )]
                let acc = (dot - row_sum * zero_point) as f32;
                let out = (acc * input_scale).mul_add(self.weight_scale, bias);
                if self.relu { out.max(0.0) } else { out }
            })
            .collect()
    }
}

/// XNNPACK's asymmetric dynamic quantization of `values` to `i8`.
///
/// Returns the quantized values, the scale that maps them back to floats, and the zero
/// point. 0.0 is exactly representable; rounding is to nearest, ties to even (`lrintf`).
fn quantize_dynamic(values: &[f32]) -> (Vec<i8>, f32, i32) {
    const QMIN: f32 = -128.0;
    const QMAX: f32 = 127.0;

    let rmin = values.iter().copied().fold(0.0_f32, f32::min);
    let rmax = values.iter().copied().fold(0.0_f32, f32::max);
    // `rmin == rmax` only when every value is zero.
    let multiplier = if rmin < rmax {
        (QMAX - QMIN) / (rmax - rmin)
    } else {
        1.0
    };

    let descaled_min = rmin * multiplier;
    let descaled_max = rmax * multiplier;
    let zero_point = if (QMIN + descaled_min) + (QMAX + descaled_max) > 0.0 {
        QMIN - descaled_min
    } else {
        QMAX - descaled_max
    };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the zero point is clamped to the i8 range first"
    )]
    let zero_point = zero_point.clamp(QMIN, QMAX).round_ties_even() as i32;
    #[expect(clippy::cast_precision_loss, reason = "the zero point is in -128..=127")]
    let zero_point_f = zero_point as f32;

    let quantized = values
        .iter()
        .map(|&v| {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the value is clamped to the i8 range first"
            )]
            let q = ((v * multiplier).round_ties_even() + zero_point_f).clamp(QMIN, QMAX) as i8;
            q
        })
        .collect();

    (quantized, 1.0 / multiplier, zero_point)
}

/// TFLite's float `Softmax` with `beta`, in place.
pub(crate) fn softmax(values: &mut [f32], beta: f32) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in values.iter_mut() {
        *v = ((*v - max) * beta).exp();
        sum += *v;
    }

    for v in values.iter_mut() {
        *v /= sum;
    }
}

#[cfg(test)]
mod tests {
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::*;

    /// Maps quantized values back to floats.
    fn dequantize(values: &[f32]) -> (Vec<f32>, f32) {
        let (quantized, scale, zero_point) = quantize_dynamic(values);
        let zero_point = f32::from(i16::try_from(zero_point).expect("zero point is in the i8 range"));
        let restored = quantized.iter().map(|&q| (f32::from(q) - zero_point) * scale).collect();
        (restored, scale)
    }

    #[test]
    fn quantize_dynamic_maps_extremes_to_the_i8_range() {
        let (quantized, _, _) = quantize_dynamic(&[-1.0, 0.0, 0.5, 3.0]);
        assert_eq!((quantized[0], quantized[3]), (-128, 127));
    }

    #[test]
    fn hybrid_dense_approximates_the_float_product() {
        let weights: Box<[i8]> = Box::new([10, -20, 30, 5, 5, 5]);
        let linear = HybridDense::new(weights, 3, 0.1, Box::new([0.5, -100.0]), false);
        let out = linear.apply(&[1.0, 2.0, -0.5]);
        // Exact: 0.5 + 0.1 * (10 - 40 - 15) and -100 + 0.1 * (5 + 10 - 2.5).
        assert!((out[0] + 4.0).abs() < 0.05, "{out:?}");
        assert!((out[1] + 98.75).abs() < 0.05, "{out:?}");
    }

    #[test]
    fn hybrid_dense_does_not_overflow_at_the_largest_supported_width() {
        // The worst case: every product and the zero point correction at their extremes,
        // with opposite signs. Debug builds panic on overflow.
        let weights: Box<[i8]> = vec![-128; MAX_DENSE_INPUTS].into();
        let linear = HybridDense::new(weights, MAX_DENSE_INPUTS, 1.0, Box::new([0.0]), false);
        let mut input = vec![-1.0; MAX_DENSE_INPUTS];
        input[0] = 1e-6;
        let out = linear.apply(&input);
        assert!(out[0].is_finite(), "{out:?}");
        let out = linear.apply(&vec![1.0; MAX_DENSE_INPUTS]);
        assert!(out[0].is_finite(), "{out:?}");
    }

    #[test]
    fn hybrid_dense_with_relu_clamps_negative_outputs() {
        let weights: Box<[i8]> = Box::new([10, -20, 30, 5, 5, 5]);
        let relu = HybridDense::new(weights, 3, 0.1, Box::new([0.5, -100.0]), true);
        assert_eq!(relu.apply(&[1.0, 2.0, -0.5]), [0.0, 0.0]);
        // An all-zero input yields the (clamped) bias.
        assert_eq!(relu.apply(&[0.0; 3]), [0.5, 0.0]);
    }

    #[test]
    fn kmeans_embedding_averages_codebook_blocks_until_padding() {
        let embedding = KmeansEmbedding {
            // Ids 0..=2, two blocks each.
            codes: Box::new([0, 0, 0, 1, 1, 0]),
            codes_per_id: 2,
            codebook: Box::new([1.0, 2.0, 10.0, 20.0]),
            block: 2,
        };
        let mut out = vec![7.0];
        embedding.lookup_into(&[1, 2, 0, 1], &mut out).unwrap();
        // Id 1 -> [1, 2, 10, 20], id 2 -> [10, 20, 1, 2]; the 0 stops the row.
        assert_eq!(out, [7.0, 5.5, 11.0, 5.5, 11.0]);
    }

    proptest! {
        #[test]
        #[expect(clippy::float_cmp, reason = "zero must be represented exactly")]
        fn quantize_dynamic_loses_at_most_one_step(values in vec(-100.0_f32..100.0, 1..64)) {
            let (restored, scale) = dequantize(&values);
            for (&value, &restored) in values.iter().zip(&restored) {
                prop_assert!((value - restored).abs() <= scale, "{value} -> {restored}, scale {scale}");
            }
            let (zero, _) = dequantize(&[0.0]);
            prop_assert_eq!(zero[0], 0.0);
        }

        #[test]
        fn softmax_returns_a_distribution_in_input_order(
            mut values in vec(-50.0_f32..50.0, 1..200),
            beta in 0.1_f32..2.0,
        ) {
            let input = values.clone();
            softmax(&mut values, beta);
            prop_assert!((values.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            prop_assert!(values.iter().all(|p| (0.0..=1.0).contains(p)));
            for (i, j) in (0..input.len()).flat_map(|i| (0..input.len()).map(move |j| (i, j))) {
                if input[i] > input[j] {
                    prop_assert!(values[i] >= values[j]);
                }
            }
        }
    }
}
