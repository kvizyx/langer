use std::collections::HashMap;
use std::fmt;
use std::io::{Cursor, Read};

use crate::layers::{HybridDense, KmeansEmbedding, MAX_DENSE_INPUTS, softmax};
use crate::ngram::NgramHash;
use crate::schema_generated::tflite as fb;
use crate::{LoadError, PredictError};

/// MediaPipe's language detector model (`language_detector.tflite`, float32, version 1);
/// see `README.md`.
const BUNDLED_MODEL: &[u8] = include_bytes!("../vendor/model/language_detector.tflite");

/// File in the zip archive appended to the model that lists the labels, one per line.
const LABELS_FILE: &str = "labels.txt";

/// `NGramHash` default when the op options omit `max_splits`.
const DEFAULT_MAX_SPLITS: usize = 128;

/// A language predicted for a text, with its probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction<'a> {
    /// Language code as reported by MediaPipe, e.g. `en`, `iw`, `zh-Latn` or `unknown`.
    pub language: &'a str,
    /// Probability in `0.0..=1.0`.
    pub probability: f32,
}

/// MediaPipe's language detector.
///
/// Immutable after loading and cheap to share between threads.
pub struct Detector {
    ngram: NgramHash,
    /// Embeddings in concatenation order, each with the `NGramHash` head it reads.
    embeddings: Box<[(usize, KmeansEmbedding)]>,
    dense: Box<[HybridDense]>,
    softmax_beta: f32,
    labels: Box<[Box<str>]>,
}

impl Detector {
    /// Creates a detector from the bundled MediaPipe model.
    ///
    /// # Errors
    ///
    /// Returns an error only if the bundled model fails to load, which tests rule out.
    pub fn new() -> Result<Self, LoadError> {
        Self::from_bytes(BUNDLED_MODEL)
    }

    /// Loads a MediaPipe language detector model from the bytes of its `.tflite` file.
    ///
    /// # Errors
    ///
    /// Returns an error if the data is not a TFLite model with the structure of MediaPipe's
    /// language detector (`NGramHash`, `KmeansEmbeddingLookup`, hybrid fully connected
    /// layers, softmax) and a `labels.txt` file appended.
    pub fn from_bytes(model: &[u8]) -> Result<Self, LoadError> {
        let labels = read_labels(model)?;
        let root = fb::root_as_model(model).map_err(|e| LoadError::with_source("not a TFLite model", e))?;
        let graph = Graph::new(root, model)?;
        let detector = graph.detector(labels)?;
        Ok(detector)
    }

    /// Returns the language codes the detector can report, in model order.
    pub fn labels(&self) -> impl ExactSizeIterator<Item = &str> {
        self.labels.iter().map(AsRef::as_ref)
    }

    /// Returns the approximate heap memory held by the model, in bytes.
    #[must_use]
    pub fn heap_size(&self) -> usize {
        let embeddings: usize = self.embeddings.iter().map(|(_, e)| e.heap_size()).sum();
        let dense: usize = self.dense.iter().map(HybridDense::heap_size).sum();
        let labels: usize = self.labels.iter().map(|l| l.len()).sum();
        embeddings + dense + labels
    }

    /// Returns the `k` most probable languages of `text`, most probable first.
    ///
    /// Any text has a prediction, as in MediaPipe: an empty one is classified by the start
    /// and end markers alone.
    ///
    /// # Errors
    ///
    /// Returns an error only if the model is internally inconsistent, which loading checks.
    pub fn predict(&self, text: &str, k: usize) -> Result<Vec<Prediction<'_>>, PredictError> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let ids = self.ngram.hash(text)?;
        let width = self.embeddings.iter().map(|(_, e)| e.width()).sum();
        let mut features = Vec::with_capacity(width);
        for (head, embedding) in &self.embeddings {
            let head_ids = ids
                .get(*head)
                .ok_or_else(|| PredictError::new("embedding reads a missing n-gram head"))?;
            embedding.lookup_into(head_ids, &mut features)?;
        }
        let mut scores = self.dense.iter().fold(features, |x, layer| layer.apply(&x));
        softmax(&mut scores, self.softmax_beta);

        let mut ranked: Vec<Prediction<'_>> = self
            .labels
            .iter()
            .zip(scores)
            .map(|(language, probability)| Prediction { language, probability })
            .collect();
        // Stable, so ties keep model order.
        ranked.sort_by(|a, b| b.probability.total_cmp(&a.probability));
        ranked.truncate(k);
        Ok(ranked)
    }
}

impl fmt::Debug for Detector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Detector")
            .field("labels", &self.labels.len())
            .field("heap_size", &self.heap_size())
            .finish_non_exhaustive()
    }
}

fn read_labels(model: &[u8]) -> Result<Box<[Box<str>]>, LoadError> {
    let mut archive =
        zip::ZipArchive::new(Cursor::new(model)).map_err(|e| LoadError::with_source("no labels archive", e))?;
    let mut file = archive
        .by_name(LABELS_FILE)
        .map_err(|e| LoadError::with_source("no labels.txt in the labels archive", e))?;
    let mut text = String::new();
    file.read_to_string(&mut text)
        .map_err(|e| LoadError::with_source("unreadable labels.txt", e))?;
    Ok(text.lines().map(|l| l.trim().into()).collect())
}

/// The model graph, with each operator's kind resolved.
struct Graph<'a> {
    file: &'a [u8],
    model: fb::Model<'a>,
    subgraph: fb::SubGraph<'a>,
    tensors: flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<fb::Tensor<'a>>>,
    ops: Vec<(Kind<'a>, fb::Operator<'a>)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind<'a> {
    Builtin(fb::BuiltinOperator),
    Custom(&'a str),
}

impl<'a> Graph<'a> {
    fn new(model: fb::Model<'a>, file: &'a [u8]) -> Result<Self, LoadError> {
        let subgraphs = model.subgraphs().ok_or_else(|| LoadError::new("no subgraphs"))?;
        if subgraphs.len() != 1 {
            return Err(LoadError::new("expected exactly one subgraph"));
        }
        let subgraph = subgraphs.get(0);
        let tensors = subgraph.tensors().ok_or_else(|| LoadError::new("no tensors"))?;
        let codes = model
            .operator_codes()
            .ok_or_else(|| LoadError::new("no operator codes"))?;
        let mut ops = Vec::new();
        for op in subgraph.operators().ok_or_else(|| LoadError::new("no operators"))? {
            let index = op.opcode_index() as usize;
            if index >= codes.len() {
                return Err(LoadError::new("operator code index out of range"));
            }
            let code = codes.get(index);
            // TFLite resolves the code as the larger of the two fields.
            let builtin = fb::BuiltinOperator(code.builtin_code().0.max(i32::from(code.deprecated_builtin_code())));
            let kind = match code.custom_code() {
                Some(name) if builtin == fb::BuiltinOperator::CUSTOM => Kind::Custom(name),
                _ => Kind::Builtin(builtin),
            };
            ops.push((kind, op));
        }
        Ok(Self {
            file,
            model,
            subgraph,
            tensors,
            ops,
        })
    }

    fn detector(&self, labels: Box<[Box<str>]>) -> Result<Detector, LoadError> {
        let input = single(self.subgraph.inputs(), "subgraph inputs")?;
        let (hash_op, ngram) = self.ngram_hash(input)?;
        let hashed = single(hash_op.outputs(), "NGramHash outputs")?;

        let unpack = self.consumer(hashed, Kind::Builtin(fb::BuiltinOperator::UNPACK))?;
        let heads: Vec<i32> = ints(unpack.outputs(), "Unpack outputs")?;
        let options = unpack
            .builtin_options_as_unpack_options()
            .ok_or_else(|| LoadError::new("Unpack has no options"))?;
        if options.axis() != 1 || heads.len() != ngram.heads.len() {
            return Err(LoadError::new("Unpack does not split the NGramHash heads"));
        }

        let concat = self.find(Kind::Builtin(fb::BuiltinOperator::CONCATENATION))?;
        let options = concat
            .builtin_options_as_concatenation_options()
            .ok_or_else(|| LoadError::new("Concatenation has no options"))?;
        if !matches!(options.axis(), 1 | -1) || options.fused_activation_function() != fb::ActivationFunctionType::NONE
        {
            return Err(LoadError::new("unsupported Concatenation options"));
        }
        let mut embeddings = Vec::new();
        for part in ints(concat.inputs(), "Concatenation inputs")? {
            let op = self.producer(part, Kind::Custom("KmeansEmbeddingLookup"))?;
            let ids = ints(op.inputs(), "KmeansEmbeddingLookup inputs")?;
            let [ids, codes, codebook] = ids[..] else {
                return Err(LoadError::new("KmeansEmbeddingLookup needs three inputs"));
            };
            let head = heads
                .iter()
                .position(|&t| t == ids)
                .ok_or_else(|| LoadError::new("KmeansEmbeddingLookup does not read an NGramHash head"))?;
            let embedding = self.embedding(codes, codebook)?;
            let &(_, vocab) = ngram
                .heads
                .get(head)
                .ok_or_else(|| LoadError::new("Unpack does not split the NGramHash heads"))?;
            if u64::from(vocab) >= embedding.ids() as u64 {
                return Err(LoadError::new("embedding table is smaller than the hash vocabulary"));
            }
            embeddings.push((head, embedding));
        }

        let mut dense = Vec::new();
        let mut current = single(concat.outputs(), "Concatenation outputs")?;
        let mut width: usize = embeddings.iter().map(|(_, e)| e.width()).sum();
        let mut softmax_op = None;

        // Every step reads another operator, so a graph whose layers loop ends after all of them.
        for _ in 0..self.ops.len() {
            let next = self
                .ops
                .iter()
                .find(|(_, op)| op.inputs().is_some_and(|i| i.iter().next() == Some(current)))
                .ok_or_else(|| LoadError::new("the graph ends before Softmax"))?;

            match next.0 {
                Kind::Builtin(fb::BuiltinOperator::FULLY_CONNECTED) => {
                    let layer = self.dense(next.1, width)?;
                    width = layer.outputs();
                    dense.push(layer);
                    current = single(next.1.outputs(), "FullyConnected outputs")?;
                }
                Kind::Builtin(fb::BuiltinOperator::SOFTMAX) => {
                    softmax_op = Some(next.1);
                    break;
                }
                kind => return Err(LoadError::new(format!("unsupported operator {kind:?}"))),
            }
        }
        let softmax_op = softmax_op.ok_or_else(|| LoadError::new("the fully connected layers loop"))?;
        let beta = softmax_op
            .builtin_options_as_softmax_options()
            .ok_or_else(|| LoadError::new("Softmax has no options"))?
            .beta();
        if single(softmax_op.outputs(), "Softmax outputs")? != single(self.subgraph.outputs(), "subgraph outputs")? {
            return Err(LoadError::new("Softmax is not the model output"));
        }
        if self.ops.len() != 4 + embeddings.len() + dense.len() {
            return Err(LoadError::new("the graph has unsupported operators"));
        }
        if labels.len() != width {
            return Err(LoadError::new(format!(
                "{} labels for {width} model outputs",
                labels.len()
            )));
        }

        Ok(Detector {
            ngram,
            embeddings: embeddings.into_boxed_slice(),
            dense: dense.into_boxed_slice(),
            softmax_beta: beta,
            labels,
        })
    }

    fn ngram_hash(&self, input: i32) -> Result<(fb::Operator<'a>, NgramHash), LoadError> {
        let op = self.consumer(input, Kind::Custom("NGramHash"))?;
        let options = op
            .custom_options()
            .ok_or_else(|| LoadError::new("NGramHash has no options"))?;
        let root = flexbuffers::Reader::get_root(options.bytes())
            .map_err(|e| LoadError::with_source("invalid NGramHash options", e))?;
        let map = root
            .get_map()
            .map_err(|e| LoadError::with_source("invalid NGramHash options", e))?;
        let ints = |key: &str| -> Result<Vec<u64>, LoadError> {
            let vector = map
                .index(key)
                .and_then(|v| v.get_vector())
                .map_err(|e| LoadError::with_source(format!("invalid NGramHash {key}"), e))?;
            vector
                .iter()
                .map(|v| u64::try_from(v.as_i64()).map_err(|e| LoadError::with_source(format!("negative {key}"), e)))
                .collect()
        };
        let lengths = ints("ngram_lengths")?;
        let vocab_sizes = ints("vocab_sizes")?;
        if lengths.is_empty() || lengths.len() != vocab_sizes.len() {
            return Err(LoadError::new("NGramHash heads are inconsistent"));
        }
        // Bucket ids are stored as i32 in TFLite.
        if vocab_sizes.iter().any(|&v| v == 0 || v >= i32::MAX as u64) {
            return Err(LoadError::new("invalid NGramHash vocabulary size"));
        }
        let seed = map
            .index("seed")
            .map_err(|e| LoadError::with_source("NGramHash has no seed", e))?
            .as_u64();
        let max_splits = match map.index("max_splits") {
            Ok(v) if !v.flexbuffer_type().is_null() => {
                usize::try_from(v.as_i64()).map_err(|e| LoadError::with_source("invalid max_splits", e))?
            }
            _ => DEFAULT_MAX_SPLITS,
        };
        let lowercase = match map.index("lowercase_input") {
            Ok(v) if !v.flexbuffer_type().is_null() => v.as_bool(),
            _ => true,
        };
        if max_splits == 0 {
            return Err(LoadError::new("max_splits must be positive"));
        }
        let heads = lengths
            .into_iter()
            .zip(vocab_sizes)
            .map(|(n, vocab)| {
                let n = usize::try_from(n)
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or_else(|| LoadError::new("invalid n-gram length"))?;
                let vocab =
                    u32::try_from(vocab).map_err(|e| LoadError::with_source("invalid NGramHash vocabulary size", e))?;
                Ok((n, vocab))
            })
            .collect::<Result<_, _>>()?;
        Ok((
            op,
            NgramHash {
                seed,
                heads,
                max_splits,
                lowercase,
            },
        ))
    }

    fn embedding(&self, codes: i32, codebook: i32) -> Result<KmeansEmbedding, LoadError> {
        let (codes_shape, codes_data) = self.tensor(codes, fb::TensorType::UINT8)?;
        let (book_shape, book_data) = self.tensor(codebook, fb::TensorType::FLOAT32)?;
        let (&[ids, codes_per_id], &[centroids, block]) = (codes_shape.as_slice(), book_shape.as_slice()) else {
            return Err(LoadError::new("embedding tensors must be matrices"));
        };
        let codes: Box<[u8]> = codes_data.into();
        if codes.len() != ids * codes_per_id || codes_per_id == 0 || block == 0 {
            return Err(LoadError::new("embedding code table has the wrong size"));
        }
        if codes.iter().any(|&c| usize::from(c) >= centroids) {
            return Err(LoadError::new("embedding code out of the codebook"));
        }
        let codebook = floats(book_data, centroids * block)?;
        Ok(KmeansEmbedding {
            codes,
            codes_per_id,
            codebook,
            block,
        })
    }

    fn dense(&self, op: fb::Operator<'a>, inputs: usize) -> Result<HybridDense, LoadError> {
        let options = op
            .builtin_options_as_fully_connected_options()
            .ok_or_else(|| LoadError::new("FullyConnected has no options"))?;
        let relu = match options.fused_activation_function() {
            fb::ActivationFunctionType::NONE => false,
            fb::ActivationFunctionType::RELU => true,
            other => return Err(LoadError::new(format!("unsupported activation {other:?}"))),
        };
        if !options.asymmetric_quantize_inputs()
            || options.weights_format() != fb::FullyConnectedOptionsWeightsFormat::DEFAULT
        {
            return Err(LoadError::new(
                "only asymmetric hybrid FullyConnected layers are supported",
            ));
        }
        let [_, weights, bias] = ints(op.inputs(), "FullyConnected inputs")?[..] else {
            return Err(LoadError::new("FullyConnected needs weights and bias"));
        };
        let (shape, data) = self.tensor(weights, fb::TensorType::INT8)?;
        let [outputs, width] = shape[..] else {
            return Err(LoadError::new("FullyConnected weights must be a matrix"));
        };
        if width != inputs || outputs.checked_mul(inputs) != Some(data.len()) {
            return Err(LoadError::new("FullyConnected weights do not match the input"));
        }
        if inputs > MAX_DENSE_INPUTS {
            return Err(LoadError::new(format!(
                "FullyConnected has {inputs} inputs, more than the supported {MAX_DENSE_INPUTS}"
            )));
        }
        let quantization = self
            .tensors
            .get(usize::try_from(weights).map_err(|e| LoadError::with_source("invalid tensor index", e))?)
            .quantization()
            .ok_or_else(|| LoadError::new("FullyConnected weights are not quantized"))?;
        let scales = quantization
            .scale()
            .ok_or_else(|| LoadError::new("FullyConnected weights have no scale"))?;
        let zero_point = quantization.zero_point().map_or(0, |z| z.iter().next().unwrap_or(0));
        if scales.len() != 1 || zero_point != 0 {
            return Err(LoadError::new(
                "only per-tensor symmetric weight quantization is supported",
            ));
        }
        let (bias_shape, bias_data) = self.tensor(bias, fb::TensorType::FLOAT32)?;
        if bias_shape[..] != [outputs] {
            return Err(LoadError::new("FullyConnected bias does not match the weights"));
        }
        let weights = data.iter().map(|&b| b.cast_signed()).collect();
        Ok(HybridDense::new(
            weights,
            inputs,
            scales.get(0),
            floats(bias_data, outputs)?,
            relu,
        ))
    }

    /// Shape and constant data of tensor `index`, which must have type `expected`.
    fn tensor(&self, index: i32, expected: fb::TensorType) -> Result<(Vec<usize>, &'a [u8]), LoadError> {
        let index = usize::try_from(index).map_err(|e| LoadError::with_source("invalid tensor index", e))?;
        if index >= self.tensors.len() {
            return Err(LoadError::new("tensor index out of range"));
        }
        let tensor = self.tensors.get(index);
        if tensor.type_() != expected {
            return Err(LoadError::new(format!(
                "tensor {:?} has type {:?}, expected {expected:?}",
                tensor.name(),
                tensor.type_()
            )));
        }
        let shape = tensor
            .shape()
            .map(|s| s.iter().map(usize::try_from).collect::<Result<Vec<_>, _>>())
            .transpose()
            .map_err(|e| LoadError::with_source("negative tensor dimension", e))?
            .unwrap_or_default();
        let buffers = self.model.buffers().ok_or_else(|| LoadError::new("no buffers"))?;
        let buffer_index = tensor.buffer() as usize;
        if buffer_index >= buffers.len() {
            return Err(LoadError::new("buffer index out of range"));
        }
        let buffer = buffers.get(buffer_index);
        let data = match buffer.data() {
            Some(data) => data.bytes(),
            // Large models keep buffers after the flatbuffer, addressed by offset and size.
            None => usize::try_from(buffer.offset())
                .ok()
                .zip(usize::try_from(buffer.size()).ok())
                .and_then(|(offset, size)| self.file.get(offset..offset.checked_add(size)?))
                .ok_or_else(|| LoadError::new("tensor data out of range"))?,
        };
        Ok((shape, data))
    }

    fn find(&self, kind: Kind<'_>) -> Result<fb::Operator<'a>, LoadError> {
        let mut matching = self.ops.iter().filter(|(k, _)| *k == kind);
        match (matching.next(), matching.next()) {
            (Some((_, op)), None) => Ok(*op),
            _ => Err(LoadError::new(format!("expected exactly one {kind:?} operator"))),
        }
    }

    /// The operator of `kind` that reads `tensor` as its first input.
    fn consumer(&self, tensor: i32, kind: Kind<'_>) -> Result<fb::Operator<'a>, LoadError> {
        self.ops
            .iter()
            .find(|(k, op)| *k == kind && op.inputs().is_some_and(|i| i.iter().next() == Some(tensor)))
            .map(|(_, op)| *op)
            .ok_or_else(|| LoadError::new(format!("no {kind:?} operator reads the expected tensor")))
    }

    /// The operator of `kind` that writes `tensor`.
    fn producer(&self, tensor: i32, kind: Kind<'_>) -> Result<fb::Operator<'a>, LoadError> {
        let producers: HashMap<i32, usize> = self
            .ops
            .iter()
            .enumerate()
            .flat_map(|(i, (_, op))| op.outputs().into_iter().flatten().map(move |t| (t, i)))
            .collect();
        producers
            .get(&tensor)
            .and_then(|&i| self.ops.get(i))
            .filter(|(k, _)| *k == kind)
            .map(|(_, op)| *op)
            .ok_or_else(|| LoadError::new(format!("no {kind:?} operator writes the expected tensor")))
    }
}

fn ints(vector: Option<flatbuffers::Vector<'_, i32>>, what: &'static str) -> Result<Vec<i32>, LoadError> {
    vector
        .map(|v| v.iter().collect())
        .ok_or_else(|| LoadError::new(format!("missing {what}")))
}

fn single(vector: Option<flatbuffers::Vector<'_, i32>>, what: &'static str) -> Result<i32, LoadError> {
    match ints(vector, what)?[..] {
        [one] => Ok(one),
        _ => Err(LoadError::new(format!("expected exactly one of {what}"))),
    }
}

fn floats(data: &[u8], len: usize) -> Result<Box<[f32]>, LoadError> {
    if data.len() != len * 4 {
        return Err(LoadError::new("float tensor has the wrong size"));
    }
    Ok(data.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect())
}
