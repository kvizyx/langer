//! Validates `data/languages.json` and compiles it into the minified body of
//! `GET /detect/languages`, sorted by code.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::{env, fs};

use anyhow::{Context as _, ensure};

const SOURCE: &str = "data/languages.json";

/// Values of `Method` in `src/detect.rs`.
const METHODS: [&str; 2] = ["mediapipe", "lingua"];

/// One language, in the field order of the response.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct Language {
    iso_639_1: String,
    name: String,
    methods: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    println!("cargo::rerun-if-changed={SOURCE}");

    let source = fs::read_to_string(SOURCE).with_context(|| format!("cannot read {SOURCE}"))?;
    let languages: Vec<Language> =
        serde_json::from_str(&source).with_context(|| format!("{SOURCE} is not a JSON array of languages"))?;

    let mut by_code = BTreeMap::new();
    for language in &languages {
        ensure!(
            !language.iso_639_1.is_empty() && !language.name.is_empty(),
            "{SOURCE}: empty code or name"
        );

        ensure!(
            !language.methods.is_empty(),
            "{SOURCE}: {} has no methods",
            language.iso_639_1
        );

        for method in &language.methods {
            ensure!(
                METHODS.contains(&method.as_str()),
                "{SOURCE}: {} has unknown method {method:?}",
                language.iso_639_1
            );
        }

        let duplicate = by_code.insert(language.iso_639_1.as_str(), language);
        ensure!(duplicate.is_none(), "{SOURCE}: duplicate code {}", language.iso_639_1);
    }

    let out = PathBuf::from(env::var_os("OUT_DIR").context("cargo did not set OUT_DIR")?);
    let sorted: Vec<&Language> = by_code.values().copied().collect();
    let json = serde_json::to_string(&sorted)?;
    fs::write(out.join("languages.json"), json).context("cannot write to OUT_DIR")?;

    Ok(())
}
