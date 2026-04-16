//! Prompt builder — constructs LLM prompts from templates and schema.
//!
//! Templates are Markdown files with `{{variable}}` placeholders.
//! Loaded from the filesystem at runtime — no prompts in compiled code.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::PipelineError;
use crate::schema::ExtractionSchema;
use crate::types::ExtractionResult;

/// The complete output of prompt assembly, including the rendered text
/// and cryptographic hashes of every input file. Stored alongside
/// extraction runs for full reproducibility.
///
/// ## Rust Learning: Returning rich results
///
/// Instead of returning just a String, we return a struct that bundles
/// the result with its metadata. The caller gets everything needed for
/// audit and reproducibility in one value, without needing to track
/// file hashes separately.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PromptArtifact {
    /// The fully assembled prompt text sent to the LLM.
    pub prompt_text: String,
    /// Template filename that was used.
    pub template_name: String,
    /// SHA-256 hex digest of the template file content.
    pub template_hash: String,
    /// Rules filename that was used (None if no rules file).
    pub rules_name: Option<String>,
    /// SHA-256 hex digest of the rules file content (None if no rules file).
    pub rules_hash: Option<String>,
    /// SHA-256 hex digest of the schema JSON that was embedded in the prompt.
    pub schema_hash: String,
}

/// Builds LLM prompts from templates and extraction schemas.
pub struct PromptBuilder {
    template_dir: PathBuf,
    cache: HashMap<String, String>,
}

impl PromptBuilder {
    /// Create a new PromptBuilder that loads templates from the given directory.
    pub fn new(template_dir: &Path) -> Self {
        Self {
            template_dir: template_dir.to_path_buf(),
            cache: HashMap::new(),
        }
    }

    /// Load a template file and cache it.
    fn load_template(&mut self, name: &str) -> Result<&str, PipelineError> {
        if !self.cache.contains_key(name) {
            let path = self.template_dir.join(name);
            let content = std::fs::read_to_string(&path).map_err(|e| {
                PipelineError::Template(format!(
                    "Failed to load template '{}': {}",
                    path.display(),
                    e
                ))
            })?;
            self.cache.insert(name.to_string(), content);
        }
        Ok(self.cache.get(name).expect("just inserted"))
    }

    /// Compute SHA-256 hex digest of a string.
    fn sha256_hex(content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Substitute `{{variable}}` placeholders in a template string.
    fn substitute(template: &str, vars: &HashMap<&str, String>) -> String {
        let mut result = template.to_string();
        for (key, value) in vars {
            let placeholder = format!("{{{{{}}}}}", key);
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Build a Pass 1 extraction prompt.
    ///
    /// `template_name` — optional template filename. Defaults to
    /// `"pass1_template.md"` if `None`.
    ///
    /// Returns a [`PromptArtifact`] containing the rendered prompt text
    /// plus SHA-256 hashes of every input file for reproducibility.
    pub fn build_extraction_prompt(
        &mut self,
        schema: &ExtractionSchema,
        document_text: &str,
        context: Option<&str>,
        admin_instructions: Option<&str>,
        rules_file: Option<&str>,
        template_name: Option<&str>,
    ) -> Result<PromptArtifact, PipelineError> {
        let actual_template_name = template_name.unwrap_or("pass1_template.md");
        let template_content = self.load_template(actual_template_name)?.to_string();
        let template_hash = Self::sha256_hex(&template_content);

        let schema_json = schema.to_prompt_json()?;
        let schema_hash = Self::sha256_hex(&schema_json);

        let (rules_content, rules_name, rules_hash) = match rules_file {
            Some(name) => {
                let content = self.load_template(name)?.to_string();
                let hash = Self::sha256_hex(&content);
                (content, Some(name.to_string()), Some(hash))
            }
            None => (String::new(), None, None),
        };

        let mut vars = HashMap::new();
        vars.insert("schema_json", schema_json);
        vars.insert("global_rules", rules_content);
        vars.insert("document_text", document_text.to_string());
        vars.insert(
            "context",
            context
                .unwrap_or("None — this is the first document.")
                .to_string(),
        );
        vars.insert(
            "admin_instructions",
            admin_instructions.unwrap_or("None.").to_string(),
        );

        let prompt_text = Self::substitute(&template_content, &vars);

        Ok(PromptArtifact {
            prompt_text,
            template_name: actual_template_name.to_string(),
            template_hash,
            rules_name,
            rules_hash,
            schema_hash,
        })
    }

    /// Build a Pass 2 synthesis prompt.
    pub fn build_synthesis_prompt(
        &mut self,
        schema: &ExtractionSchema,
        pass1_result: &ExtractionResult,
    ) -> Result<String, PipelineError> {
        let template = self.load_template("pass2_template.md")?.to_string();

        let schema_json = schema.to_prompt_json()?;
        let pass1_json = serde_json::to_string_pretty(pass1_result)?;

        let mut vars = HashMap::new();
        vars.insert("schema_json", schema_json);
        vars.insert("pass1_output", pass1_json);

        Ok(Self::substitute(&template, &vars))
    }
}
