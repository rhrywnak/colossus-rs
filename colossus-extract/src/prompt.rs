//! Prompt builder — constructs LLM prompts from templates and schema.
//!
//! Templates are Markdown files with `{{variable}}` placeholders.
//! Loaded from the filesystem at runtime — no prompts in compiled code.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::error::PipelineError;
use crate::schema::ExtractionSchema;
use crate::types::ExtractionResult;

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
            let content = std::fs::read_to_string(&path)
                .map_err(|e| PipelineError::Template(
                    format!("Failed to load template '{}': {}", path.display(), e)
                ))?;
            self.cache.insert(name.to_string(), content);
        }
        Ok(self.cache.get(name).expect("just inserted"))
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
    pub fn build_extraction_prompt(
        &mut self,
        schema: &ExtractionSchema,
        document_text: &str,
        context: Option<&str>,
        admin_instructions: Option<&str>,
        rules_file: Option<&str>,
        template_name: Option<&str>,
    ) -> Result<String, PipelineError> {
        let template = self
            .load_template(template_name.unwrap_or("pass1_template.md"))?
            .to_string();

        let schema_json = schema.to_prompt_json()?;

        let rules = match rules_file {
            Some(name) => self.load_template(name)?.to_string(),
            None => String::new(),
        };

        let mut vars = HashMap::new();
        vars.insert("schema_json", schema_json);
        vars.insert("global_rules", rules);
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

        Ok(Self::substitute(&template, &vars))
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
