# Changelog

## v0.15.0 — 2026-05-11

### Breaking changes

- **colossus-extract:** `EntityTypeConfig.required` and `EntityTypeConfig.min_count`
  no longer default when absent from YAML. Schemas must declare both fields
  explicitly for every entity type. Loading a schema YAML without these fields
  produces a serde missing-field error.

### Crate versions

- `colossus-extract` 0.5.0 → 0.6.0 (breaking, pre-1.0 semver)
- Other workspace crates unchanged. Repo tag `v0.15.0` advances as the
  coordination marker; crate versions are independent of the repo tag.

### Why

Per silent-fallback audit defect #4.1.1: schemas that intended to demand at
least one Party (or any other completeness rule) silently accepted zero
because `#[serde(default)]` made `required` and `min_count` permissive on
absence. The defaults defeated the completeness validation they were
supposed to enable.

### Migration

For every schema YAML loaded by colossus-extract:

```yaml
entity_types:
  - name: Party
    required: true       # add explicit value
    min_count: 2         # add explicit value
    # ...other fields
```

Failure to add these fields produces:

```
missing field `required` at line X column Y
```

at schema load time. The error names the missing field for diagnosis.
