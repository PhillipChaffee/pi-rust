//! Model-search text at upstream pin `60e7e76bd7ea25cad1dd6f3f1ce0d18814a42759`.

/// Repeated provider/id spellings preserve upstream's fuzzy-match ranking.
#[must_use]
pub fn get_model_search_text(id: &str, provider: &str, name: Option<&str>) -> String {
    let name = name
        .filter(|name| !name.is_empty())
        .map_or_else(String::new, |name| format!(" {name}"));
    format!("{id} {provider} {provider}/{id} {provider} {id}{name}")
}
