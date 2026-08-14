//! Julia language parser.
//!
//! Julia's structure differs from the class-oriented languages in one way that
//! shapes this whole module: **methods do not live inside types**. Functions are
//! free and dispatch on argument types, so there is no `HAS_METHOD` nesting to
//! recover — functions belong to their module, and the type/function relationship
//! is expressed through dispatch, which is not statically resolvable here.
//!
//! Coverage:
//!   - `module` / `baremodule` → nested naming scope for everything inside.
//!   - `struct` / `mutable struct` → ClassInfo (kind `struct`), with `<:`
//!     supertypes recorded as bases and an `extends` TypeRelationship.
//!   - `abstract type` / `primitive type` → ClassInfo (kinds `abstract` /
//!     `primitive`), likewise carrying their supertype.
//!   - `function f(...) ... end`, short-form `f(x) = ...`, and `macro m(...)`
//!     → FunctionInfo, including definitions nested inside other functions.
//!   - `const` bindings → ConstantInfo.
//!   - `include("path.jl")` → FileInfo.imports, resolved to a module path.
//!   - `using` / `import` → FileInfo.imports as module names.
//!   - Triple-quoted docstrings preceding a definition.
//!
//! ## Why `include` is the important edge
//!
//! Julia has no file-grain import statement. A package is assembled by a root
//! file that `include`s its parts, and `using`/`import` name *modules*, which do
//! not correspond to files one-to-one (one file may define several, and a module
//! may span many files). So `include` is the only construct that states a
//! file-to-file dependency, and it is resolved here against the including file's
//! own directory — the same treatment JS/TS relative specifiers get.
//!
//! Known gaps: `using .Local` / `..Parent` are module-relative rather than
//! path-relative, so they are emitted as bare module names without file
//! resolution (mapping them would need a module-declaration index). `include`
//! wrapped in arbitrary computation is recovered only when a literal path string
//! is present (`include(joinpath(@__DIR__, "x.jl"))` works; a fully computed path
//! does not).

use std::path::Path;
use tree_sitter::{Node, Parser, Tree};

use super::shared::node_text;
use super::LanguageParser;
use crate::code_tree::models::{
    ClassInfo, ConstantInfo, FileInfo, FunctionInfo, ParseResult, TypeRelationship,
};

/// Base-library names that would otherwise dominate call-edge resolution.
pub const JULIA_NOISE_NAMES: &[&str] = &[
    "println",
    "print",
    "error",
    "throw",
    "length",
    "size",
    "push!",
    "pop!",
    "append!",
    "get",
    "get!",
    "haskey",
    "keys",
    "values",
    "collect",
    "map",
    "filter",
    "reduce",
    "sum",
    "prod",
    "minimum",
    "maximum",
    "min",
    "max",
    "abs",
    "sqrt",
    "zeros",
    "ones",
    "similar",
    "copy",
    "deepcopy",
    "convert",
    "string",
    "join",
    "split",
    "isempty",
    "in",
    "typeof",
    "isa",
    "include",
    "sort",
    "sort!",
    "reverse",
    "enumerate",
    "zip",
    "range",
    "identity",
];

pub struct JuliaParser;

thread_local! {
    static TS_PARSER: std::cell::RefCell<Parser> = {
        let mut p = Parser::new();
        p.set_language(&tree_sitter_julia::LANGUAGE.into())
            .expect("loading tree-sitter-julia grammar");
        std::cell::RefCell::new(p)
    };
}

impl JuliaParser {
    pub fn new() -> Self {
        JuliaParser
    }

    fn parse_tree(&self, source: &[u8]) -> Option<Tree> {
        TS_PARSER.with(|p| p.borrow_mut().parse(source, None))
    }

    /// Dotted module path for a `.jl` file, rooted like the other dotted
    /// languages so the shared resolver's root-prefix recovery applies.
    fn file_to_module_path(filepath: &Path, src_root: &Path) -> String {
        let rel = filepath.strip_prefix(src_root).unwrap_or(filepath);
        let mut parts: Vec<String> = rel
            .components()
            .filter_map(|c| c.as_os_str().to_str().map(str::to_string))
            .collect();
        if let Some(last) = parts.last_mut() {
            if let Some(stem) = last.strip_suffix(".jl") {
                *last = stem.to_string();
            }
        }
        let pkg = src_root.file_name().and_then(|o| o.to_str()).unwrap_or("");
        if parts.is_empty() {
            pkg.to_string()
        } else if pkg.is_empty() || parts.first().map(String::as_str) == Some(pkg) {
            parts.join(".")
        } else {
            format!("{}.{}", pkg, parts.join("."))
        }
    }

    /// Resolve an `include` path against the including file's own module path.
    ///
    /// `include("sub/x.jl")` from `pkg.src.Root` yields `pkg.src.sub.x`, matching
    /// what `file_to_module_path` produces for the included file.
    fn resolve_include(spec: &str, module_path: &str) -> Option<String> {
        if spec.is_empty() || module_path.is_empty() {
            return None;
        }
        let mut parts: Vec<&str> = module_path.split('.').collect();
        parts.pop(); // the including file itself -> its directory

        for segment in spec.split('/') {
            match segment {
                "" | "." => {}
                ".." => {
                    parts.pop()?;
                }
                other => parts.push(other),
            }
        }
        let last = parts.pop()?;
        parts.push(last.strip_suffix(".jl").unwrap_or(last));
        if parts.iter().any(|p| p.is_empty()) || parts.is_empty() {
            return None;
        }
        Some(parts.join("."))
    }

    /// The literal path argument of an `include(...)` call, if there is one.
    ///
    /// Takes the last string literal in the argument list so the common
    /// `include(joinpath(@__DIR__, "x.jl"))` wrapping still yields `x.jl`.
    fn include_path(node: Node, source: &[u8]) -> Option<String> {
        let callee = node.named_child(0)?;
        if node_text(callee, source) != "include" {
            return None;
        }
        let mut cursor = node.walk();
        let args = node
            .child_by_field_name("arguments")
            .or_else(|| node.children(&mut cursor).find(|c| c.kind() == "argument_list"))?;
        let mut found: Option<String> = None;
        Self::visit_string_literals(args, source, &mut found);
        found
    }

    /// Record the last string literal found anywhere under `node`.
    fn visit_string_literals(node: Node, source: &[u8], out: &mut Option<String>) {
        if node.kind() == "string_literal" {
            let mut cursor = node.walk();
            let text: String = node
                .children(&mut cursor)
                .filter(|c| c.kind() == "content")
                .map(|c| node_text(c, source))
                .collect();
            if !text.is_empty() {
                *out = Some(text);
            }
            return;
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            Self::visit_string_literals(child, source, out);
        }
    }

    /// The module name(s) named by a `using` / `import` statement.
    ///
    /// Julia allows `using A`, `using A.B`, `using A: x, y`, and the
    /// module-relative `using .A` / `using ..A`. Leading dots are stripped: they
    /// address the module tree rather than the filesystem, so the bare name is
    /// the most that can be said without a module-declaration index.
    fn import_targets(node: Node, source: &[u8], out: &mut Vec<String>) {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "identifier" | "import_path" | "field_expression" | "scoped_identifier" => {
                    let text = node_text(child, source).trim_start_matches('.');
                    if !text.is_empty() {
                        out.push(text.to_string());
                    }
                }
                // `A: x, y` — only the module before the colon is a dependency.
                "selected_import" => {
                    if let Some(first) = child.named_child(0) {
                        let text = node_text(first, source).trim_start_matches('.');
                        if !text.is_empty() {
                            out.push(text.to_string());
                        }
                    }
                }
                _ => {}
            }
        }
    }

    /// Docstring attached to a definition: Julia places it in a string literal
    /// immediately preceding the definition.
    fn preceding_docstring(node: Node, source: &[u8]) -> Option<String> {
        let mut sibling = node.prev_named_sibling();
        while let Some(s) = sibling {
            match s.kind() {
                "line_comment" | "block_comment" => sibling = s.prev_named_sibling(),
                "string_literal" => {
                    let mut cursor = s.walk();
                    let text: String = s
                        .children(&mut cursor)
                        .filter(|c| c.kind() == "content")
                        .map(|c| node_text(c, source))
                        .collect();
                    let trimmed = text.trim();
                    return (!trimmed.is_empty()).then(|| trimmed.to_string());
                }
                _ => return None,
            }
        }
        None
    }

    /// Name and supertype from a `type_head`, which is either a bare identifier
    /// or a `X <: Y` binary expression.
    fn split_type_head(node: Node, source: &[u8]) -> Option<(String, Option<String>)> {
        let mut cursor = node.walk();
        let head = node
            .child_by_field_name("type_head")
            .or_else(|| node.children(&mut cursor).find(|c| c.kind() == "type_head"))?;
        let inner = head.named_child(0).unwrap_or(head);
        if inner.kind() == "binary_expression" {
            let name = inner.named_child(0)?;
            let supertype = inner.named_child(2).or_else(|| inner.named_child(1));
            return Some((
                node_text(name, source).to_string(),
                supertype
                    .map(|s| node_text(s, source).to_string())
                    .filter(|s| s != "<:"),
            ));
        }
        Some((node_text(inner, source).to_string(), None))
    }

    /// The declared name of a `function` / `macro` definition.
    ///
    /// The signature is a `call_expression`, optionally wrapped in a
    /// `typed_expression` when a return type is declared, and may be a
    /// `field_expression` for a qualified definition like `Base.show`.
    fn signature_name(node: Node, source: &[u8]) -> Option<(String, Option<String>)> {
        let mut cursor = node.walk();
        let signature = node
            .children(&mut cursor)
            .find(|c| c.kind() == "signature")?;
        let mut inner = signature.named_child(0)?;
        let mut return_type = None;
        if inner.kind() == "typed_expression" {
            return_type = inner.named_child(1).map(|n| node_text(n, source).to_string());
            inner = inner.named_child(0)?;
        }
        // `where` clauses wrap the call in another binary/where expression.
        while !matches!(inner.kind(), "call_expression" | "identifier") {
            inner = inner.named_child(0)?;
        }
        let name_node = if inner.kind() == "call_expression" {
            inner.named_child(0)?
        } else {
            inner
        };
        Some((node_text(name_node, source).to_string(), return_type))
    }

    /// Julia exports nothing by default; `export a, b` inside a module lists the
    /// public surface.
    fn export_names(node: Node, source: &[u8], out: &mut Vec<String>) {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if child.kind() == "identifier" {
                out.push(node_text(child, source).to_string());
            }
        }
    }

    fn visibility(name: &str, exports: &[String]) -> &'static str {
        if exports.iter().any(|e| e == name) {
            "public"
        } else {
            "private"
        }
    }

    fn extract_calls(node: Node, source: &[u8]) -> Vec<(String, u32)> {
        fn walk(n: Node, source: &[u8], out: &mut Vec<(String, u32)>) {
            if n.kind() == "call_expression" {
                if let Some(callee) = n.named_child(0) {
                    let text = node_text(callee, source);
                    // `A.b(x)` dispatches on `b`; keep the terminal segment.
                    let bare = text.rsplit('.').next().unwrap_or(text).trim();
                    if !bare.is_empty()
                        && !bare.contains(' ')
                        && !bare.contains('(')
                        && !bare.starts_with('@')
                    {
                        out.push((bare.to_string(), n.start_position().row as u32 + 1));
                    }
                }
            }
            let mut cursor = n.walk();
            for child in n.named_children(&mut cursor) {
                // A nested definition owns its own calls.
                if !matches!(child.kind(), "function_definition" | "macro_definition") {
                    walk(child, source, out);
                }
            }
        }
        let mut calls = Vec::new();
        walk(node, source, &mut calls);
        calls
    }

    fn build_signature(node: Node, source: &[u8]) -> String {
        let mut cursor = node.walk();
        node.children(&mut cursor)
            .take_while(|c| c.kind() != "end")
            .filter(|c| matches!(c.kind(), "function" | "macro" | "signature"))
            .map(|c| node_text(c, source))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[allow(clippy::too_many_arguments)]
    fn push_function(
        node: Node,
        source: &[u8],
        scope: &str,
        rel_path: &str,
        exports: &[String],
        is_macro: bool,
        result: &mut ParseResult,
    ) {
        let Some((name, return_type)) = Self::signature_name(node, source) else {
            return;
        };
        // A qualified definition (`function Base.show(...)`) belongs to the
        // foreign module, but the bare name is what call resolution looks for.
        let bare = name.rsplit('.').next().unwrap_or(&name).to_string();
        let display = if is_macro {
            format!("@{bare}")
        } else {
            bare.clone()
        };
        let qualified_name = if scope.is_empty() {
            display.clone()
        } else {
            format!("{scope}.{display}")
        };

        result.functions.push(FunctionInfo {
            qualified_name,
            visibility: Self::visibility(&bare, exports).to_string(),
            is_async: false,
            is_method: false,
            signature: Self::build_signature(node, source),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            name: display,
            docstring: Self::preceding_docstring(node, source),
            return_type,
            decorators: Vec::new(),
            calls: Self::extract_calls(node, source),
            references: Vec::new(),
            function_refs: Vec::new(),
            type_parameters: None,
            end_line: Some(node.end_position().row as u32 + 1),
            parameters: Vec::new(),
            branch_count: None,
            param_count: None,
            max_nesting: None,
            is_recursive: None,
            procedure_names: Vec::new(),
            metadata: Default::default(),
        });
    }

    fn push_type(
        node: Node,
        source: &[u8],
        scope: &str,
        rel_path: &str,
        exports: &[String],
        kind: &str,
        result: &mut ParseResult,
    ) {
        let Some((name, supertype)) = Self::split_type_head(node, source) else {
            return;
        };
        let bare_name = name.clone();
        let qualified_name = if scope.is_empty() {
            name.clone()
        } else {
            format!("{scope}.{name}")
        };

        result.classes.push(ClassInfo {
            qualified_name,
            visibility: Self::visibility(&name, exports).to_string(),
            name,
            kind: kind.to_string(),
            file_path: rel_path.to_string(),
            line_number: node.start_position().row as u32 + 1,
            docstring: Self::preceding_docstring(node, source),
            bases: supertype.clone().into_iter().collect(),
            type_parameters: None,
            end_line: Some(node.end_position().row as u32 + 1),
            metadata: Default::default(),
        });

        // `<:` is Julia's subtyping declaration — the closest analogue to
        // inheritance, and the only static type relationship available. Both
        // ends are bare names, matching the other parsers: a qualified source
        // would not match the node just emitted and would synthesize a duplicate.
        if let Some(supertype) = supertype {
            result.type_relationships.push(TypeRelationship {
                source_type: bare_name,
                target_type: Some(supertype),
                relationship: "extends".to_string(),
                methods: Vec::new(),
            });
        }
    }

    fn push_constants(
        node: Node,
        source: &[u8],
        scope: &str,
        rel_path: &str,
        exports: &[String],
        result: &mut ParseResult,
    ) {
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if child.kind() != "assignment" {
                continue;
            }
            let Some(target) = child.named_child(0) else {
                continue;
            };
            // Skip `const f(x) = ...`, which is a function, not a binding.
            if target.kind() != "identifier" {
                continue;
            }
            let name = node_text(target, source).to_string();
            let value = child.named_child(2).or_else(|| child.named_child(1));
            let value_preview = value.map(|v| {
                let text = node_text(v, source);
                let take = text
                    .char_indices()
                    .nth(100)
                    .map(|(i, _)| i)
                    .unwrap_or(text.len());
                text[..take].to_string()
            });
            let qualified_name = if scope.is_empty() {
                name.clone()
            } else {
                format!("{scope}.{name}")
            };
            result.constants.push(ConstantInfo {
                qualified_name,
                visibility: Self::visibility(&name, exports).to_string(),
                name,
                kind: "constant".to_string(),
                type_annotation: None,
                value_preview,
                file_path: rel_path.to_string(),
                line_number: child.start_position().row as u32 + 1,
            });
        }
    }

    /// Walk a block, dispatching each declaration.
    ///
    /// `scope` is the dotted prefix for qualified names — the file's module path
    /// at the top level, extended by each `module` entered. The walk descends
    /// into function bodies as well, since Julia definitions nest freely and
    /// `include` is routinely called from inside a conditional or a function.
    fn walk_block(
        node: Node,
        source: &[u8],
        scope: &str,
        module_path: &str,
        rel_path: &str,
        result: &mut ParseResult,
        file_info: &mut FileInfo,
    ) {
        // `export` is collected first so visibility is known for every
        // declaration in the block, whatever order they appear in.
        let mut exports: Vec<String> = Vec::new();
        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            if child.kind() == "export_statement" {
                Self::export_names(child, source, &mut exports);
            }
        }
        file_info.exports.extend(exports.iter().cloned());

        let mut cursor = node.walk();
        for child in node.named_children(&mut cursor) {
            match child.kind() {
                "module_definition" | "baremodule_definition" => {
                    let name = child
                        .child_by_field_name("name")
                        .map(|n| node_text(n, source).to_string());
                    // Julia's convention is `src/Foo.jl` declaring `module Foo`, so
                    // the module the path already names must not extend the scope
                    // (that would yield `…src.Foo.Foo`) nor be recorded as a
                    // submodule of itself. Genuinely nested modules do both.
                    let names_this_file = name.as_deref() == scope.rsplit('.').next();
                    let inner_scope = match &name {
                        _ if names_this_file => scope.to_string(),
                        Some(n) if scope.is_empty() => n.clone(),
                        Some(n) => format!("{scope}.{n}"),
                        None => scope.to_string(),
                    };
                    if let Some(n) = name {
                        if !names_this_file {
                            // `build_contains_edges` prefixes with the file's
                            // module path, so the declaration must be recorded
                            // relative to it, not to the enclosing module.
                            let relative = inner_scope
                                .strip_prefix(module_path)
                                .and_then(|r| r.strip_prefix('.'))
                                .map(str::to_string)
                                .unwrap_or(n);
                            file_info.submodule_declarations.push(relative);
                        }
                    }
                    Self::walk_block(
                        child,
                        source,
                        &inner_scope,
                        module_path,
                        rel_path,
                        result,
                        file_info,
                    );
                }
                "using_statement" | "import_statement" => {
                    Self::import_targets(child, source, &mut file_info.imports);
                }
                "struct_definition" => {
                    Self::push_type(
                        child, source, scope, rel_path, &exports, "struct", result,
                    );
                }
                "abstract_definition" => {
                    Self::push_type(
                        child, source, scope, rel_path, &exports, "abstract", result,
                    );
                }
                "primitive_definition" => {
                    Self::push_type(
                        child, source, scope, rel_path, &exports, "primitive", result,
                    );
                }
                "function_definition" => {
                    Self::push_function(child, source, scope, rel_path, &exports, false, result);
                    Self::walk_block(
                        child, source, scope, module_path, rel_path, result, file_info,
                    );
                }
                "macro_definition" => {
                    Self::push_function(child, source, scope, rel_path, &exports, true, result);
                }
                "const_statement" => {
                    Self::push_constants(child, source, scope, rel_path, &exports, result);
                }
                // Short-form definition: `f(x) = expr`.
                "assignment" => {
                    if let Some(target) = child.named_child(0) {
                        if target.kind() == "call_expression" {
                            Self::push_short_function(
                                child, target, source, scope, rel_path, &exports, result,
                            );
                        }
                    }
                }
                "call_expression" => {
                    if let Some(spec) = Self::include_path(child, source) {
                        if let Some(resolved) = Self::resolve_include(&spec, module_path) {
                            file_info.imports.push(resolved);
                        }
                    }
                }
                // Everything else may still contain an `include` or a nested
                // definition — a conditional include is idiomatic.
                _ => {
                    Self::walk_block(
                        child, source, scope, module_path, rel_path, result, file_info,
                    );
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn push_short_function(
        assignment: Node,
        call: Node,
        source: &[u8],
        scope: &str,
        rel_path: &str,
        exports: &[String],
        result: &mut ParseResult,
    ) {
        let Some(name_node) = call.named_child(0) else {
            return;
        };
        let name = node_text(name_node, source);
        let bare = name.rsplit('.').next().unwrap_or(name).to_string();
        if bare.is_empty() {
            return;
        }
        let qualified_name = if scope.is_empty() {
            bare.clone()
        } else {
            format!("{scope}.{bare}")
        };
        result.functions.push(FunctionInfo {
            qualified_name,
            visibility: Self::visibility(&bare, exports).to_string(),
            is_async: false,
            is_method: false,
            signature: node_text(call, source).to_string(),
            file_path: rel_path.to_string(),
            line_number: assignment.start_position().row as u32 + 1,
            name: bare,
            docstring: Self::preceding_docstring(assignment, source),
            return_type: None,
            decorators: Vec::new(),
            calls: Self::extract_calls(assignment, source),
            references: Vec::new(),
            function_refs: Vec::new(),
            type_parameters: None,
            end_line: Some(assignment.end_position().row as u32 + 1),
            parameters: Vec::new(),
            branch_count: None,
            param_count: None,
            max_nesting: None,
            is_recursive: None,
            procedure_names: Vec::new(),
            metadata: Default::default(),
        });
    }
}

impl Default for JuliaParser {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageParser for JuliaParser {
    fn language_name(&self) -> &'static str {
        "julia"
    }

    fn file_extensions(&self) -> &'static [&'static str] {
        &["jl"]
    }

    fn noise_names(&self) -> &'static [&'static str] {
        JULIA_NOISE_NAMES
    }

    fn parse_file(&self, filepath: &Path, src_root: &Path) -> ParseResult {
        let mut result = ParseResult::new();
        let Ok(source) = std::fs::read_to_string(filepath) else {
            return result;
        };
        let source_bytes = source.as_bytes();
        let rel_path = filepath
            .strip_prefix(src_root)
            .unwrap_or(filepath)
            .to_string_lossy()
            .to_string();
        let module_path = Self::file_to_module_path(filepath, src_root);

        let Some(tree) = self.parse_tree(source_bytes) else {
            return result;
        };

        let filename = filepath
            .file_name()
            .and_then(|o| o.to_str())
            .unwrap_or("")
            .to_string();
        let is_test = crate::code_tree::parsers::shared::is_test_path(
            &rel_path,
            &filename,
            &["_test.jl", "_tests.jl", "runtests.jl"],
        );
        let mut file_info = FileInfo {
            path: rel_path.clone(),
            filename,
            loc: source.lines().count() as u32,
            module_path: module_path.clone(),
            language: "julia".to_string(),
            submodule_declarations: Vec::new(),
            imports: Vec::new(),
            exports: Vec::new(),
            annotations: None,
            is_test,
            skip_reason: None,
        };

        Self::walk_block(
            tree.root_node(),
            source_bytes,
            &module_path,
            &module_path,
            &rel_path,
            &mut result,
            &mut file_info,
        );

        result.files.push(file_info);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Parse `src` as `<root>/angelo/<rel>` and return the result.
    fn parse(rel: &str, src: &str) -> (ParseResult, std::path::PathBuf) {
        static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir()
            .join(format!("kgl_jl_{}_{}", std::process::id(), seq))
            .join("angelo");
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().expect("has parent")).expect("mkdir");
        std::fs::File::create(&path)
            .expect("create")
            .write_all(src.as_bytes())
            .expect("write");
        let result = JuliaParser::new().parse_file(&path, &root);
        let _ = std::fs::remove_dir_all(root.parent().expect("has parent"));
        (result, path)
    }

    fn imports(rel: &str, src: &str) -> Vec<String> {
        let (result, _) = parse(rel, src);
        let mut out = result
            .files
            .first()
            .map(|f| f.imports.clone())
            .unwrap_or_default();
        out.sort();
        out
    }

    #[test]
    fn include_resolves_to_the_included_file_module_path() {
        // `include` is the only construct that states a file-to-file dependency
        // in Julia, so this is the edge that matters most.
        let out = imports("src/Root.jl", "include(\"helpers.jl\")\ninclude(\"sub/deep.jl\")\n");
        assert!(out.contains(&"angelo.src.helpers".to_string()), "{out:?}");
        assert!(out.contains(&"angelo.src.sub.deep".to_string()), "{out:?}");
    }

    #[test]
    fn include_walks_up_out_of_its_directory() {
        let out = imports("src/inner/Mod.jl", "include(\"../shared/util.jl\")\n");
        assert!(out.contains(&"angelo.src.shared.util".to_string()), "{out:?}");
    }

    #[test]
    fn include_wrapped_in_joinpath_still_resolves() {
        // Idiomatic Julia wraps the path; the literal is still recoverable.
        let out = imports("src/Root.jl", "include(joinpath(@__DIR__, \"wrapped.jl\"))\n");
        assert!(out.contains(&"angelo.src.wrapped".to_string()), "{out:?}");
    }

    #[test]
    fn include_inside_a_conditional_or_function_is_found() {
        // Conditional includes are common, and a top-level-only walk misses them.
        let out = imports(
            "src/Root.jl",
            "if Sys.iswindows()\n    include(\"win.jl\")\nend\n\nfunction setup()\n    include(\"late.jl\")\nend\n",
        );
        assert!(out.contains(&"angelo.src.win".to_string()), "{out:?}");
        assert!(out.contains(&"angelo.src.late".to_string()), "{out:?}");
    }

    #[test]
    fn using_and_import_record_module_names_with_leading_dots_stripped() {
        let out = imports(
            "src/Root.jl",
            "using LinearAlgebra\nusing .Sibling: thing\nusing ..Parent\nimport Base: show\nimport OtherPkg\n",
        );
        assert!(out.contains(&"LinearAlgebra".to_string()), "{out:?}");
        assert!(out.contains(&"Sibling".to_string()), "{out:?}");
        assert!(out.contains(&"Parent".to_string()), "{out:?}");
        assert!(out.contains(&"Base".to_string()), "{out:?}");
        assert!(out.contains(&"OtherPkg".to_string()), "{out:?}");
    }

    #[test]
    fn types_carry_their_supertype_and_module_scope() {
        let (result, _) = parse(
            "src/Types.jl",
            "module Shapes\nexport Circle\nabstract type Shape end\nstruct Circle <: Shape\n    r::Float64\nend\nmutable struct Counter\n    n::Int\nend\nprimitive type Bits8 8 end\nend\n",
        );
        let by_name = |n: &str| {
            result
                .classes
                .iter()
                .find(|c| c.name == n)
                .unwrap_or_else(|| panic!("missing {n} in {:?}", result.classes))
                .clone()
        };
        let circle = by_name("Circle");
        assert_eq!(circle.kind, "struct");
        assert_eq!(circle.bases, vec!["Shape".to_string()]);
        // Declarations nest under the module they are declared in. `Shapes` does
        // not match the file stem `Types`, so it genuinely extends the scope.
        assert_eq!(circle.qualified_name, "angelo.src.Types.Shapes.Circle");
        assert_eq!(
            result.files.first().expect("file").submodule_declarations,
            vec!["Shapes".to_string()]
        );
        // `export` drives visibility; Julia types are private by default.
        assert_eq!(circle.visibility, "public");
        assert_eq!(by_name("Counter").visibility, "private");
        assert_eq!(by_name("Shape").kind, "abstract");
        assert_eq!(by_name("Bits8").kind, "primitive");
        assert!(result
            .type_relationships
            .iter()
            .any(|r| r.target_type.as_deref() == Some("Shape") && r.relationship == "extends"));
    }

    #[test]
    fn long_short_and_nested_function_forms_are_all_captured() {
        let (result, _) = parse(
            "src/Fns.jl",
            "\"\"\"\nAdds.\n\"\"\"\nfunction add(a::Int, b::Int)::Int\n    return a + b\nend\n\ndouble(x) = 2x\n\nfunction outer(y)\n    function inner(z)\n        z\n    end\n    inner(y)\nend\n\nmacro sayhi(x)\n    x\nend\n",
        );
        let names: Vec<&str> = result.functions.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"add"), "{names:?}");
        assert!(names.contains(&"double"), "{names:?}");
        assert!(names.contains(&"outer"), "{names:?}");
        // Julia nests definitions freely; a top-level-only walk would miss this.
        assert!(names.contains(&"inner"), "{names:?}");
        assert!(names.contains(&"@sayhi"), "{names:?}");

        let add = result.functions.iter().find(|f| f.name == "add").expect("add");
        assert_eq!(add.return_type.as_deref(), Some("Int"));
        assert_eq!(add.docstring.as_deref(), Some("Adds."));
        // Julia has no methods-in-types, so nothing is a method.
        assert!(result.functions.iter().all(|f| !f.is_method));
    }

    #[test]
    fn const_bindings_become_constants() {
        let (result, _) = parse("src/C.jl", "const LIMIT = 10\nconst NAME = \"x\"\n");
        let names: Vec<&str> = result.constants.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"LIMIT"), "{names:?}");
        assert!(names.contains(&"NAME"), "{names:?}");
        let limit = result
            .constants
            .iter()
            .find(|c| c.name == "LIMIT")
            .expect("LIMIT");
        assert_eq!(limit.value_preview.as_deref(), Some("10"));
    }

    #[test]
    fn calls_are_attributed_to_the_innermost_definition() {
        let (result, _) = parse(
            "src/Calls.jl",
            "function outer(y)\n    helper(y)\n    function inner(z)\n        deep(z)\n    end\nend\n",
        );
        let outer = result
            .functions
            .iter()
            .find(|f| f.name == "outer")
            .expect("outer");
        let inner = result
            .functions
            .iter()
            .find(|f| f.name == "inner")
            .expect("inner");
        fn called<'a>(f: &'a FunctionInfo) -> Vec<&'a str> {
            f.calls.iter().map(|(n, _)| n.as_str()).collect()
        }
        assert!(called(outer).contains(&"helper"));
        // A nested definition owns its own calls rather than leaking them up.
        assert!(!called(outer).contains(&"deep"));
        assert!(called(inner).contains(&"deep"));
    }

    #[test]
    fn a_module_matching_its_own_file_does_not_nest_under_itself() {
        // The Julia package convention: `src/MyPkg.jl` declaring `module MyPkg`.
        // Doubling it would qualify every declaration as `…src.MyPkg.MyPkg.x`
        // and register the file's own module as a submodule of itself.
        let (result, _) = parse(
            "src/MyPkg.jl",
            "module MyPkg\nstruct Widget\n    r::Int\nend\nmodule Inner\nstruct Deep\n    n::Int\nend\nend\nend\n",
        );
        let qname = |n: &str| {
            result
                .classes
                .iter()
                .find(|c| c.name == n)
                .unwrap_or_else(|| panic!("missing {n}"))
                .qualified_name
                .clone()
        };
        assert_eq!(qname("Widget"), "angelo.src.MyPkg.Widget");
        assert_eq!(qname("Deep"), "angelo.src.MyPkg.Inner.Deep");
        assert_eq!(
            result.files.first().expect("file").submodule_declarations,
            vec!["Inner".to_string()]
        );
    }

    #[test]
    fn test_files_are_flagged() {
        let (result, _) = parse("test/runtests.jl", "using Test\n");
        assert!(result.files.first().expect("file").is_test);
    }
}
