//! Integration tests for [`JsonSchemaPruner`].
//!
//! Exercises the full public surface through the crate API:
//! - [`ConstraintPruner`] trait methods (`is_valid`, `batch_is_valid`,
//!   `manifold_score`, `propagate`, `on_backtrack`)
//! - [`ScreeningPruner`] trait methods (`arm_id`, `arm_label`, `screen`,
//!   `batch_screen`)
//! - [`JsonState`] direct API (`new`, `step`, `is_valid_next`, `is_complete`)
//! - [`JsonSchemaPruner::parse`] and [`JsonSchemaPruner::is_complete`]
//! - Trait-object compatibility (`Box<dyn ConstraintPruner>`,
//!   `Box<dyn ScreeningPruner>`)
//! - Stateless derivation (`propagate`/`on_backtrack` are no-ops)
//! - End-to-end [`speculative_decode`] integration

use ns_engine::draft::UniformDraftModel;
use ns_engine::pruners::{
    JsonError, JsonSchemaPruner, JsonState, JSON_SCHEMA_ARM_ID, JSON_SCHEMA_LABEL,
};
use ns_engine::speculative_decode;
use ns_engine::traits::{ConstraintPruner, ScreeningPruner};
use ns_engine::types::{DecodeConfig, TokenId};

// ── Test helpers ───────────────────────────────────────────────

/// Map a character to its default token ID (Unicode code point).
fn tok(ch: char) -> TokenId {
    ch as TokenId
}

/// Convert a string to a token vector using the default mapping.
fn toks(s: &str) -> Vec<TokenId> {
    s.chars().map(tok).collect()
}

/// Build a pruner over the ASCII range (vocab_size = 128).
fn pruner() -> JsonSchemaPruner {
    JsonSchemaPruner::from_vocab_size(128)
}

/// Step a [`JsonState`] through a string, returning the final state.
fn state_from(s: &str) -> Result<JsonState, JsonError> {
    let mut state = JsonState::new();
    for ch in s.chars() {
        state.step(ch)?;
    }
    Ok(state)
}

// ── Construction / builder ─────────────────────────────────────

#[test]
fn test_from_vocab_size_defaults() {
    let p = pruner();
    assert_eq!(p.arm_id(), JSON_SCHEMA_ARM_ID);
    assert_eq!(p.arm_label(), JSON_SCHEMA_LABEL);
    assert_eq!(p.vocab_size(), 128);
}

#[test]
fn test_with_arm_id_override() {
    let p = JsonSchemaPruner::from_vocab_size(128).with_arm_id(99);
    assert_eq!(p.arm_id(), 99);
}

#[test]
fn test_with_label_override() {
    let p = pruner().with_label("custom_json");
    assert_eq!(p.arm_label(), "custom_json");
}

#[test]
fn test_with_token_map_custom() {
    // Token 0 → '{', token 1 → '}', token 2 → unmapped.
    let map = vec![Some('{'), Some('}'), None];
    let p = JsonSchemaPruner::with_token_map(map);
    assert_eq!(p.vocab_size(), 3);
    assert!(p.is_valid(0, 0, &[])); // '{' at document start
    assert!(!p.is_valid(0, 1, &[])); // '}' at document start
    assert!(!p.is_valid(0, 2, &[])); // unmapped
}

#[test]
fn test_token_to_char_lookup() {
    let p = pruner();
    assert_eq!(p.token_to_char(65), Some('A'));
    assert_eq!(p.token_to_char(123), Some('{'));
    assert_eq!(p.token_to_char(128), None); // out of range
}

// ── DocumentStart validity ─────────────────────────────────────

#[test]
fn test_document_start_accepts_value_chars() {
    let p = pruner();
    for ch in ["{", "[", "\"", "-", "0", "1", "9", "t", "f", "n"] {
        assert!(
            p.is_valid(0, tok(ch.parse::<char>().unwrap_or_default()), &[]),
            "'{ch}' should be valid at document start"
        );
    }
    // Whitespace is also valid (no-op).
    for ch in [' ', '\t', '\n', '\r'] {
        assert!(p.is_valid(0, tok(ch), &[]), "'{ch}' ws should be valid");
    }
}

#[test]
fn test_document_start_rejects_non_value_chars() {
    let p = pruner();
    for ch in ['}', ']', ':', ','] {
        assert!(
            !p.is_valid(0, tok(ch), &[]),
            "'{ch}' should be rejected at start"
        );
    }
}

// ── Objects ────────────────────────────────────────────────────

#[test]
fn test_empty_object() {
    let p = pruner();
    assert!(p.is_valid(0, tok('{'), &[]));
    assert!(p.is_valid(1, tok('}'), &toks("{")));
    assert!(p.is_complete(&toks("{}")));
}

#[test]
fn test_simple_object_step_by_step() {
    let p = pruner();
    let steps = [
        "{",
        "{\"",
        "{\"a",
        "{\"a\"",
        "{\"a\":",
        "{\"a\":1",
        "{\"a\":1}",
    ];
    let mut prefix = Vec::new();
    for (i, s) in steps.iter().enumerate() {
        let chars: Vec<char> = s.chars().collect();
        let last = *chars.last().expect("non-empty");
        if i > 0 {
            assert!(p.is_valid(i - 1, tok(last), &prefix), "step {i}: '{last}'");
        }
        prefix.push(tok(last));
    }
    assert!(p.is_complete(&prefix));
}

#[test]
fn test_nested_object() {
    let p = pruner();
    assert!(p.is_complete(&toks(r#"{"a":{"b":true}}"#)));
    assert!(p.is_complete(&toks(r#"{"outer":{"inner":{"deep":42}}}"#)));
}

#[test]
fn test_object_after_key_expects_colon() {
    let p = pruner();
    let prefix = toks(r#"{"a""#);
    assert!(p.is_valid(3, tok(':'), &prefix));
    assert!(!p.is_valid(3, tok(','), &prefix));
    assert!(!p.is_valid(3, tok('1'), &prefix));
}

#[test]
fn test_object_multiple_pairs() {
    let p = pruner();
    assert!(p.is_complete(&toks(r#"{"a":1,"b":2,"c":3}"#)));
}

// ── Arrays ─────────────────────────────────────────────────────

#[test]
fn test_empty_array() {
    let p = pruner();
    assert!(p.is_valid(0, tok('['), &[]));
    assert!(p.is_valid(1, tok(']'), &toks("[")));
    assert!(p.is_complete(&toks("[]")));
}

#[test]
fn test_simple_array() {
    let p = pruner();
    assert!(p.is_complete(&toks("[1,2,3]")));
}

#[test]
fn test_nested_array() {
    let p = pruner();
    assert!(p.is_complete(&toks("[[1],[2,3]]")));
    assert!(p.is_complete(&toks("[[[[1]]]]")));
}

#[test]
fn test_mixed_array() {
    let p = pruner();
    assert!(p.is_complete(&toks(r#"[1,"two",true,null,{},[]]"#)));
}

#[test]
fn test_array_after_comma_expects_value() {
    let p = pruner();
    let prefix = toks("[1,");
    assert!(p.is_valid(2, tok('2'), &prefix));
    assert!(!p.is_valid(2, tok(']'), &prefix)); // no trailing comma
}

// ── Strings ────────────────────────────────────────────────────

#[test]
fn test_string_simple() {
    let p = pruner();
    assert!(p.is_complete(&toks(r#""hello""#)));
    assert!(p.is_complete(&toks(r#""""#))); // empty string
}

#[test]
fn test_string_escapes() {
    let p = pruner();
    assert!(p.is_complete(&toks(r#""a\nb""#)));
    assert!(p.is_complete(&toks(r#""tab\there""#)));
    assert!(p.is_complete(&toks(r#""quote\"here""#)));
    assert!(p.is_complete(&toks(r#""back\\slash""#)));
    assert!(p.is_complete(&toks(r#""slash\/forward""#)));
}

#[test]
fn test_string_unicode_escape() {
    let p = pruner();
    assert!(p.is_complete(&toks(r#""\u0041""#))); // 'A'
    assert!(p.is_complete(&toks(r#""\uabcd""#)));
    assert!(p.is_complete(&toks(r#""\uABCD""#)));
}

#[test]
fn test_string_rejects_control_char() {
    let p = pruner();
    // Literal newline inside string — must be escaped as \n.
    assert!(!p.is_valid(2, tok('\n'), &toks("\"a")));
    assert!(!p.is_valid(2, tok('\t'), &toks("\"a")));
}

#[test]
fn test_string_rejects_invalid_escape() {
    let p = pruner();
    // \x is not a valid JSON escape.
    assert!(!p.is_valid(3, tok('x'), &toks(r#""a\"#)));
}

#[test]
fn test_string_rejects_short_unicode_escape() {
    let p = pruner();
    // \u00 then 'g' (not a hex digit).
    assert!(!p.is_valid(6, tok('g'), &toks(r#""\u00"#)));
}

#[test]
fn test_string_key_vs_value_transition() {
    let p = pruner();
    // After key string closes, expect colon (not value directly).
    let prefix = toks(r#"{"key""#);
    assert!(p.is_valid(6, tok(':'), &prefix));
    assert!(!p.is_valid(6, tok('1'), &prefix));
}

// ── Numbers ────────────────────────────────────────────────────

#[test]
fn test_number_integers() {
    let p = pruner();
    assert!(p.is_complete(&toks("0")));
    assert!(p.is_complete(&toks("42")));
    assert!(p.is_complete(&toks("123456789")));
}

#[test]
fn test_number_negative() {
    let p = pruner();
    assert!(p.is_complete(&toks("-1")));
    assert!(p.is_complete(&toks("-0")));
    assert!(p.is_complete(&toks("-42")));
}

#[test]
fn test_number_floats() {
    let p = pruner();
    assert!(p.is_complete(&toks("3.14")));
    assert!(p.is_complete(&toks("0.5")));
    assert!(p.is_complete(&toks("-0.001")));
    assert!(p.is_complete(&toks("10.0")));
}

#[test]
fn test_number_exponents() {
    let p = pruner();
    assert!(p.is_complete(&toks("1e10")));
    assert!(p.is_complete(&toks("1E10")));
    assert!(p.is_complete(&toks("1.5e3")));
    assert!(p.is_complete(&toks("1e+5")));
    assert!(p.is_complete(&toks("1e-5")));
    assert!(p.is_complete(&toks("1.5E-3")));
}

#[test]
fn test_number_rejects_leading_zero() {
    let p = pruner();
    // 01 is invalid: leading zero cannot be followed by another digit.
    assert!(!p.is_valid(1, tok('1'), &toks("0")));
    assert!(p.is_complete(&toks("0"))); // bare 0 is fine
}

#[test]
fn test_number_rejects_lone_minus() {
    let p = pruner();
    assert!(!p.is_complete(&toks("-")));
    // After '-', only a digit is valid.
    assert!(p.is_valid(1, tok('0'), &toks("-")));
    assert!(p.is_valid(1, tok('5'), &toks("-")));
    assert!(!p.is_valid(1, tok('.'), &toks("-")));
}

#[test]
fn test_number_rejects_trailing_dot() {
    let p = pruner();
    // 1. is incomplete: dot needs at least one fraction digit.
    assert!(!p.is_complete(&toks("1.")));
    assert!(p.is_valid(2, tok('5'), &toks("1.")));
}

#[test]
fn test_number_rejects_trailing_e() {
    let p = pruner();
    assert!(!p.is_complete(&toks("1e")));
    assert!(!p.is_complete(&toks("1e+")));
    assert!(p.is_valid(3, tok('5'), &toks("1e+")));
}

#[test]
fn test_number_rejects_dot_first() {
    let p = pruner();
    assert!(!p.is_valid(0, tok('.'), &[]));
}

#[test]
fn test_number_in_array_context() {
    let p = pruner();
    // After "[1", the number is still in progress (Int phase):
    // - ',' or ']' ends the number and is valid (via end_number → ArrComma).
    // - '.' is also valid — it continues the number into its fraction part
    //   (1. → 1.5), so the number is NOT "done" yet.
    // - '}' is invalid (mismatched bracket).
    let prefix = toks("[1");
    assert!(p.is_valid(2, tok(','), &prefix));
    assert!(p.is_valid(2, tok(']'), &prefix));
    assert!(p.is_valid(2, tok('.'), &prefix));
    assert!(!p.is_valid(2, tok('}'), &prefix));
}

// ── Keywords ───────────────────────────────────────────────────

#[test]
fn test_keyword_true() {
    let p = pruner();
    assert!(p.is_complete(&toks("true")));
    // Step by step: t → r → u → e
    assert!(p.is_valid(0, tok('t'), &[]));
    assert!(p.is_valid(1, tok('r'), &toks("t")));
    assert!(p.is_valid(2, tok('u'), &toks("tr")));
    assert!(p.is_valid(3, tok('e'), &toks("tru")));
}

#[test]
fn test_keyword_false() {
    let p = pruner();
    assert!(p.is_complete(&toks("false")));
    assert!(!p.is_valid(2, tok('x'), &toks("fa")));
    assert!(p.is_valid(2, tok('l'), &toks("fa")));
}

#[test]
fn test_keyword_null() {
    let p = pruner();
    assert!(p.is_complete(&toks("null")));
    assert!(!p.is_valid(3, tok('x'), &toks("nul")));
    assert!(p.is_valid(3, tok('l'), &toks("nul")));
}

#[test]
fn test_keyword_rejects_wrong_prefix() {
    let p = pruner();
    assert!(!p.is_valid(1, tok('x'), &toks("t")));
    assert!(!p.is_valid(2, tok('X'), &toks("tr")));
}

// ── Structural rejection ───────────────────────────────────────

#[test]
fn test_reject_missing_colon() {
    assert!(state_from(r#"{"a"1}"#).is_err());
}

#[test]
fn test_reject_missing_comma_between_pairs() {
    assert!(state_from(r#"{"a":1 "b":2}"#).is_err());
}

#[test]
fn test_reject_missing_comma_between_elements() {
    assert!(state_from("[1 2]").is_err());
}

#[test]
fn test_reject_trailing_comma_array() {
    assert!(state_from("[1,]").is_err());
}

#[test]
fn test_reject_trailing_comma_object() {
    assert!(state_from(r#"{"a":1,}"#).is_err());
}

#[test]
fn test_unclosed_string_is_valid_prefix_but_incomplete() {
    // `"unclosed` is a structurally VALID prefix — it's inside a string and
    // could still become `"unclosed"` (valid JSON). So `parse` succeeds.
    // It is NOT complete, however (the string hasn't closed).
    let state = state_from(r#""unclosed"#).expect("valid incomplete prefix");
    assert!(!state.is_complete());
}

#[test]
fn test_reject_unclosed_object() {
    let p = pruner();
    assert!(!p.is_complete(&toks(r#"{"a":1"#)));
}

#[test]
fn test_reject_unclosed_array() {
    let p = pruner();
    assert!(!p.is_complete(&toks("[1,2")));
}

#[test]
fn test_reject_value_after_complete_document() {
    let p = pruner();
    // After "true" (complete), a non-whitespace char is invalid.
    assert!(!p.is_valid(4, tok('1'), &toks("true")));
    assert!(p.is_valid(4, tok(' '), &toks("true"))); // ws is ok
}

#[test]
fn test_reject_mismatched_brackets() {
    let p = pruner();
    // {] is invalid: '}' expected, not ']'.
    assert!(!p.is_valid(1, tok(']'), &toks("{")));
    // [} is invalid: ']' expected, not '}'.
    assert!(!p.is_valid(1, tok('}'), &toks("[")));
}

#[test]
fn test_reject_double_dot_in_number() {
    assert!(state_from("1.2.3").is_err());
}

#[test]
fn test_reject_double_e_in_number() {
    assert!(state_from("1e2e3").is_err());
}

// ── Whitespace ─────────────────────────────────────────────────

#[test]
fn test_whitespace_between_tokens() {
    let p = pruner();
    assert!(p.is_complete(&toks("{ \"a\" : 1 }")));
    assert!(p.is_complete(&toks("[ 1 , 2 , 3 ]")));
    assert!(p.is_complete(&toks("\n  42  \n")));
}

#[test]
fn test_whitespace_inside_string_is_content() {
    let p = pruner();
    // Space (U+0020) is valid string content.
    assert!(p.is_complete(&toks("\"hello world\"")));
    // Tab (U+0009) is a control char — must be escaped.
    assert!(state_from("\"a\tb\"").is_err());
    // Newline (U+000A) is a control char — must be escaped.
    assert!(state_from("\"a\nb\"").is_err());
}

#[test]
fn test_trailing_whitespace_keeps_complete() {
    let p = pruner();
    assert!(p.is_complete(&toks("42 ")));
    assert!(p.is_complete(&toks("true\n")));
    assert!(p.is_complete(&toks("{} ")));
}

// ── Deep nesting ───────────────────────────────────────────────

#[test]
fn test_deep_nesting_valid() {
    let p = pruner();
    assert!(p.is_complete(&toks("[[[[[1]]]]]")));
    assert!(p.is_complete(&toks(r#"{"a":{"b":{"c":{"d":1}}}}"#)));
}

// ── Prefix validity (partial sequences) ────────────────────────

#[test]
fn test_partial_object_prefix() {
    let p = pruner();
    // After `{"a"` (key string closed), the parser expects ':'.
    let prefix = toks("{\"a\"");
    assert!(p.is_valid(4, tok(':'), &prefix)); // colon is exactly what's expected
    assert!(!p.is_valid(4, tok('1'), &prefix)); // value can't come before colon
    assert!(!p.is_valid(4, tok(','), &prefix)); // comma can't come before colon
}

#[test]
fn test_partial_string_prefix() {
    let p = pruner();
    let prefix = toks("\"hello");
    assert!(p.is_valid(5, tok('!'), &prefix)); // string content
    assert!(p.is_valid(5, tok('"'), &prefix)); // close string
}

#[test]
fn test_partial_number_prefix() {
    let p = pruner();
    let prefix = toks("12");
    // 12 is a valid integer — next can be more digits, dot, e, or structural.
    assert!(p.is_valid(2, tok('3'), &prefix));
    assert!(p.is_valid(2, tok('.'), &prefix));
    assert!(p.is_valid(2, tok('e'), &prefix));
}

// ── batch_is_valid ─────────────────────────────────────────────

#[test]
fn test_batch_matches_individual() {
    let p = pruner();
    let prefix = toks("{\"a\":");
    let candidates: Vec<TokenId> = (0..128).collect();
    let mut batch = vec![false; 128];
    p.batch_is_valid(0, &candidates, &prefix, &mut batch);
    for (i, &cand) in candidates.iter().enumerate() {
        let individual = p.is_valid(0, cand, &prefix);
        assert_eq!(batch[i], individual, "mismatch at token {cand}");
    }
}

#[test]
fn test_batch_invalid_prefix_all_false() {
    let p = pruner();
    // Structurally invalid prefix: value after key without a colon.
    // `parse` fails → all candidates return false regardless of char.
    let prefix = toks(r#"{"a"1}"#);
    let candidates = vec![tok('"'), tok('a'), tok('}')];
    let mut results = vec![true; 3];
    p.batch_is_valid(0, &candidates, &prefix, &mut results);
    assert_eq!(results, vec![false, false, false]);
}

#[test]
fn test_batch_empty_candidates() {
    let p = pruner();
    let mut results: [bool; 0] = [];
    p.batch_is_valid(0, &[], &[], &mut results);
    // Should not panic.
}

#[test]
fn test_batch_shorter_results() {
    let p = pruner();
    let candidates = vec![tok('{'), tok('}'), tok('[')];
    let mut results = vec![false; 2]; // shorter than candidates
    p.batch_is_valid(0, &candidates, &[], &mut results);
    assert!(results[0]); // '{' valid
    assert!(!results[1]); // '}' invalid at start
}

// ── ScreeningPruner trait ──────────────────────────────────────

#[test]
fn test_screen_binary() {
    let p = pruner();
    assert_eq!(p.screen(0, tok('{'), &[]), 1.0);
    assert_eq!(p.screen(0, tok('}'), &[]), 0.0);
}

#[test]
fn test_batch_screen_binary() {
    let p = pruner();
    let candidates = vec![tok('{'), tok('}'), tok('[')];
    let mut results = vec![0.0; 3];
    p.batch_screen(0, &candidates, &[], &mut results);
    assert_eq!(results, vec![1.0, 0.0, 1.0]);
}

#[test]
fn test_manifold_score_matches_is_valid() {
    let p = pruner();
    let prefix = toks("[1,");
    for ch in ' '..='~' {
        let token = tok(ch);
        let score = p.manifold_score(2, token, &prefix);
        let valid = p.is_valid(2, token, &prefix);
        if valid {
            assert!(
                (score - 1.0).abs() < 1e-6,
                "valid char '{ch}' score {score}"
            );
        } else {
            assert!(score.abs() < 1e-6, "invalid char '{ch}' score {score}");
        }
    }
}

// ── parse / is_complete ────────────────────────────────────────

#[test]
fn test_parse_valid_complete() {
    let p = pruner();
    let state = p.parse(&toks(r#"{"a":[1,2,3]}"#)).expect("valid JSON");
    assert!(state.is_complete());
}

#[test]
fn test_parse_valid_incomplete() {
    let p = pruner();
    let state = p.parse(&toks(r#"{"a":"#)).expect("valid prefix");
    assert!(!state.is_complete());
}

#[test]
fn test_parse_invalid_returns_err() {
    let p = pruner();
    assert!(p.parse(&toks(r#"{"a"1}"#)).is_err());
    assert!(p.parse(&toks("[01]")).is_err());
}

#[test]
fn test_parse_unmapped_token_returns_err() {
    let p = pruner(); // vocab 128
    let tokens = vec![200]; // out of range
    assert!(matches!(p.parse(&tokens), Err(JsonError::UnmappedToken)));
}

#[test]
fn test_is_complete_various() {
    let p = pruner();
    // Complete.
    assert!(p.is_complete(&toks("42")));
    assert!(p.is_complete(&toks(r#""hi""#)));
    assert!(p.is_complete(&toks("true")));
    assert!(p.is_complete(&toks("null")));
    assert!(p.is_complete(&toks("{}")));
    assert!(p.is_complete(&toks("[]")));
    assert!(p.is_complete(&toks("[1,2]")));
    // Incomplete.
    assert!(!p.is_complete(&toks("")));
    assert!(!p.is_complete(&toks("{")));
    assert!(!p.is_complete(&toks("[")));
    assert!(!p.is_complete(&toks("\"abc")));
    assert!(!p.is_complete(&toks("tru")));
    assert!(!p.is_complete(&toks("1.")));
    assert!(!p.is_complete(&toks("1e")));
}

// ── JsonState direct API ───────────────────────────────────────

#[test]
fn test_state_new_and_step() {
    let mut state = JsonState::new();
    assert!(!state.is_complete());
    state.step('{').expect("open brace");
    assert!(!state.is_complete());
    state.step('}').expect("close brace");
    assert!(state.is_complete());
}

#[test]
fn test_state_is_valid_next() {
    let mut state = JsonState::new();
    state.step('[').expect("open bracket");
    assert!(state.is_valid_next('1'));
    assert!(state.is_valid_next(']'));
    assert!(!state.is_valid_next('}'));
}

#[test]
fn test_state_clone_independent() {
    let mut a = JsonState::new();
    a.step('{').expect("brace");
    let b = a.clone();
    a.step('"').expect("quote");
    // b should be unaffected by a's mutation.
    assert!(!b.is_complete());
}

#[test]
fn test_state_is_complete_after_keyword() {
    let mut state = JsonState::new();
    for ch in "null".chars() {
        state.step(ch).expect("keyword char");
    }
    assert!(state.is_complete());
}

#[test]
fn test_state_bare_number_is_complete() {
    let mut state = JsonState::new();
    for ch in "3.14".chars() {
        state.step(ch).expect("number char");
    }
    assert!(state.is_complete());
}

// ── JsonError Display ──────────────────────────────────────────

#[test]
fn test_json_error_display() {
    assert_eq!(
        JsonError::ExpectedColon.to_string(),
        "json error: expected ':'"
    );
    assert_eq!(
        JsonError::InvalidNumber.to_string(),
        "json error: invalid number"
    );
    assert_eq!(
        JsonError::UnescapedControlChar.to_string(),
        "json error: unescaped control character in string"
    );
}

// ── Trait object compatibility ─────────────────────────────────

#[test]
fn test_constraint_pruner_trait_object() {
    let p: Box<dyn ConstraintPruner> = Box::new(pruner());
    assert!(p.is_valid(0, tok('{'), &[]));
    assert!(!p.is_valid(0, tok('}'), &[]));
}

#[test]
fn test_screening_pruner_trait_object() {
    let p: Box<dyn ScreeningPruner> = Box::new(pruner());
    assert_eq!(p.arm_id(), JSON_SCHEMA_ARM_ID);
    assert_eq!(p.arm_label(), JSON_SCHEMA_LABEL);
    assert_eq!(p.screen(0, tok('{'), &[]), 1.0);
}

// ── Stateless derivation ───────────────────────────────────────

#[test]
fn test_propagate_is_noop() {
    let mut p = pruner();
    // Propagate should not change behavior.
    p.propagate(0, tok('{'), &[]);
    assert!(p.is_valid(0, tok('{'), &[]));
    p.propagate(1, tok('}'), &toks("{"));
    assert!(p.is_valid(1, tok('}'), &toks("{")));
}

#[test]
fn test_on_backtrack_is_noop() {
    let mut p = pruner();
    p.on_backtrack(1, tok('{'), &[]);
    // After backtrack, behavior unchanged.
    assert!(p.is_valid(0, tok('{'), &[]));
}

// ── End-to-end: speculative_decode integration ─────────────────

#[test]
fn test_decode_produces_valid_json_prefix() {
    let mut pruner_instance = pruner();
    let model = UniformDraftModel::new(128);
    let config = DecodeConfig {
        max_tokens: 20,
        top_k: 128,
        seed: 42,
        backtrack: true,
        max_attempts: 5_000,
    };

    let result = speculative_decode(&model, &mut pruner_instance, &config);

    assert!(result.verified, "decode result must be verified by pruner");
    assert!(
        !result.tokens.is_empty(),
        "decode should produce at least one token"
    );

    // Every token at depth i must be valid given tokens[0..i].
    for (i, &token) in result.tokens.iter().enumerate() {
        assert!(
            pruner_instance.is_valid(i, token, &result.tokens[..i]),
            "token {token} at depth {i} is invalid — JSON structural \
             violation in decode output"
        );
    }
}
