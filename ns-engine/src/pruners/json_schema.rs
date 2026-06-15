//! JSON-schema structural-validity pruner: validates token sequences that
//! build JSON character-by-character.
//!
//! A token is valid if appending its character to the current prefix could
//! still lead to a syntactically valid JSON document. This is prefix-validity
//! semantics, analogous to [`RegexPruner`](super::RegexPruner) but for the
//! JSON grammar.
//!
//! # Incremental state machine
//!
//! The pruner maintains a [`JsonState`] — a stack-based incremental JSON
//! parser — and answers one question per candidate: *can `char(token)` follow
//! the current prefix while keeping the document structurally completable?*
//!
//! ```text
//! parent_tokens ──► chars ──► JsonState.step() × n ──► current state
//!                                                               │
//!                                                               ▼
//! candidate char ──► state.clone().step() ──► Ok? ──► is_valid
//! ```
//!
//! The state machine tracks:
//! - **Container nesting** — open `{`/`[` via a stack; close `}`/`]` must
//!   match the innermost open container.
//! - **String state** — inside `"..."`, after `\` (escape), after `\u` (hex
//!   digits). Control characters (`U+0000`–`U+001F`) must be escaped.
//! - **Number state** — leading minus, integer part (no leading zeros),
//!   fraction (`.`), exponent (`e`/`E` with optional sign).
//! - **Keyword state** — prefix matching for `true`, `false`, `null`.
//! - **Structural expectations** — colons after keys, commas between
//!   elements, close brackets matching open containers.
//!
//! # Vocabulary
//!
//! Token IDs map to characters via a configurable mapping (default: token `t`
//! in `0..=0x10FFFF` maps to `char::from_u32(t)`; higher tokens are invalid).
//! Use [`JsonSchemaPruner::with_token_map`] to override.
//!
//! # Use as a bandit arm
//!
//! `screen` returns `1.0` for valid prefixes and `0.0` otherwise — binary,
//! like [`RegexPruner`](super::RegexPruner) and
//! [`SudokuPruner`](super::SudokuPruner).
//!
//! # Stateless
//!
//! The pruner holds no mutable state. `propagate` and `on_backtrack` are
//! no-ops. All validity derives from the immutable token-to-char map and
//! the `parent_tokens` prefix, parsed fresh on each `is_valid` call.

use crate::traits::{ConstraintPruner, ScreeningPruner};
use crate::types::{ArmId, TokenId};

/// Stable arm identifier for [`JsonSchemaPruner`].
///
/// Unique within ns-engine's built-in pruners (Sudoku=1, Ngram=2, Regex=3,
/// NoPruner=4, Escrow=5, JsonSchema=6). Must not collide when composed in
/// a bandit.
pub const JSON_SCHEMA_ARM_ID: ArmId = 6;

/// Default arm label reported by [`JsonSchemaPruner::arm_label`].
pub const JSON_SCHEMA_LABEL: &str = "json_schema";

/// Maximum token ID supported by the default token-to-char mapping.
const MAX_DEFAULT_CHAR_TOKEN: TokenId = 0x10FFFF;

// ── Error type ─────────────────────────────────────────────────

/// Errors returned by [`JsonState::step`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonError {
    /// Expected the start of a value (`{`, `[`, `"`, digit, keyword).
    ExpectedValue,
    /// Expected a key string (`"`) inside an object.
    ExpectedKey,
    /// Expected a key string or `}` after `{` or `,` in an object.
    ExpectedKeyOrClose,
    /// Expected `:` after an object key.
    ExpectedColon,
    /// Expected `,` or the matching close bracket after a value.
    ExpectedCommaOrClose,
    /// Unexpected token after a complete top-level document.
    UnexpectedToken,
    /// A token has no character mapping in the pruner's vocab.
    UnmappedToken,
    /// Unescaped control character (`U+0000`–`U+001F`) inside a string.
    UnescapedControlChar,
    /// Invalid escape sequence (char after `\` is not a valid escape).
    InvalidEscape,
    /// Invalid `\u` escape (expected 4 hex digits).
    InvalidUnicodeEscape,
    /// Invalid number (violates JSON number grammar).
    InvalidNumber,
    /// Invalid keyword (does not complete `true`, `false`, or `null`).
    InvalidKeyword,
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            JsonError::ExpectedValue => "expected a value",
            JsonError::ExpectedKey => "expected a key string",
            JsonError::ExpectedKeyOrClose => "expected a key or '}'",
            JsonError::ExpectedColon => "expected ':'",
            JsonError::ExpectedCommaOrClose => "expected ',' or close bracket",
            JsonError::UnexpectedToken => "unexpected token",
            JsonError::UnmappedToken => "token has no character mapping",
            JsonError::UnescapedControlChar => "unescaped control character in string",
            JsonError::InvalidEscape => "invalid escape sequence",
            JsonError::InvalidUnicodeEscape => "invalid \\u escape sequence",
            JsonError::InvalidNumber => "invalid number",
            JsonError::InvalidKeyword => "invalid keyword",
        };
        write!(f, "json error: {msg}")
    }
}

impl std::error::Error for JsonError {}

// ── Internal state machine enums ───────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Container {
    Object,
    Array,
}

/// Sub-state of a number being parsed incrementally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumberPhase {
    /// After `-`. Need first digit (`0` or `1`–`9`).
    Minus,
    /// After leading `0` (or `-0`). Can end, go to frac, or go to exp.
    LeadingZero,
    /// In integer digits (started with `1`–`9`). Can continue, frac, exp, end.
    Int,
    /// After `.`. Need at least one fraction digit.
    Dot,
    /// In fraction digits. Can continue, go to exp, or end.
    Frac,
    /// After `e`/`E`. Need sign or digit.
    ExpMarker,
    /// After `+`/`-` in exponent. Need digit.
    ExpSign,
    /// In exponent digits. Can continue or end.
    Exp,
}

/// Sub-state of a keyword (`true`/`false`/`null`) being matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeywordPhase {
    /// After `t` (expect `r`).
    T,
    /// After `tr` (expect `u`).
    Tr,
    /// After `tru` (expect `e`, completes `true`).
    Tru,
    /// After `f` (expect `a`).
    F,
    /// After `fa` (expect `l`).
    Fa,
    /// After `fal` (expect `s`).
    Fal,
    /// After `fals` (expect `e`, completes `false`).
    Fals,
    /// After `n` (expect `u`).
    N,
    /// After `nu` (expect `l`).
    Nu,
    /// After `nul` (expect `l`, completes `null`).
    Nul,
}

/// Whether a string is an object key or a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrContext {
    /// Object key — after closing `"`, expect `:`.
    Key,
    /// Value — after closing `"`, expect comma/close/done.
    Value,
}

/// Current expectation of the incremental JSON parser.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Top-level: expecting the first value or whitespace.
    DocumentStart,
    /// Top-level: after a complete value, expecting only whitespace.
    DocumentDone,

    // ── Object phases ──
    /// After `{`: expect `"` (key), `}` (close), or whitespace.
    ObjKeyOrClose,
    /// After `,` in object: expect `"` (next key).
    ObjKey,
    /// After a key string: expect `:`.
    ObjColon,
    /// After `:`: expect a value.
    ObjValue,
    /// After a value: expect `,` or `}`.
    ObjComma,

    // ── Array phases ──
    /// After `[`: expect value, `]` (close), or whitespace.
    ArrValueOrClose,
    /// After `,` in array: expect a value.
    ArrValue,
    /// After a value: expect `,` or `]`.
    ArrComma,

    // ── String phases ──
    /// Inside string content.
    Str(StrContext),
    /// After `\` inside a string.
    StrEsc(StrContext),
    /// After `\u`, collecting hex digits (`count` = digits seen so far).
    StrU(StrContext, u8),

    // ── Composite phases ──
    /// Inside a number.
    Num(NumberPhase),
    /// Inside a keyword.
    Kw(KeywordPhase),
}

/// Outcome of [`JsonState::start_value`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ValueStart {
    /// A value was started (phase updated).
    Started,
    /// The character was whitespace (phase unchanged).
    Whitespace,
    /// The character cannot start a value.
    Invalid,
}

// ── JsonState ──────────────────────────────────────────────────

/// Incremental JSON parser state.
///
/// Tracks the structural state of a JSON document being built one character
/// at a time. Each call to [`step`](Self::step) advances the state by one
/// character; `Err` means the character is structurally invalid at the
/// current position.
///
/// Used by [`JsonSchemaPruner`] to answer prefix-validity queries: parse the
/// `parent_tokens` prefix into a `JsonState`, then check if the candidate
/// character can follow via [`is_valid_next`](Self::is_valid_next).
#[derive(Clone, Debug)]
pub struct JsonState {
    /// Open containers, outermost first. Innermost is `last()`.
    containers: Vec<Container>,
    /// Current structural expectation.
    phase: Phase,
}

impl JsonState {
    /// Create a new state at document start (expecting the first value).
    pub fn new() -> Self {
        Self {
            containers: Vec::new(),
            phase: Phase::DocumentStart,
        }
    }

    /// Whether the prefix parsed so far is a COMPLETE valid JSON document.
    ///
    /// `true` when the parser has consumed a full top-level value (and
    /// optional trailing whitespace). A bare number at top level without a
    /// trailing delimiter also counts as complete (e.g. `42`, `3.14`).
    pub fn is_complete(&self) -> bool {
        match self.phase {
            Phase::DocumentDone => true,
            Phase::Num(np) if self.containers.is_empty() => matches!(
                np,
                NumberPhase::LeadingZero | NumberPhase::Int | NumberPhase::Frac | NumberPhase::Exp
            ),
            _ => false,
        }
    }

    /// Check if `ch` can follow the current prefix WITHOUT mutating state.
    ///
    /// Equivalent to `self.clone().step(ch).is_ok()` but expressed as a
    /// non-mutating query.
    pub fn is_valid_next(&self, ch: char) -> bool {
        self.clone().step(ch).is_ok()
    }

    /// Advance the state by one character.
    ///
    /// Returns `Err` if `ch` is structurally invalid at the current position.
    pub fn step(&mut self, ch: char) -> Result<(), JsonError> {
        match self.phase {
            Phase::DocumentStart | Phase::ObjValue | Phase::ArrValue => {
                match self.start_value(ch) {
                    ValueStart::Started | ValueStart::Whitespace => Ok(()),
                    ValueStart::Invalid => Err(JsonError::ExpectedValue),
                }
            }
            Phase::DocumentDone => {
                if is_ws(ch) {
                    Ok(())
                } else {
                    Err(JsonError::UnexpectedToken)
                }
            }
            Phase::ObjKeyOrClose => self.step_obj_key_or_close(ch),
            Phase::ObjKey => self.step_obj_key(ch),
            Phase::ObjColon => self.step_obj_colon(ch),
            Phase::ObjComma => self.step_obj_comma(ch),
            Phase::ArrValueOrClose => self.step_arr_value_or_close(ch),
            Phase::ArrComma => self.step_arr_comma(ch),
            Phase::Str(ctx) => self.step_string(ctx, ch),
            Phase::StrEsc(ctx) => self.step_str_escape(ctx, ch),
            Phase::StrU(ctx, n) => self.step_str_unicode(ctx, n, ch),
            Phase::Num(np) => self.step_number(ch, np),
            Phase::Kw(kp) => self.step_keyword(ch, kp),
        }
    }

    // ── Phase transition helpers ──

    /// Transition to the post-value state based on the innermost container.
    ///
    /// Called after a value completes (string ends, number ends, keyword
    /// completes, or a container closes). If there's no parent container,
    /// the document is done.
    fn after_value(&mut self) {
        match self.containers.last() {
            None => self.phase = Phase::DocumentDone,
            Some(Container::Object) => self.phase = Phase::ObjComma,
            Some(Container::Array) => self.phase = Phase::ArrComma,
        }
    }

    /// Try to start a value with `ch`. Updates phase on success.
    fn start_value(&mut self, ch: char) -> ValueStart {
        match ch {
            '{' => {
                self.containers.push(Container::Object);
                self.phase = Phase::ObjKeyOrClose;
                ValueStart::Started
            }
            '[' => {
                self.containers.push(Container::Array);
                self.phase = Phase::ArrValueOrClose;
                ValueStart::Started
            }
            '"' => {
                self.phase = Phase::Str(StrContext::Value);
                ValueStart::Started
            }
            '-' => {
                self.phase = Phase::Num(NumberPhase::Minus);
                ValueStart::Started
            }
            '0' => {
                self.phase = Phase::Num(NumberPhase::LeadingZero);
                ValueStart::Started
            }
            '1'..='9' => {
                self.phase = Phase::Num(NumberPhase::Int);
                ValueStart::Started
            }
            't' => {
                self.phase = Phase::Kw(KeywordPhase::T);
                ValueStart::Started
            }
            'f' => {
                self.phase = Phase::Kw(KeywordPhase::F);
                ValueStart::Started
            }
            'n' => {
                self.phase = Phase::Kw(KeywordPhase::N);
                ValueStart::Started
            }
            c if is_ws(c) => ValueStart::Whitespace,
            _ => ValueStart::Invalid,
        }
    }

    // ── Structural phase handlers ──

    fn step_obj_key_or_close(&mut self, ch: char) -> Result<(), JsonError> {
        match ch {
            '}' => {
                self.containers.pop();
                self.after_value();
                Ok(())
            }
            '"' => {
                self.phase = Phase::Str(StrContext::Key);
                Ok(())
            }
            c if is_ws(c) => Ok(()),
            _ => Err(JsonError::ExpectedKeyOrClose),
        }
    }

    fn step_obj_key(&mut self, ch: char) -> Result<(), JsonError> {
        match ch {
            '"' => {
                self.phase = Phase::Str(StrContext::Key);
                Ok(())
            }
            c if is_ws(c) => Ok(()),
            _ => Err(JsonError::ExpectedKey),
        }
    }

    fn step_obj_colon(&mut self, ch: char) -> Result<(), JsonError> {
        match ch {
            ':' => {
                self.phase = Phase::ObjValue;
                Ok(())
            }
            c if is_ws(c) => Ok(()),
            _ => Err(JsonError::ExpectedColon),
        }
    }

    fn step_obj_comma(&mut self, ch: char) -> Result<(), JsonError> {
        match ch {
            ',' => {
                self.phase = Phase::ObjKey;
                Ok(())
            }
            '}' => {
                self.containers.pop();
                self.after_value();
                Ok(())
            }
            c if is_ws(c) => Ok(()),
            _ => Err(JsonError::ExpectedCommaOrClose),
        }
    }

    fn step_arr_value_or_close(&mut self, ch: char) -> Result<(), JsonError> {
        match ch {
            ']' => {
                self.containers.pop();
                self.after_value();
                Ok(())
            }
            _ => match self.start_value(ch) {
                ValueStart::Started | ValueStart::Whitespace => Ok(()),
                ValueStart::Invalid => Err(JsonError::ExpectedValue),
            },
        }
    }

    fn step_arr_comma(&mut self, ch: char) -> Result<(), JsonError> {
        match ch {
            ',' => {
                self.phase = Phase::ArrValue;
                Ok(())
            }
            ']' => {
                self.containers.pop();
                self.after_value();
                Ok(())
            }
            c if is_ws(c) => Ok(()),
            _ => Err(JsonError::ExpectedCommaOrClose),
        }
    }

    // ── String handlers ──

    fn step_string(&mut self, ctx: StrContext, ch: char) -> Result<(), JsonError> {
        match ch {
            '"' => match ctx {
                StrContext::Key => {
                    self.phase = Phase::ObjColon;
                    Ok(())
                }
                StrContext::Value => {
                    self.after_value();
                    Ok(())
                }
            },
            '\\' => {
                self.phase = Phase::StrEsc(ctx);
                Ok(())
            }
            c if (c as u32) < 0x20 => Err(JsonError::UnescapedControlChar),
            _ => Ok(()),
        }
    }

    fn step_str_escape(&mut self, ctx: StrContext, ch: char) -> Result<(), JsonError> {
        match ch {
            '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {
                self.phase = Phase::Str(ctx);
                Ok(())
            }
            'u' => {
                self.phase = Phase::StrU(ctx, 0);
                Ok(())
            }
            _ => Err(JsonError::InvalidEscape),
        }
    }

    fn step_str_unicode(&mut self, ctx: StrContext, count: u8, ch: char) -> Result<(), JsonError> {
        match ch {
            '0'..='9' | 'a'..='f' | 'A'..='F' => {
                if count + 1 >= 4 {
                    self.phase = Phase::Str(ctx);
                } else {
                    self.phase = Phase::StrU(ctx, count + 1);
                }
                Ok(())
            }
            _ => Err(JsonError::InvalidUnicodeEscape),
        }
    }

    // ── Number handler ──

    fn step_number(&mut self, ch: char, np: NumberPhase) -> Result<(), JsonError> {
        match np {
            NumberPhase::Minus => match ch {
                '0' => {
                    self.phase = Phase::Num(NumberPhase::LeadingZero);
                    Ok(())
                }
                '1'..='9' => {
                    self.phase = Phase::Num(NumberPhase::Int);
                    Ok(())
                }
                _ => Err(JsonError::InvalidNumber),
            },
            NumberPhase::LeadingZero => match ch {
                '.' => {
                    self.phase = Phase::Num(NumberPhase::Dot);
                    Ok(())
                }
                'e' | 'E' => {
                    self.phase = Phase::Num(NumberPhase::ExpMarker);
                    Ok(())
                }
                _ => self.end_number(ch),
            },
            NumberPhase::Int => match ch {
                '0'..='9' => Ok(()),
                '.' => {
                    self.phase = Phase::Num(NumberPhase::Dot);
                    Ok(())
                }
                'e' | 'E' => {
                    self.phase = Phase::Num(NumberPhase::ExpMarker);
                    Ok(())
                }
                _ => self.end_number(ch),
            },
            NumberPhase::Dot => match ch {
                '0'..='9' => {
                    self.phase = Phase::Num(NumberPhase::Frac);
                    Ok(())
                }
                _ => Err(JsonError::InvalidNumber),
            },
            NumberPhase::Frac => match ch {
                '0'..='9' => Ok(()),
                'e' | 'E' => {
                    self.phase = Phase::Num(NumberPhase::ExpMarker);
                    Ok(())
                }
                _ => self.end_number(ch),
            },
            NumberPhase::ExpMarker => match ch {
                '0'..='9' => {
                    self.phase = Phase::Num(NumberPhase::Exp);
                    Ok(())
                }
                '+' | '-' => {
                    self.phase = Phase::Num(NumberPhase::ExpSign);
                    Ok(())
                }
                _ => Err(JsonError::InvalidNumber),
            },
            NumberPhase::ExpSign => match ch {
                '0'..='9' => {
                    self.phase = Phase::Num(NumberPhase::Exp);
                    Ok(())
                }
                _ => Err(JsonError::InvalidNumber),
            },
            NumberPhase::Exp => match ch {
                '0'..='9' => Ok(()),
                _ => self.end_number(ch),
            },
        }
    }

    /// End a complete number and re-process `ch` in the post-value state.
    ///
    /// Called when a number is in a completable phase and the incoming char
    /// does not continue the number (it must be a structural char or
    /// whitespace). Transition to the post-value state via [`after_value`],
    /// then re-dispatch `ch` from there. Recursion depth is bounded to 2
    /// because the post-value phase never re-enters `step_number`.
    fn end_number(&mut self, ch: char) -> Result<(), JsonError> {
        self.after_value();
        self.step(ch)
    }

    // ── Keyword handler ──

    fn step_keyword(&mut self, ch: char, kp: KeywordPhase) -> Result<(), JsonError> {
        match kp {
            KeywordPhase::T => match ch {
                'r' => {
                    self.phase = Phase::Kw(KeywordPhase::Tr);
                    Ok(())
                }
                _ => Err(JsonError::InvalidKeyword),
            },
            KeywordPhase::Tr => match ch {
                'u' => {
                    self.phase = Phase::Kw(KeywordPhase::Tru);
                    Ok(())
                }
                _ => Err(JsonError::InvalidKeyword),
            },
            KeywordPhase::Tru => match ch {
                'e' => {
                    self.after_value();
                    Ok(())
                }
                _ => Err(JsonError::InvalidKeyword),
            },
            KeywordPhase::F => match ch {
                'a' => {
                    self.phase = Phase::Kw(KeywordPhase::Fa);
                    Ok(())
                }
                _ => Err(JsonError::InvalidKeyword),
            },
            KeywordPhase::Fa => match ch {
                'l' => {
                    self.phase = Phase::Kw(KeywordPhase::Fal);
                    Ok(())
                }
                _ => Err(JsonError::InvalidKeyword),
            },
            KeywordPhase::Fal => match ch {
                's' => {
                    self.phase = Phase::Kw(KeywordPhase::Fals);
                    Ok(())
                }
                _ => Err(JsonError::InvalidKeyword),
            },
            KeywordPhase::Fals => match ch {
                'e' => {
                    self.after_value();
                    Ok(())
                }
                _ => Err(JsonError::InvalidKeyword),
            },
            KeywordPhase::N => match ch {
                'u' => {
                    self.phase = Phase::Kw(KeywordPhase::Nu);
                    Ok(())
                }
                _ => Err(JsonError::InvalidKeyword),
            },
            KeywordPhase::Nu => match ch {
                'l' => {
                    self.phase = Phase::Kw(KeywordPhase::Nul);
                    Ok(())
                }
                _ => Err(JsonError::InvalidKeyword),
            },
            KeywordPhase::Nul => match ch {
                'l' => {
                    self.after_value();
                    Ok(())
                }
                _ => Err(JsonError::InvalidKeyword),
            },
        }
    }
}

impl Default for JsonState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Whitespace helper ──────────────────────────────────────────

/// JSON whitespace: space (U+0020), tab (U+0009), newline (U+000A),
/// carriage return (U+000D).
fn is_ws(ch: char) -> bool {
    matches!(ch, ' ' | '\t' | '\n' | '\r')
}

// ── JsonSchemaPruner ───────────────────────────────────────────

/// JSON-schema structural-validity screening pruner.
///
/// See the [module docs](self) for incremental state-machine semantics.
pub struct JsonSchemaPruner {
    /// `token_to_char[t] = Some(ch)` if token `t` maps to character `ch`.
    token_to_char: Vec<Option<char>>,
    /// Diagnostics identifier (see [`JSON_SCHEMA_ARM_ID`]).
    arm_id: ArmId,
    /// Human-readable label for logs and arena reports.
    label: String,
}

impl JsonSchemaPruner {
    /// Build a pruner with the default token-to-char mapping for a vocabulary
    /// of `vocab_size` tokens (token `t` ↔ `char::from_u32(t)`).
    pub fn from_vocab_size(vocab_size: usize) -> Self {
        Self {
            token_to_char: build_default_token_to_char(vocab_size),
            arm_id: JSON_SCHEMA_ARM_ID,
            label: JSON_SCHEMA_LABEL.to_string(),
        }
    }

    /// Build a pruner with a custom token-to-char mapping.
    ///
    /// `token_to_char[t] = Some(ch)` maps token `t` to character `ch`.
    /// `None` entries are invalid tokens (always rejected).
    pub fn with_token_map(token_to_char: Vec<Option<char>>) -> Self {
        Self {
            token_to_char,
            arm_id: JSON_SCHEMA_ARM_ID,
            label: JSON_SCHEMA_LABEL.to_string(),
        }
    }

    /// Override the arm identifier for bandit composition.
    pub fn with_arm_id(mut self, arm_id: ArmId) -> Self {
        self.arm_id = arm_id;
        self
    }

    /// Override the arm label for diagnostics.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Vocab size implied by the token-to-char map length.
    pub fn vocab_size(&self) -> usize {
        self.token_to_char.len()
    }

    /// Look up the character for a token, if mapped.
    pub fn token_to_char(&self, token: TokenId) -> Option<char> {
        let idx = token as usize;
        self.token_to_char.get(idx).copied().flatten()
    }

    /// Parse `tokens` into a [`JsonState`].
    ///
    /// Returns the final state if all tokens map to valid characters and
    /// form a structurally valid prefix, or the first error encountered.
    pub fn parse(&self, tokens: &[TokenId]) -> Result<JsonState, JsonError> {
        let mut state = JsonState::new();
        for &t in tokens {
            let ch = self.token_to_char(t).ok_or(JsonError::UnmappedToken)?;
            state.step(ch)?;
        }
        Ok(state)
    }

    /// Whether `tokens` form a COMPLETE valid JSON document.
    pub fn is_complete(&self, tokens: &[TokenId]) -> bool {
        match self.parse(tokens) {
            Ok(state) => state.is_complete(),
            Err(_) => false,
        }
    }
}

impl ConstraintPruner for JsonSchemaPruner {
    fn is_valid(&self, _depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> bool {
        let Ok(state) = self.parse(parent_tokens) else {
            return false;
        };
        let Some(ch) = self.token_to_char(token) else {
            return false;
        };
        state.is_valid_next(ch)
    }

    fn batch_is_valid(
        &self,
        _depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [bool],
    ) {
        let len = candidates.len().min(results.len());
        let state = match self.parse(parent_tokens) {
            Ok(s) => s,
            Err(_) => {
                results[..len].fill(false);
                return;
            }
        };
        for i in 0..len {
            results[i] = match self.token_to_char(candidates[i]) {
                Some(ch) => state.is_valid_next(ch),
                None => false,
            };
        }
    }

    fn manifold_score(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        ScreeningPruner::screen(self, depth, token, parent_tokens)
    }

    fn propagate(&mut self, _depth: usize, _token: TokenId, _parent_tokens: &[TokenId]) {
        // Stateless — no internal structure to update.
    }

    fn on_backtrack(&mut self, _depth: usize, _token: TokenId, _parent_tokens: &[TokenId]) {
        // Stateless — nothing to undo.
    }
}

impl ScreeningPruner for JsonSchemaPruner {
    fn arm_id(&self) -> ArmId {
        self.arm_id
    }

    fn arm_label(&self) -> &str {
        &self.label
    }

    fn screen(&self, depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        if self.is_valid(depth, token, parent_tokens) {
            1.0
        } else {
            0.0
        }
    }

    fn batch_screen(
        &self,
        depth: usize,
        candidates: &[TokenId],
        parent_tokens: &[TokenId],
        results: &mut [f32],
    ) {
        let len = candidates.len().min(results.len());
        let mut mask = vec![false; len];
        self.batch_is_valid(depth, candidates, parent_tokens, &mut mask);
        for i in 0..len {
            results[i] = if mask[i] { 1.0 } else { 0.0 };
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────

/// Default token-to-char mapping: token `t` ↔ `char::from_u32(t)` for
/// `t <= MAX_DEFAULT_CHAR_TOKEN`, else `None`.
fn default_token_to_char(token: TokenId) -> Option<char> {
    if token <= MAX_DEFAULT_CHAR_TOKEN {
        char::from_u32(token)
    } else {
        None
    }
}

/// Build the default token-to-char vector for a vocabulary of `vocab_size`.
///
/// Caps at `MAX_DEFAULT_CHAR_TOKEN + 1` to avoid allocating beyond the
/// Unicode range.
fn build_default_token_to_char(vocab_size: usize) -> Vec<Option<char>> {
    let cap = vocab_size.min((MAX_DEFAULT_CHAR_TOKEN as usize) + 1);
    let mut map = Vec::with_capacity(cap);
    for t in 0..cap {
        map.push(default_token_to_char(t as TokenId));
    }
    map
}

// ── Inline unit tests (private internals only) ─────────────────
//
// Structural tests (JsonState parsing, is_complete, reject cases) live in
// `tests/json_schema.rs` as integration tests. This module covers only what
// the integration tests CANNOT access: private helper functions and private
// struct fields.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_valid_next_does_not_mutate() {
        let mut state = JsonState::new();
        state.step('{').unwrap();
        let before = state.clone();
        assert!(state.is_valid_next('}'));
        assert_eq!(state.phase, before.phase);
        assert_eq!(state.containers.len(), before.containers.len());
    }

    #[test]
    fn test_default_token_to_char_ascii() {
        assert_eq!(default_token_to_char(65), Some('A'));
        assert_eq!(default_token_to_char(0), Some('\0'));
    }

    #[test]
    fn test_default_token_to_char_out_of_range() {
        assert_eq!(default_token_to_char(0x110000), None);
    }

    #[test]
    fn test_build_default_token_to_char_size() {
        let map = build_default_token_to_char(128);
        assert_eq!(map.len(), 128);
        assert_eq!(map[65], Some('A'));
    }

    #[test]
    fn test_build_default_token_to_char_caps_at_unicode_max() {
        let map = build_default_token_to_char(0x200000);
        assert_eq!(map.len(), (MAX_DEFAULT_CHAR_TOKEN as usize) + 1);
    }

    #[test]
    fn test_send_sync_bounds() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<JsonSchemaPruner>();
        assert_send_sync::<JsonState>();
        assert_send_sync::<JsonError>();
    }
}
