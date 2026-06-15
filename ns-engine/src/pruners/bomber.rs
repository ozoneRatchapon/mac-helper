//! `BomberActionPruner` — Bomberman domain: forward model + legality pruner.
//!
//! Phase 5 domain-generalization deliverable. The modelless thesis applied
//! to a grid Bomberman puzzle: deterministic mechanics encoded as a
//! [`GameState`] forward model ([`BomberState`]), with a matching
//! [`ConstraintPruner`] ([`BomberActionPruner`]) that mirrors legality by
//! stateless replay. No model weights, no learned policy — just deterministic
//! forward-model replay.
//!
//! # Turn order
//!
//! Each turn = (action phase) → (fuse tick) → (detonation) → (terminal check):
//!
//! - **Action**: `Move` to an adjacent walkable cell (Floor/Exit, no bomb
//!   there); `PlaceBomb` drops a bomb (fuse = `config.bomb_fuse`) at the
//!   player's cell (one per cell); `Wait` holds position.
//! - **Fuse tick**: every bomb's fuse decrements by 1 (saturating at 0).
//! - **Detonation**: bombs at fuse 0 explode in a cross (center +
//!   `blast_radius` per direction). Walls block + survive; blocks block +
//!   are destroyed (→ Floor); Floor/Exit pass the blast. A blast reaching
//!   another bomb detonates it immediately (chain reaction). A player in any
//!   blast cell dies.
//! - **Terminal**: dead → terminal, not goal (reward −1.0); on Exit →
//!   terminal, goal (reward +1.0).
//!
//! # Stateless replay convention
//!
//! [`BomberActionPruner::is_valid`] holds no mutable state for structural
//! facts. It replays `parent_tokens` from [`BomberState::initial`] (using the
//! pruner's own config copy) to reconstruct the current state, then delegates
//! to [`BomberState::is_legal`]. This mirrors the escrow pruner's
//! `EscrowState::derive` convention: `propagate` and `on_backtrack` are
//! no-ops; legality is recomputed from scratch each call.
//!
//! # Token space
//!
//! Actions occupy token IDs 0–5:
//! - 0 = Move North, 1 = Move South, 2 = Move East, 3 = Move West
//! - 4 = Place Bomb, 5 = Wait
//!
//! `BOMBER_ARM_ID = 7` (Sudoku=1, Ngram=2, Regex=3, NoPruner=4, Escrow=5,
//! JsonSchema=6, Bomber=7).

use blake3::Hasher;

use crate::game::GameState;
use crate::traits::{ConstraintPruner, ScreeningPruner};
use crate::types::{ArmId, TokenId};

// ── Action token space ─────────────────────────────────────────

/// Action token: move one cell North (decreasing `y`).
pub const ACTION_MOVE_N: TokenId = 0;
/// Action token: move one cell South (increasing `y`).
pub const ACTION_MOVE_S: TokenId = 1;
/// Action token: move one cell East (increasing `x`).
pub const ACTION_MOVE_E: TokenId = 2;
/// Action token: move one cell West (decreasing `x`).
pub const ACTION_MOVE_W: TokenId = 3;
/// Action token: place a bomb at the player's current cell.
pub const ACTION_PLACE_BOMB: TokenId = 4;
/// Action token: wait one turn (bombs tick toward detonation).
pub const ACTION_WAIT: TokenId = 5;

/// Number of distinct action tokens in the Bomber domain.
pub const BOMBER_VOCAB: usize = 6;

/// Stable arm identifier for [`BomberActionPruner`] within a
/// [`BanditPruner`](crate::bandit::BanditPruner).
///
/// Sudoku=1, Ngram=2, Regex=3, NoPruner=4, Escrow=5, JsonSchema=6, Bomber=7.
pub const BOMBER_ARM_ID: ArmId = 7;

/// Human-readable arm label for diagnostics and logging.
pub const BOMBER_LABEL: &str = "bomber-action";

/// The four cardinal directions as `(dx, dy)` offsets.
const DIRECTIONS: [(isize, isize); 4] = [(0, -1), (0, 1), (1, 0), (-1, 0)];

/// Returns the `(dx, dy)` delta for a move action, or `None` for non-move
/// actions (place bomb, wait, or unknown).
const fn move_delta(action: TokenId) -> Option<(isize, isize)> {
    match action {
        ACTION_MOVE_N => Some((0, -1)),
        ACTION_MOVE_S => Some((0, 1)),
        ACTION_MOVE_E => Some((1, 0)),
        ACTION_MOVE_W => Some((-1, 0)),
        _ => None,
    }
}

// ── Cell ───────────────────────────────────────────────────────

/// A single grid cell in the Bomberman world.
///
/// The grid is a flat `Vec<Cell>` in row-major order: index `y * width + x`.
/// Cells are either walkable (player can enter) or blocking (stops movement
/// and blast propagation):
///
/// | Variant | Walkable | Blocks blast | Destructible |
/// |---------|----------|--------------|--------------|
/// | Floor   | ✓        | ✗            | n/a          |
/// | Wall    | ✗        | ✓ (survives) | ✗            |
/// | Block   | ✗        | ✓ (destroyed)| ✓            |
/// | Exit    | ✓        | ✗            | n/a          |
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cell {
    /// Empty floor — walkable, blast passes through.
    Floor,
    /// Indestructible wall — blocks movement and blast; never destroyed.
    Wall,
    /// Destructible block — blocks movement; destroyed by blast (becomes Floor).
    Block,
    /// Exit — walkable; standing on it wins the game. Never destroyed.
    Exit,
}

impl Cell {
    /// Can the player enter this cell?
    pub const fn is_walkable(self) -> bool {
        matches!(self, Self::Floor | Self::Exit)
    }

    /// Does this cell stop blast propagation?
    pub const fn blocks_blast(self) -> bool {
        matches!(self, Self::Wall | Self::Block)
    }
}

// ── Bomb ───────────────────────────────────────────────────────

/// A bomb placed on the grid, with a countdown fuse.
///
/// Each turn, the fuse decrements by 1. When it reaches 0, the bomb
/// detonates: blast propagates in a cross pattern (center + `blast_radius`
/// cells in each cardinal direction). Chain reactions detonate bombs caught
/// in any blast immediately.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Bomb {
    /// Grid index of the bomb.
    pub pos: usize,
    /// Remaining fuse ticks. Detonates when it reaches 0 after ticking.
    pub fuse: u8,
}

// ── BomberConfig ───────────────────────────────────────────────

/// Configuration for a Bomberman level: immutable grid layout and gameplay
/// parameters.
///
/// The config is held by both [`BomberState`] (for the forward model) and
/// [`BomberActionPruner`] (for stateless replay). The pruner's config must
/// match the state's initial config for legality to agree — this is the
/// canonical-origin invariant described in
/// [`speculative_generate`](crate::speculative_generate).
///
/// # Validation
///
/// Use [`validate`](Self::validate) before constructing a state. Invalid
/// configs cause [`BomberState::initial`] to panic (fail-fast on programmer
/// error).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BomberConfig {
    /// Grid width in cells.
    pub width: usize,
    /// Grid height in cells.
    pub height: usize,
    /// Static grid layout, row-major (`y * width + x`). Must have length
    /// `width * height`.
    pub initial_grid: Vec<Cell>,
    /// Player starting position (grid index). Must be walkable.
    pub start: usize,
    /// Default fuse for newly placed bombs. Must be ≥ 1. A fuse of 1 means
    /// the bomb detonates on the same turn it is placed (suicidal unless the
    /// player is outside the blast — which they never are at placement).
    pub bomb_fuse: u8,
    /// Blast radius in cells per direction (excluding the bomb cell itself).
    /// Must be ≥ 1.
    pub blast_radius: usize,
}

/// Validation error for [`BomberConfig`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BomberConfigError {
    /// `width` or `height` is zero.
    ZeroDimension,
    /// `initial_grid.len() != width * height`.
    GridSizeMismatch,
    /// `start` is outside `[0, width * height)`.
    StartOutOfBounds,
    /// The cell at `start` is not walkable (Wall or Block).
    StartNotWalkable,
    /// `bomb_fuse` is zero (bomb would detonate instantly on placement).
    ZeroFuse,
    /// `blast_radius` is zero (bombs would have no blast).
    ZeroBlastRadius,
}

impl std::fmt::Display for BomberConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "grid dimensions must be positive"),
            Self::GridSizeMismatch => write!(f, "initial_grid length must equal width * height"),
            Self::StartOutOfBounds => write!(f, "start index is out of bounds"),
            Self::StartNotWalkable => write!(f, "start cell must be walkable (Floor or Exit)"),
            Self::ZeroFuse => write!(f, "bomb_fuse must be >= 1"),
            Self::ZeroBlastRadius => write!(f, "blast_radius must be >= 1"),
        }
    }
}

impl std::error::Error for BomberConfigError {}

impl BomberConfig {
    /// Validate the config invariants.
    ///
    /// Returns `Ok(())` if the config is safe to pass to
    /// [`BomberState::initial`].
    pub fn validate(&self) -> Result<(), BomberConfigError> {
        if self.width == 0 || self.height == 0 {
            return Err(BomberConfigError::ZeroDimension);
        }
        let expected = self.width * self.height;
        if self.initial_grid.len() != expected {
            return Err(BomberConfigError::GridSizeMismatch);
        }
        if self.start >= expected {
            return Err(BomberConfigError::StartOutOfBounds);
        }
        if !self.initial_grid[self.start].is_walkable() {
            return Err(BomberConfigError::StartNotWalkable);
        }
        if self.bomb_fuse == 0 {
            return Err(BomberConfigError::ZeroFuse);
        }
        if self.blast_radius == 0 {
            return Err(BomberConfigError::ZeroBlastRadius);
        }
        Ok(())
    }

    /// Find the grid index of the first [`Cell::Exit`], if any.
    ///
    /// A config with no Exit produces a domain that can never be "won" —
    /// [`BomberState::is_goal`] will always return `false`. This is valid for
    /// testing mechanics without a goal.
    pub fn exit_index(&self) -> Option<usize> {
        self.initial_grid.iter().position(|&c| c == Cell::Exit)
    }
}

impl Default for BomberConfig {
    /// A minimal 3×3 open grid: player starts top-left (0,0), exit at
    /// bottom-right (2,2). No walls or blocks. Bomb fuse = 3, blast radius = 1.
    fn default() -> Self {
        Self {
            width: 3,
            height: 3,
            initial_grid: vec![
                Cell::Floor,
                Cell::Floor,
                Cell::Floor,
                Cell::Floor,
                Cell::Floor,
                Cell::Floor,
                Cell::Floor,
                Cell::Floor,
                Cell::Exit,
            ],
            start: 0,
            bomb_fuse: 3,
            blast_radius: 1,
        }
    }
}

// ── BomberState ────────────────────────────────────────────────

/// The dynamic state of a Bomberman game: player position, active bombs,
/// mutable grid (blocks destroyed as the game progresses), and liveness /
/// victory flags.
///
/// Implements [`GameState`] for use with
/// [`speculative_generate`](crate::speculative_generate). The `step` method
/// is pure (snapshot semantics): it clones `self`, applies the action, ticks
/// bombs, resolves detonations, and checks terminal conditions — all without
/// mutating the receiver.
///
/// # Snapshot semantics
///
/// Because `step` returns a new state, the speculative generate loop
/// snapshots by calling `step` at branch points. No undo / backtrack logic
/// is needed — the old state is simply retained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BomberState {
    /// Player's current grid index.
    pub player: usize,
    /// Active bombs on the grid.
    pub bombs: Vec<Bomb>,
    /// Dynamic grid — a clone of `config.initial_grid` with destroyed blocks
    /// turned to [`Cell::Floor`].
    pub grid: Vec<Cell>,
    /// Is the player alive? Set to `false` when caught in a blast.
    pub alive: bool,
    /// Has the player reached the Exit? Set to `true` when the player stands
    /// on a [`Cell::Exit`] cell at end of turn.
    pub won: bool,
    /// Action token that produced this state, or `None` for the initial state.
    pub last_action: Option<TokenId>,
    /// The level configuration (immutable params + static grid layout).
    pub config: BomberConfig,
}

impl BomberState {
    /// Construct the initial state from a config.
    ///
    /// # Panics
    ///
    /// Panics if `config.validate()` returns an error. This is a fail-fast
    /// precondition: invalid configs are programmer errors, not runtime
    /// conditions.
    pub fn initial(config: BomberConfig) -> Self {
        if let Err(e) = config.validate() {
            panic!("invalid BomberConfig: {e}");
        }
        let mut state = Self {
            player: config.start,
            bombs: Vec::new(),
            grid: config.initial_grid.clone(),
            alive: true,
            won: false,
            last_action: None,
            config,
        };
        state.check_terminal();
        state
    }

    /// Human-readable one-line description for test diagnostics.
    pub fn describe(&self) -> String {
        let bombs_str = self
            .bombs
            .iter()
            .map(|b| format!("(fuse={fuse}@{pos})", fuse = b.fuse, pos = b.pos))
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "player={player} alive={alive} won={won} bombs=[{bombs_str}] last={last}",
            player = self.player,
            alive = self.alive,
            won = self.won,
            bombs_str = bombs_str,
            last = match self.last_action {
                Some(a) => a.to_string(),
                None => "-".to_string(),
            },
        )
    }

    // ── Coordinate helpers ─────────────────────────────────────

    /// Convert a grid index to `(x, y)` coordinates.
    fn to_xy(&self, index: usize) -> (usize, usize) {
        (index % self.config.width, index / self.config.width)
    }

    /// Compute the target grid index for a move action, or `None` if the
    /// target is out of bounds. Does NOT check walkability or bomb occupancy.
    fn move_target(&self, action: TokenId) -> Option<usize> {
        let (dx, dy) = move_delta(action)?;
        let (px, py) = self.to_xy(self.player);
        let nx = px as isize + dx;
        let ny = py as isize + dy;
        if nx < 0 || ny < 0 {
            return None;
        }
        let ux = nx as usize;
        let uy = ny as usize;
        if ux >= self.config.width || uy >= self.config.height {
            return None;
        }
        Some(uy * self.config.width + ux)
    }

    /// Find the grid index of the Exit cell in the current (dynamic) grid.
    fn exit_index(&self) -> Option<usize> {
        self.grid.iter().position(|&c| c == Cell::Exit)
    }

    /// Is there a destructible Block orthogonally adjacent to `pos`?
    fn adjacent_to_block(&self, pos: usize) -> bool {
        let (px, py) = self.to_xy(pos);
        for &(dx, dy) in &DIRECTIONS {
            let nx = px as isize + dx;
            let ny = py as isize + dy;
            if nx < 0 || ny < 0 {
                continue;
            }
            let ux = nx as usize;
            let uy = ny as usize;
            if ux >= self.config.width || uy >= self.config.height {
                continue;
            }
            let idx = uy * self.config.width + ux;
            if self.grid[idx] == Cell::Block {
                return true;
            }
        }
        false
    }

    // ── Forward model mechanics ────────────────────────────────

    /// Apply the action phase: move, place bomb, or wait.
    ///
    /// # Panics
    ///
    /// Panics if the action is illegal in this state. Callers must pre-filter
    /// via [`is_legal`](Self::is_legal) or [`try_step`](GameState::try_step).
    fn apply_action(&mut self, action: TokenId) {
        assert!(!self.is_terminal(), "apply_action called on terminal state");
        match action {
            ACTION_MOVE_N | ACTION_MOVE_S | ACTION_MOVE_E | ACTION_MOVE_W => {
                let target = match self.move_target(action) {
                    Some(t) => t,
                    None => panic!("move action {action} targets out-of-bounds cell"),
                };
                assert!(
                    self.grid[target].is_walkable(),
                    "move target {target} is not walkable"
                );
                assert!(
                    !self.bombs.iter().any(|b| b.pos == target),
                    "move target {target} is occupied by a bomb"
                );
                self.player = target;
            }
            ACTION_PLACE_BOMB => {
                assert!(
                    !self.bombs.iter().any(|b| b.pos == self.player),
                    "bomb already exists at player cell {player}",
                    player = self.player,
                );
                self.bombs.push(Bomb {
                    pos: self.player,
                    fuse: self.config.bomb_fuse,
                });
            }
            ACTION_WAIT => {
                // No-op — bombs tick in the detonation phase.
            }
            _ => panic!("unknown action token: {action}"),
        }
    }

    /// Tick all bomb fuses by 1, then resolve detonations with chain reactions.
    ///
    /// Bombs reaching fuse 0 are detonated. A blast propagating to another
    /// bomb detonates it immediately (chain reaction). Blocks in blast cells
    /// are destroyed (→ Floor). The player dies if caught in any blast cell.
    fn tick_and_detonate(&mut self) {
        // 1. Decrement all fuses.
        for b in &mut self.bombs {
            b.fuse = b.fuse.saturating_sub(1);
        }

        if self.bombs.is_empty() {
            return;
        }

        // 2. Seed detonation queue with bombs at fuse 0.
        let mut detonated = vec![false; self.bombs.len()];
        let mut queue: Vec<usize> = (0..self.bombs.len())
            .filter(|&i| self.bombs[i].fuse == 0)
            .collect();

        // 3. Process detonations (LIFO worklist for chain reactions).
        let mut all_blast: Vec<usize> = Vec::new();
        while let Some(i) = queue.pop() {
            if detonated[i] {
                continue;
            }
            detonated[i] = true;
            let bomb_pos = self.bombs[i].pos;
            let cells = self.compute_blast(bomb_pos);
            // Destroy blocks hit by this blast.
            for &c in &cells {
                if self.grid[c] == Cell::Block {
                    self.grid[c] = Cell::Floor;
                }
            }
            all_blast.extend_from_slice(&cells);
            // Chain reaction: queue any bomb caught in this blast.
            for (j, b) in self.bombs.iter().enumerate() {
                if detonated[j] {
                    continue;
                }
                if cells.contains(&b.pos) {
                    queue.push(j);
                }
            }
        }

        // 4. Kill the player if caught in any blast cell.
        if !all_blast.is_empty() && all_blast.contains(&self.player) {
            self.alive = false;
        }

        // 5. Remove detonated bombs.
        if detonated.iter().any(|&d| d) {
            let mut kept: Vec<Bomb> = Vec::with_capacity(self.bombs.len());
            for (i, b) in self.bombs.drain(..).enumerate() {
                if !detonated[i] {
                    kept.push(b);
                }
            }
            self.bombs = kept;
        }
    }

    /// Compute blast cells for a bomb at `pos` (no chain reactions).
    ///
    /// Returns the bomb's own cell plus `blast_radius` cells in each cardinal
    /// direction, stopping at walls (block blast, survive) and blocks (block
    /// blast, destroyed by caller).
    fn compute_blast(&self, pos: usize) -> Vec<usize> {
        let mut cells = vec![pos];
        let (bx, by) = self.to_xy(pos);
        let radius = self.config.blast_radius;
        for &(dx, dy) in &DIRECTIONS {
            for r in 1..=radius {
                let nx = bx as isize + dx * r as isize;
                let ny = by as isize + dy * r as isize;
                if nx < 0 || ny < 0 {
                    break;
                }
                let ux = nx as usize;
                let uy = ny as usize;
                if ux >= self.config.width || uy >= self.config.height {
                    break;
                }
                let idx = uy * self.config.width + ux;
                match self.grid[idx] {
                    Cell::Wall => break,
                    Cell::Block => {
                        cells.push(idx);
                        break;
                    }
                    Cell::Floor | Cell::Exit => {
                        cells.push(idx);
                    }
                }
            }
        }
        cells
    }

    /// Check terminal conditions: death (from blast) or victory (on Exit).
    ///
    /// Death takes precedence: a player who dies on the Exit cell is dead,
    /// not victorious.
    fn check_terminal(&mut self) {
        if !self.alive {
            return;
        }
        if matches!(self.grid[self.player], Cell::Exit) {
            self.won = true;
        }
    }

    // ── Screening heuristic ────────────────────────────────────

    /// Graded screen score ∈ [0.0, 1.0] for a **legal** action in this state.
    ///
    /// Used by [`BomberActionPruner::screen`]. Encodes a lightweight
    /// domain heuristic (no search):
    /// - Move that reduces Manhattan distance to exit → `0.9`
    /// - Move that does not reduce distance → `0.6`
    /// - Place bomb adjacent to ≥1 block → `0.85`
    /// - Place bomb with no adjacent block → `0.5`
    /// - Wait → `0.4`
    ///
    /// Caller must ensure the action is legal; this method returns `0.0` for
    /// illegal or unknown actions.
    fn action_screen_score(&self, action: TokenId) -> f32 {
        match action {
            ACTION_MOVE_N | ACTION_MOVE_S | ACTION_MOVE_E | ACTION_MOVE_W => {
                let exit = match self.exit_index() {
                    Some(e) => e,
                    None => return 0.6,
                };
                let (ex, ey) = self.to_xy(exit);
                let (px, py) = self.to_xy(self.player);
                let cur_dist = (px as isize - ex as isize).unsigned_abs()
                    + (py as isize - ey as isize).unsigned_abs();
                let target = match self.move_target(action) {
                    Some(t) => t,
                    None => return 0.6,
                };
                let (tx, ty) = self.to_xy(target);
                let new_dist = (tx as isize - ex as isize).unsigned_abs()
                    + (ty as isize - ey as isize).unsigned_abs();
                if new_dist < cur_dist {
                    0.9
                } else {
                    0.6
                }
            }
            ACTION_PLACE_BOMB => {
                if self.adjacent_to_block(self.player) {
                    0.85
                } else {
                    0.5
                }
            }
            ACTION_WAIT => 0.4,
            _ => 0.0,
        }
    }
}

// ── GameState impl ─────────────────────────────────────────────

impl GameState for BomberState {
    fn last_action(&self) -> Option<TokenId> {
        self.last_action
    }

    fn step(&self, action: TokenId) -> Self {
        let mut next = self.clone();
        next.last_action = Some(action);
        next.apply_action(action);
        next.tick_and_detonate();
        next.check_terminal();
        next
    }

    /// Direct legality predicate — O(1) per call, overriding the default
    /// `legal_actions` enumeration. This is the authoritative legality check
    /// that [`BomberActionPruner`] mirrors via stateless replay.
    fn is_legal(&self, action: TokenId) -> bool {
        if self.is_terminal() {
            return false;
        }
        match action {
            ACTION_MOVE_N | ACTION_MOVE_S | ACTION_MOVE_E | ACTION_MOVE_W => {
                let target = match self.move_target(action) {
                    Some(t) => t,
                    None => return false,
                };
                if !self.grid[target].is_walkable() {
                    return false;
                }
                if self.bombs.iter().any(|b| b.pos == target) {
                    return false;
                }
                true
            }
            ACTION_PLACE_BOMB => !self.bombs.iter().any(|b| b.pos == self.player),
            ACTION_WAIT => true,
            _ => false,
        }
    }

    fn legal_actions(&self) -> Vec<TokenId> {
        if self.is_terminal() {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(BOMBER_VOCAB);
        for a in 0..BOMBER_VOCAB as TokenId {
            if self.is_legal(a) {
                out.push(a);
            }
        }
        out
    }

    fn is_terminal(&self) -> bool {
        !self.alive || self.won
    }

    fn is_goal(&self) -> bool {
        self.won
    }

    fn reward(&self) -> f32 {
        if self.won {
            1.0
        } else if !self.alive {
            -1.0
        } else {
            0.0
        }
    }

    fn hash(&self) -> [u8; 32] {
        let mut h = Hasher::new();
        h.update(&(self.player as u32).to_le_bytes());
        h.update(&(self.alive as u8).to_le_bytes());
        h.update(&(self.won as u8).to_le_bytes());
        h.update(&(self.bombs.len() as u32).to_le_bytes());
        for b in &self.bombs {
            h.update(&(b.pos as u32).to_le_bytes());
            h.update(&b.fuse.to_le_bytes());
        }
        for &c in &self.grid {
            h.update(&[c as u8]);
        }
        *h.finalize().as_bytes()
    }
}

// ── BomberActionPruner ─────────────────────────────────────────

/// Stateless-replay legality pruner for the Bomberman domain.
///
/// The modelless thesis in action: the entire Bomberman legality ruleset is
/// encoded as a deterministic forward model ([`BomberState`]), and this
/// pruner mirrors it by replaying `parent_tokens` from a canonical initial
/// state. No model weights, no learned policy — just deterministic replay.
///
/// The pruner holds an immutable [`BomberConfig`] (the canonical origin).
/// [`is_valid`](ConstraintPruner::is_valid) replays `parent_tokens` to
/// reconstruct the current state, then delegates to
/// [`BomberState::is_legal`]. By construction, the pruner's verdict is
/// identical to the forward model's own legality check.
///
/// # Example
///
/// ```
/// use ns_engine::pruners::{BomberActionPruner, BomberConfig, BomberState};
/// use ns_engine::pruners::{ACTION_MOVE_E, ACTION_PLACE_BOMB, ACTION_WAIT};
/// use ns_engine::traits::ConstraintPruner;
///
/// let pruner = BomberActionPruner::new(BomberConfig::default());
///
/// // From the start (top-left), moving East is legal.
/// assert!(pruner.is_valid(0, ACTION_MOVE_E, &[]));
///
/// // Placing a bomb at the start is legal (no bomb there yet).
/// assert!(pruner.is_valid(0, ACTION_PLACE_BOMB, &[]));
///
/// // After placing a bomb, placing another at the same cell is illegal.
/// assert!(!pruner.is_valid(1, ACTION_PLACE_BOMB, &[ACTION_PLACE_BOMB]));
/// ```
pub struct BomberActionPruner {
    config: BomberConfig,
}

impl BomberActionPruner {
    /// Create a new pruner with the given canonical config.
    ///
    /// The config must match the [`BomberState`] initial config used by the
    /// generate loop. A mismatch would cause the pruner's legality verdict to
    /// diverge from the real forward model.
    pub fn new(config: BomberConfig) -> Self {
        if let Err(e) = config.validate() {
            panic!("invalid BomberConfig: {e}");
        }
        Self { config }
    }

    /// Borrow the canonical config.
    pub fn config(&self) -> &BomberConfig {
        &self.config
    }

    /// Reconstruct the state after replaying `history` from the canonical
    /// initial state. Returns `None` if `history` contains an illegal action
    /// (the trace is unreachable from the canonical origin).
    pub fn state_of(&self, history: &[TokenId]) -> Option<BomberState> {
        self.replay(history)
    }

    /// Stateless replay: reconstruct the current state from `parent_tokens`.
    fn replay(&self, parent_tokens: &[TokenId]) -> Option<BomberState> {
        let mut state = BomberState::initial(self.config.clone());
        for &a in parent_tokens {
            state = state.try_step(a)?;
        }
        Some(state)
    }
}

impl ConstraintPruner for BomberActionPruner {
    fn is_valid(&self, _depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> bool {
        match self.replay(parent_tokens) {
            Some(state) => state.is_legal(token),
            None => false,
        }
    }
}

impl ScreeningPruner for BomberActionPruner {
    fn arm_id(&self) -> ArmId {
        BOMBER_ARM_ID
    }

    fn arm_label(&self) -> &str {
        BOMBER_LABEL
    }

    fn screen(&self, _depth: usize, token: TokenId, parent_tokens: &[TokenId]) -> f32 {
        let state = match self.replay(parent_tokens) {
            Some(s) => s,
            None => return 0.0,
        };
        if !state.is_legal(token) {
            return 0.0;
        }
        state.action_screen_score(token)
    }
}

// ── Inline unit tests for private internals ────────────────────
//
// Integration tests covering the full public API live in `tests/bomber.rs`.
// These inline tests cover the private mechanics: blast geometry, chain
// reactions, move deltas, and cell predicates.

#[cfg(test)]
mod tests {
    use super::*;

    // ── move_delta ──

    #[test]
    fn test_move_delta_cardinal_directions() {
        assert_eq!(move_delta(ACTION_MOVE_N), Some((0, -1)));
        assert_eq!(move_delta(ACTION_MOVE_S), Some((0, 1)));
        assert_eq!(move_delta(ACTION_MOVE_E), Some((1, 0)));
        assert_eq!(move_delta(ACTION_MOVE_W), Some((-1, 0)));
    }

    #[test]
    fn test_move_delta_non_move_actions() {
        assert_eq!(move_delta(ACTION_PLACE_BOMB), None);
        assert_eq!(move_delta(ACTION_WAIT), None);
        assert_eq!(move_delta(99), None);
    }

    // ── Cell predicates ──

    #[test]
    fn test_cell_walkable_and_blocks_blast() {
        assert!(Cell::Floor.is_walkable());
        assert!(!Cell::Floor.blocks_blast());

        assert!(!Cell::Wall.is_walkable());
        assert!(Cell::Wall.blocks_blast());

        assert!(!Cell::Block.is_walkable());
        assert!(Cell::Block.blocks_blast());

        assert!(Cell::Exit.is_walkable());
        assert!(!Cell::Exit.blocks_blast());
    }

    // ── compute_blast geometry ──

    fn open_5x5_config() -> BomberConfig {
        BomberConfig {
            width: 5,
            height: 5,
            initial_grid: vec![Cell::Floor; 25],
            start: 12, // center
            bomb_fuse: 3,
            blast_radius: 2,
        }
    }

    #[test]
    fn test_blast_cross_pattern_open_grid() {
        let s = BomberState::initial(open_5x5_config());
        let blast = s.compute_blast(12); // center (2,2), radius 2
        assert_eq!(blast.len(), 9, "center + 2 per direction");
        for &c in &[12, 2, 7, 17, 22, 10, 11, 13, 14] {
            assert!(blast.contains(&c), "blast should include {c}");
        }
    }

    #[test]
    fn test_blast_blocked_by_wall() {
        // Place walls at (2,1)=7 and (2,3)=17 and (1,2)=11 and (3,2)=13.
        // Blast from center should be just the center (all arms blocked at r=1).
        let mut cfg = open_5x5_config();
        for &w in &[7, 11, 13, 17] {
            cfg.initial_grid[w] = Cell::Wall;
        }
        let s = BomberState::initial(cfg);
        // Walls at r=1 in all directions; blast is just the center.
        assert_eq!(s.compute_blast(12), vec![12]);
    }

    #[test]
    fn test_blast_block_destroyed_and_stops() {
        // Block at (2,1)=7 (up 1). Blast hits block, includes block cell,
        // stops (does not reach (2,0)=2).
        let mut cfg = open_5x5_config();
        cfg.initial_grid[7] = Cell::Block; // up1 from center
        let s = BomberState::initial(cfg);
        let blast = s.compute_blast(12);
        for &c in &[12, 7, 17, 22, 10, 11, 13, 14] {
            assert!(blast.contains(&c));
        }
        assert!(!blast.contains(&2), "block stops blast beyond it");
    }

    // ── Chain reaction detonation ──

    #[test]
    fn test_chain_reaction_detonates_linked_bomb() {
        // 1×5 grid. Bomb A@0 (fuse 1, detonates this tick); bomb B@2 (fuse 5,
        // only via chain — A's radius-2 blast reaches index 2).
        let config = BomberConfig {
            width: 5,
            height: 1,
            initial_grid: vec![
                Cell::Floor,
                Cell::Floor,
                Cell::Floor,
                Cell::Floor,
                Cell::Exit,
            ],
            start: 4, // player safe — mechanics test, not legality
            bomb_fuse: 5,
            blast_radius: 2,
        };
        let mut s = BomberState::initial(config);
        s.player = 4;
        s.bombs = vec![Bomb { pos: 0, fuse: 1 }, Bomb { pos: 2, fuse: 5 }];
        s.tick_and_detonate();
        assert!(s.bombs.is_empty(), "both bombs gone after chain");
    }

    #[test]
    fn test_detonation_kills_player_in_blast() {
        // Player at center, bomb at center with fuse 1 → dies immediately.
        let mut state = BomberState::initial(open_5x5_config());
        state.bombs = vec![Bomb { pos: 12, fuse: 1 }];
        state.player = 12;
        assert!(state.alive);
        state.tick_and_detonate();
        assert!(!state.alive, "player at bomb center must die");
    }

    #[test]
    fn test_detonation_destroys_blocks() {
        let mut cfg = open_5x5_config();
        cfg.initial_grid[7] = Cell::Block; // up1 from center
        cfg.start = 0; // player far away
        let mut s = BomberState::initial(cfg);
        s.bombs = vec![Bomb { pos: 12, fuse: 0 }]; // detonate now
        s.tick_and_detonate();
        assert_eq!(s.grid[7], Cell::Floor, "block destroyed");
    }

    // ── action_screen_score ──

    #[test]
    fn test_screen_score_move_toward_exit() {
        // Default 3×3: player at 0, exit at 8 (2,2). Moving East (→1) or
        // South (→3) reduces Manhattan distance. Moving West/North is
        // out-of-bounds (illegal), so only E and S are "toward".
        let state = BomberState::initial(BomberConfig::default());
        assert_eq!(state.action_screen_score(ACTION_MOVE_E), 0.9);
        assert_eq!(state.action_screen_score(ACTION_MOVE_S), 0.9);
    }

    #[test]
    fn test_screen_score_wait_low() {
        let state = BomberState::initial(BomberConfig::default());
        assert_eq!(state.action_screen_score(ACTION_WAIT), 0.4);
    }

    #[test]
    fn test_screen_score_place_bomb_no_block() {
        // Open 3×3: no blocks adjacent to start → 0.5.
        let state = BomberState::initial(BomberConfig::default());
        assert_eq!(state.action_screen_score(ACTION_PLACE_BOMB), 0.5);
    }

    #[test]
    fn test_screen_score_place_bomb_adjacent_to_block() {
        let mut grid = BomberConfig::default().initial_grid;
        grid[1] = Cell::Block; // East neighbor of start
        let cfg = BomberConfig {
            initial_grid: grid,
            ..BomberConfig::default()
        };
        let state = BomberState::initial(cfg);
        assert_eq!(state.action_screen_score(ACTION_PLACE_BOMB), 0.85);
    }
}
