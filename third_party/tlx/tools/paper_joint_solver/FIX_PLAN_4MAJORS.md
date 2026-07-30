# Fix Plan: Four Major Paper-Fidelity Defects

This plan is based on the six-axis fidelity review from 2026-07-27 (with every
finding confirmed by adversarial review) and was further revised by three
critics using evidence from the actual code and dumps. All file:line references
in this document are relative to the current branch tip (`d07c30f40`, where
`paper_joint_solver/` is byte-for-byte identical to the version reviewed).

The four items to fix (all confirmed as major):

1. The handoff IR classifies `cross_warp_dependencies` at group granularity,
   while paper §4.3 and both this repository's solver and emit gate operate at
   physical-warp (lane) granularity.
2. The IR does not materialize a software-pipelined program in the sense of
   paper §5 (the ⌈L/I⌉-copy expansion is left to the expert).
3. Streaming (§5.3) never activates on forward inputs (any incoming edge
   disqualifies a node, including scalar address arithmetic and signal-only
   token edges).
4. The CONCURRENCY window contains an undisclosed relaxation for TC/TMA
   (`win=1` instead of `cycles(o)` from paper Figure 6), and archived solutions
   contain placements forbidden by the paper's constraints.

---

## Prerequisite Facts (Empirically Confirmed During Critique)

- **The baseline is red**: `tests/test_ddg_and_modulo.py` has two deterministic
  failures: `test_joint_lane_symmetry_orders_membership_columns` (:307,
  unsorted lane columns) and
  `test_joint_uses_producer_specific_spill_cost` (:847, expected sat but got
  unsat, possibly a regression from the commit that tightened the aa4 window).
  The latter is directly in Fix B's territory.
- **All nine checked-in `*_solution*.json` files use the legacy schema**. Tests
  assert that they raise `LegacySolutionError`, and `refit_check.py:58` still
  reads one of them. **Rerun results must use new filenames and must never
  overwrite the existing files.**
- pre-commit globally excludes `third_party/tlx/tools/`, so it will not format
  this directory.

---

## Phase 0: Baseline and Triage (Half a Day)

1. Set the solver dynamic-library path according to the `SOLVER_LIB_PATH`
   convention in `run_ablations.sh`, run `pytest tests/ -q`, and record the
   baseline (expected: 2 failed / 129 passed).
2. **Phase 0b (required)**: Triage the two red tests above. Fix them or add an
   explicit `xfail(reason=...)` and document it. Otherwise, regressions from
   Fix B cannot be distinguished from existing breakage. For the unsat result
   in `test_joint_uses_producer_specific_spill_cost`, first determine whether
   it is a legitimate consequence of the tightened window or an encoding bug.
3. While here, update the `importorskip` pattern to catch `YicesAPIException`
   as well (optional, one line).

---

## Phase 1: Solver Side (Fix A + A2 + B, Parallelizable with Phase 2)

### Fix A: Streaming Classification (`ddg.py:369`)

Change `has_incoming = {edge.dst for edge in edges}` to:

```python
has_incoming = {e.dst for e in data_edges if e.src not in prob.emitter_infra}
```

`data_edges` (:356) already excludes signal-only edges; `emitter_infra`
(:350-354) comes from the `warp_group < 0` markers in the baseline graph (in
both fwd dumps, these are the muli/addi nodes 0/1). Rationale: G in paper §5.3
(Figure 9) has no address-arithmetic nodes, and TMA loads are graph sources.

**Empirical revisions from the critics:**

- The normalization pool **will not** move. Only two outgoing edges with
  latency 556 are zeroed (fwd 12→0 and subtiled 1→0), and `normalization_f` is
  unchanged across all three dumps. Existing solutions are invalidated by the
  `solver_sources_sha256` change and the zeroed latencies on these two edges,
  not by global C′ drift.
- bwd has a surprise: `tt.descriptor_reduce` (id 35) is also marked as infra
  (despite being a tile-level op). This is harmless today because it has zero
  outgoing edges, but add an **invariant assertion** that exempted infra
  producers may feed VL nodes only through scalar results. Also add a bwd
  regression test pinning `streaming=={0,2}` (7 and 9 are correctly
  disqualified by incoming edges from real `tt.addptr` tile pointers).
- Tests: **extend rather than rewrite**
  `test_any_incoming_dependence_disqualifies_streaming` (:701-739), because its
  incoming edge is a real data edge and the test still passes after Fix A. The
  inline oracle at :66 in `test_load_and_derive` duplicates the old predicate
  and must be updated as well. Add a schedule_graph fixture helper that can
  create nodes with `warp_group<0` (the existing `_write_ddg` does not support
  this, and there is no conftest). The correct reference for the zeroing test
  is :670-698.
- **Bypass callers**: `viz.py:160`, `strategy_report.py:68`, and
  `desym_check.py:23` do not pass the baseline graph. After Fix A, the models
  they render will diverge from the model solved. Thread `--baseline-graph`
  through these callers, or emit a loud warning when `emitter_infra` is empty
  but the DDG contains infra candidates.

### Fix A2 (New Decision Point): WS-Semantics Exemption for Infra Edges

The CONCURRENCY spill gate at `joint_smt.py:841-848` counts infra producers as
register predecessors (`regs[1]=1` in fwd, and node 1 can never share a group
with the load because of the VARIABLELATENCY iff, so the gate is always true).
Likewise, CROSS-WARPSPILLS at `joint_smt.py:793-824` charges spill for the infra
edge 1→2. Once Fix B removes the carve-out, this creates a 12-cycle
mutual-exclusion window on W_vl for two TMA loads that does not exist in the
paper (in the paper's G, loads are source nodes, so the gate never fires).

**Recommended decision (prefer the former)**: Apply the same principle as Fix
A. Infra edges retain only DEPENDENCE ordering semantics and are **exempt from
all WS semantics** (streaming disqualification, CONCURRENCY gate predecessors,
and CROSS-WARPSPILLS charging); disclose this as an adapter rule in
FIDELITY_REVIEW. Alternatively, retain the behavior and explicitly disclose
the divergence. This decision must be made before the Phase 3 reruns because it
changes the non-subtiled fwd solution.

### Fix B: CONCURRENCY Window (`joint_smt.py:860-861`)

```python
win = prob.lat[o]          # Paper Fig. 6: [t-(cycles(o)-1), t], unconditionally
if win == 0: continue      # cycles(o)=0 → empty window (the paper formula itself)
```

Delete the Fig-2 rationale comment at :854-859 and replace it with a reference
to the paper's formula. Optionally preserve the old behavior behind an explicit
non-paper flag (off by default) for ablations.

**Empirical revision from the critics (important)**: This is **not a pure
tightening**. The subtiled case has 45/55 nodes with `lat==0` (25 native zeros +
20 truncated to zero by ZLP normalization; the lb=0 at `normalize.py:48` agrees
with the paper's C′≥0). The current code gives them `win=max(1,lat)=1`, which
forbids co-issuing at the same instant; skip-on-zero exempts them entirely. Fix
B therefore tightens TC/TMA while relaxing approximately 82% of the ops. The
conclusion is unchanged (this is the literal semantics of the paper's formula),
but:

- FIDELITY_REVIEW must disclose the relaxation as well.
- Remove the expectation that "the feasible region shrinks". The old
  (II\*,L\*) may remain SAT rather than becoming UNSAT.
- Add tests accepting a lat-0 op at t_v (new behavior) and rejecting lat≥1
  inside the window (regression coverage reproducing the shape of the archived
  counterexample: same-warp MMA with lat=2 at t−1 versus a blocking tmem_load
  at t).

**Emit-gate hardening (strongly recommended)**: Add a CONCURRENCY recheck over
the straight-line program to the gate in `schedule_plan.py` (alongside the
existing COMPLETION / DEPENDENCE / CAPACITY / VARIABLELATENCY checks). Its
semantics must be **byte-for-byte identical** to the solver, including
skip-on-zero and the A2 decision. O(V²·copies²), V≤55, so the cost is
negligible. **First dry-run** the two handwritten fixture constructors in
`test_skc_cute.py` (:104-156 and
:249-284; their cycles have never been checked against CONCURRENCY). If they
violate the window, include fixture rescheduling in this task to avoid a late
implementation failure.

---

## Phase 2: IR Side (Fix C + D, One v2 Schema Bump)

### Fix C: Lane-Granularity Cross-Warp Dependencies

1. **Canonicalize lanes first** (a prerequisite defect found by the critics):
   The solver happens to emit sorted tuples, but nothing downstream enforces
   this. Sort in `schedule_plan.py:87` (`_lane_dict`) and require sorted order
   in `_schema.py` v2's `_validate_instructions`. Tuple equality then becomes
   equivalent to set equality, and this also fixes the existing gate's
   "permuted-lane phantom spill" divergence.
2. Add `producer_lanes` / `consumer_lanes` / `spill_cost` to `ScheduledEdge`
   (:109-118). Set `spill_cost` to 0 for signal-only edges, mirroring the
   exemption at `joint_smt.py:798` (the current gate at :391-397 overcharges
   token edges; align and document this at the same time).
3. Change the predicate at `pipelined_ir.py:82-88` to:
   `producer_group != consumer_group or producer_lanes != consumer_lanes`.
4. In `_schema.py` v2, validate the lane fields and recompute the cross set with
   the lane predicate; require `spill_cost >= 0` and require it to be 0 on
   non-cross edges. Strengthen the timing check at :434-437 to
   `available >= cycle[src] + latency + spill_cost` (otherwise `spill_cost` is
   dead data).
5. **The actual audit.py workload**: The channel/program_order rules
   **automatically** follow the cross list because both are keyed off
   `cross_dependency_map`; no change is needed there. The real changes are to
   include the new fields in the `facts` tuples in `_mapping_dependencies`
   (:427-434) and `_sync_dependencies` (:972-979), and to enumerate the new
   fields explicitly in `_sync_template` at `scaffold.py:217-236` (otherwise an
   expert manual with stale fields could still pass review).
6. **Synchronize both schema-bump sites**: `pipelined_ir.py:20-21` and
   `_schema.py:11-12` use independent literals. Bump both in the same commit
   (or have `_schema` import the former), optionally adding an import-time
   assertion in `skc/compiler.py` that they match. Update `SKC_DESIGN.md:34,36`
   as well. The solution input schema **does not change** (the input format is
   unchanged).

### Fix D: Materialize the Pipeline Expansion (`pipelined_program` Section)

Add the following to `_build_ir` (pure scheduling arithmetic that does not
cross the §6.1 boundary, as confirmed by the critics):

- `instances`: `{node, copy, cycle: node.cycle + copy*ii, region}`, with
  copy ∈ [0, copies), which is the op-table from paper Figure 3.
- `instance_dependencies`: consumer (v,i) → producer (u, i−δ); for an
  out-of-bounds producer (`j < 0`), mark the dependency as `external` and record
  its `carried_distance` (δ≥0 is already guaranteed by the schema, so only the
  lower boundary can be crossed).
- `steady_state.slots`: exactly one instance of each node within the
  steady-state window, with `iteration_lag = (copies-1) - copy == stage`, which
  is the renaming fact for `V[i-1]` in Figure 1f.

In `_schema.py` v2, recompute the entire expansion from (nodes, ii, copies,
regions) and compare for **object equality** (following the existing
region-recomputation pattern at :280-281, not byte comparison).

**Test properties (corrected by the critics)**: steady contains exactly |nodes|
instances, one per node, and lag==stage; prologue+epilogue together contain
|nodes|·(copies−1). (The original "region population = window width" property
is false; the dimensions do not even match.)

**Combined test checklist for both fixes**: Update the mechanism-selection
helpers at `test_skc.py:337` / `:554` to use the lane predicate. Update v1
string hardcodes at `test_skc.py:84,197-199,625-626,701-703` and
`test_skc_cute.py:178,192`. Extend `test_skc_cute.py:287-295` so a mixed-lane
edge must enter the cross list and the sync audit requires a channel. Note that
the "fixture" in `test_skc.py` is an in-code dict constructor (`_ir()` at :73),
so update the constructor rather than regenerating a file.

---

## Phase 3: Rerun Solving (Background, Serial, Separate Log per Case)

Pin the command for each case (provenance will lock `normalization_u` and the
machine manifest):

| case | Key command arguments |
|---|---|
| fwd subtiled | `--baseline-graph …subtiled/schedule_graph.json --warp-fixed-overhead 4` |
| fwd non-subtiled | Same as above + **decide** whether to use `--normalization-u 150` (matching history) or disclose a change to the default 300 |
| bwd | case4's ddg + graph |
| bwd-LR | **Choose and disclose** `--reg-budget < 8160`: the legacy value 8192 is now rejected outright by the CLI (limit 32×255=8160), and 8192=256/thread is not actually "reduced"; the paper only says "reduced budget" and gives no numeric value |

Also rerun the four sets of UNSAT ablations in `run_ablations.sh` (first update
the warp parameters according to FIDELITY_REVIEW U5). Explicitly retain
`--ilp-seconds/--smt-seconds/--max-wall-s`. Historical runs took 4–17 minutes
per case (254–990 s), and all predate full-L-window becoming the default, so
**budget for slower runs** and run them serially (`refit_check`'s precedent of
a 150 GiB RLIMIT indicates memory pressure). **Always write outputs under new
filenames** (for example, `solutions/<case>_v7.json`). Also decide what to do
with `refit_check.py`: update the expected tuple, or annotate that CHANGE is the
expected conclusion.

---

## Phase 4: End-to-End Validation (Audits Must "Reject," Not "Pass")

For each new solution:

1. `python -m skc handoff --solution … --ddg … --baseline-graph … --ir-out … --manifest-out …`
2. `python -m skc scaffold --ir … --handoff … --out-dir …`
3. **Negative check: every audit must reject the draft.** `test_skc.py:747`
   already establishes that "scaffold output never passes review" as a
   normative requirement. Allowing audit-bundle to pass would mean fabricating
   expert approval, directly violating the paper's boundary.
4. Run the full pytest suite.

The audit pass path is covered only by synthetic "approved" fixtures in tests.

---

## Phase 5: Documentation and Cleanup

- `FIDELITY_REVIEW.md`: Remove the obsolete U6/U9 text. Add disclosures covering
  the restored exact Figure 6 window, the relaxation caused by empty zero-latency
  windows, the streaming predicate (incoming data edges from non-infra
  producers), A2's WS-semantics exemption for infra edges (if adopted), the
  lane-granularity cross list, materialization of `pipelined_program`, and
  zeroing `spill_cost` for signal-only edges.
- `REPORT.md`: Mark the historical table as "old model" and add a table for the
  new solutions. Update the schema string in `SKC_DESIGN.md` to v2.
- Formatting: pre-commit is a no-op for this directory; run `ruff`/`yapf`
  manually if needed.
- Do not commit (unless separately requested).

---

## Acceptance Criteria

1. On the subtiled dump, `streaming == {2,3}`, outgoing edge latencies are 0,
   and RRT is preserved; on bwd, `streaming == {0,2}`.
2. The archived counterexample shape (same-warp MMA with lat=2 at t−1 +
   blocking consumer at t) is rejected by both the solver and the emit gate;
   coexisting lat-0 ops at the same instant are accepted.
3. A same-group, mixed-lane edge enters `cross_warp_dependencies`, and the sync
   audit requires a channel; a same-group, same-lane edge remains legal with
   program_order.
4. The v2 IR's `pipelined_program` passes the schema's whole-expansion
   recomputation equality check; steady contains exactly |nodes| instances.
5. All tests pass (or only the explicit Phase 0b `xfail` remains); all four
   cases are rerun, all ablation artifacts are archived, and every new solution
   completes handoff → scaffold → negative audit.

**Estimated effort**: Phase 0–2 require approximately 2–3 focused workdays
(test/fixture churn is the largest component), Phase 3 runs overnight in the
background, and Phase 4–5 take half a day. The only genuine **decision point is
A2** (the recommendation is to exempt infra edges from WS semantics, following
the same principle as Fix A; without the exemption, fwd gains a 12-cycle
mutual-exclusion window absent from the paper).
