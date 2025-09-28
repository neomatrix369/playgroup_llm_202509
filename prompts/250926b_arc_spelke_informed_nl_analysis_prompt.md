### Prompt: Spelke-Informed ARC Natural Language Analysis

#### Purpose
Integrate Core Knowledge priors (per Spelke; as operationalized in ARC by Chollet) into a systematic, multi-representation analysis workflow for ARC problems. The goal is to generate precise, testable hypotheses that generalize, annotate them with core knowledge tags, implement the top candidates, and produce a concise HTML report.

#### Inputs
- `problem_id`: e.g., `08ed6ac7`
- `visualization_path`: absolute path to the problem visualization HTML/PNGs
- Access to training/test grids (list-of-lists / numpy arrays)

#### Outputs
1) Structured natural language descriptions for each training I/O pair
2) Ten hypotheses with Core Knowledge tags and explicit tie-breakers
3) Python implementations for the top 3 hypotheses
4) Verification against training examples and predictions for test input
5) HTML report including representation reflections and hypothesis annotations

---

### Phase 0: Core Knowledge Framing (Spelke → ARC)
Ground your reasoning in the following innate systems and their ARC-relevant manifestations. For each item, briefly note whether it likely applies to this problem and how.

- Objectness and elementary physics
  - Cohesion (contiguity of regions), persistence, contact-based interaction
  - Invariants: connected components, boundaries, fills, growth/shrink by contact
- Elementary geometry and topology
  - Relative position, adjacency, symmetry, reflection/rotation, connectivity, holes
  - Invariants: topological equivalence, symmetry conservation/breaking
- Place system and frames of reference
  - Allocentric vs egocentric frames, grid axes, scan orders (row/column)
  - Invariants: relative order, alignment, axis orientation, stable reference frames
- Natural numbers and elementary arithmetic
  - Counts, ranks, orderings, parity/periodicity, simple mappings, frequency
  - Invariants: conserved counts, proportionality, ordinal rules
- Agentness and goal-directedness
  - Typically not applicable in static ARC grids; consider only with clear cues
- Form and category recognition
  - Grouping by shared properties, prototypes/exemplars, category stability under transforms
  - Invariants: category membership, feature-based grouping consistency

Bias and tie-breaker policy (apply in order unless contradicted by data):
1. Prefer relational/topological rules over absolute coordinates when both fit.
2. Prefer object cohesion and connectedness-based reasoning over pixel-wise lookup.
3. Prefer minimal description length (simpler rule that fits all training pairs).
4. Use stable, data-supported tie-breakers: topmost, leftmost, earliest discovery.
5. Avoid arbitrary color lookups unless supported across all examples.

Deliverable for Phase 0 (short bullet list): which systems are implicated and why.

---

### Phase 1: Multi-Representation Analysis

1) Visual inspection
- Open `{visualization_path}` and scan training I/O pairs.
- Note salient structures: columns/rows, clusters, symmetries, regularities, boundaries.
 - Describe the visual gestalt; identify cohesive objects and boundaries; note frames of reference.

2) Natural language descriptions (for each training example N)
- Input:
  - Grid size (rows, cols)
  - Distinct regions/patterns with: position summary, size, shape/topology, color
- Output:
  - What changed: color, position, size/shape, count
  - Map input regions to output effects (where applicable)

3) List-of-lists / array excerpt
- Include compact excerpts or summary statistics that support precision.

---

### Phase 2: Pattern Recognition and Invariants
- Identify elements that remain constant across examples (counts, topology, alignments).
- Identify transformations (color mappings, motions, growth/shrink, re-labeling).
- For each observation, attach Core Knowledge tags: [Objectness], [Geometry], [Place], [Number], [Form].

---

### Phase 3: Hypothesis Generation (10 total)
Generate ten distinct, testable hypotheses ordered from most to least plausible.

For each hypothesis use this template:
```
{k}. {Hypothesis Name}
- CoreKnowledge: [Objectness?/Geometry?/Number?]
- Rule: Clear, step-by-step algorithm that fully explains all training I/O pairs
- Tie-breakers: Explicit, deterministic (e.g., topmost, leftmost, scan order)
- Edge cases: Empty regions, ties in size/height, overlapping effects
- Generalization note: Why this should transfer to novel layouts/colors
- Representation advantage: Which representation (visual / arrays / language) reveals this best
- Testable prediction: Specific prediction for the held-out test input
```

Suggested hypothesis families to consider (choose diverse ones):
- Connected component labeling and ordering (left→right, top→bottom, size/rank)
- Geometry/topology transforms (mirror, rotate, translate, dilate, fill, border)
- Numeric/ordinal schemes (parity, periodicity, rank-to-color, count-preserving)
- Discovery order effects (row/column scan, BFS/DFS traversal sequences)
- Global symmetry detection/extension vs local feature rules
- Form/category operations (group by shared features; prototype/exemplar-driven mappings)

---

### Phase 4: Select and Implement Top 3
Criteria: explains all training pairs, simplest consistent rule, strongest transfer.

Optionally include a brief implementation sketch in pseudocode before coding:
```
For each [relevant element]:
    If [condition based on core knowledge principle]:
        Apply [transformation]
    Track [what persists/changes]
Output [result]
```

Provide Python implementations (numpy preferred) with clear docstrings:
```python
import numpy as np

def hypothesis_one(initial: np.ndarray) -> np.ndarray:
    """
    CoreKnowledge: [Objectness, Geometry]
    Rule: ...
    Tie-breakers: ...
    """
    # Implementation
    return final
```

---

### Phase 5: Verification on Training
For each training example and hypothesis:
- Run function → compare with expected output
- Report: ✓ MATCH or ✗ MISMATCH with brief reason if mismatch

If multiple hypotheses match identically, note structural reasons (e.g., layout-induced equivalence of left→right and top-first orders) and how you would disambiguate on new examples.

---

### Phase 6: Apply to Test Input
- Run all implemented hypotheses on the test input
- Provide predicted output arrays; select the primary prediction with justification

---

### Phase 7: Report (HTML)
Include:
1) Prediction summary (chosen hypothesis, predicted test grid)
2) Links to original visualization and results directory
3) Representation reflection: which format helped for discovery, formation, implementation, verification
4) All natural language descriptions and invariants
5) All 10 hypotheses (mark the implemented ones ✅) with Core Knowledge tags
6) Full code for implemented hypotheses and training verification table

---

### Acceptance Criteria
1) Every training I/O pair is described precisely in words and supported by data
2) Ten hypotheses are distinct, tagged, tie-broken, and testable
3) Three implementations run and produce a verification table
4) Chosen hypothesis is justified with Core Knowledge reasoning
5) HTML report is complete, skimmable, and links assets with absolute paths when applicable

---

### Tips (Spelke-Informed)
- Look first for object cohesion and connectedness; ARC favors object-level rules.
- Favor relational/topological invariants over absolute coordinates.
- Use simple ordinal schemes (leftmost/topmost, largest/tallest) before complex ones.
- Treat symmetry/regularity as strong priors; test mirror/rotation when plausible.
- When color seems arbitrary, search for rank-to-color or position-to-color mappings supported across all training examples.

---

### Phase 8: Meta-Cognitive Reflection (optional but recommended)
- Pattern recognition strategy: Which representation did you gravitate toward first, and why?
- Cognitive load analysis: Which core systems must coordinate; working memory demands; human vs AI difficulty.
- Developmental perspective: Likely age/milestones for solving; learnability from few examples.
- AI challenge analysis: Why current AI might struggle; what data or inductive biases would help.
- Representation synergy: What visual revealed that arrays did not, and vice versa; how language bridged them.

---

### Example Usage (script outline)
```bash
# Analyze problem 08ed6ac7 and produce an HTML report
python analyze_arc_problem.py \
  --problem 08ed6ac7 \
  --visualization /ABS/PATH/TO/out/html/<ts>/index.html \
  --output_dir /ABS/PATH/TO/out/analysis/<ts>
```


