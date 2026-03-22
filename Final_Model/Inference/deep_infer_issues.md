# Issues Identified in `deep_infer.ipynb`

## Summary
The notebook has significant structural and code issues, particularly in the Explainable AI cells (Cells 2-4) which appear to have been copied from another notebook without proper adaptation.

---

## Critical Issues

### 1. Undefined Variables in Explainability Cells

The following variables are used but **never defined**:

| Variable | Used In | Description |
|----------|---------|-------------|
| `USE_LLM_EXPLAINER` | Cell 3 (`load_explainer()`) | Boolean flag |
| `EXPLAINER_MODEL` | Cell 3 | Model path string |
| `EXPLAINER_FALLBACK_MODEL` | Cell 3 | Fallback model path |
| `device` | Cell 3 | Should be `"cuda"` or `"cpu"` (defined locally in `main()`) |
| `train_df` | Cell 3 | DataFrame (actual variable is `train` in `main()`) |
| `KB_BUILD_TOP_N` | Cell 3 | Integer for KB building |
| `KB_PATH` | Cell 3 | Path object for caching KB |
| `KB_EXAMPLES_PER_MISCON` | Cell 3 | Integer for sampling |
| `USE_LLM_VERIFIER` | Cell 4 | Boolean flag |
| `correct_by_q_full` | Cell 4 | Dictionary mapping question IDs to correct answers |
| `kb` | Cell 4 | Knowledge base dictionary |

### 2. Missing Imports

```python
# These are needed but not imported
from collections import Counter  # Used in Cell 3
import json                       # Used in Cells 3 and 4
import re                         # Used in Cells 3 and 4
from pathlib import Path          # KB_PATH uses Path methods
```

### 3. Scope/Structure Problem

**Cell 1** wraps everything in `main()` function:
- Variables like `train`, `test`, `le`, `correct`, `device` are **local** to `main()`
- These variables are **not accessible** from Cells 3 and 4

**Cells 3-4** run at module level and expect:
- `train_df` (but it's called `train` and is inside `main()`)
- `device` (local to `main()`)
- Various configuration constants that don't exist

### 4. Column References That Don't Exist

In the demo section (Cell 4):
```python
row["pred_top3"]  # Never created
row["label_str"]  # Never created
```

---

## Medium Issues

### 5. Deprecation Warning
```
`torch_dtype` is deprecated! Use `dtype` instead!
```

In `AutoModelForSequenceClassification.from_pretrained()`:
- Change `torch_dtype=torch.float16` to `dtype=torch.float16`

### 6. Variable Name Mismatch
- Code uses `train_df` but data is loaded as `train`
- Inconsistent naming suggests copy-paste from another notebook

### 7. Explainer Model Not Released
- `explainer_model` loaded but never deleted/cleaned up
- Can cause GPU memory issues

---

## Minor Issues

### 8. Hardcoded Paths
- Kaggle-specific paths may fail in other environments
- No fallback or environment detection

### 9. Missing Error Handling
- `find_model_path()` has no error handling for permission issues
- Model loading has no try-except

---

## Proposed Fixes

### Fix 1: Add Missing Configuration Constants (Cell 1)

Add these at the top of Cell 1 with other configurations:
```python
# Explainable AI Configuration
USE_LLM_EXPLAINER = True
USE_LLM_VERIFIER = False
EXPLAINER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXPLAINER_FALLBACK_MODEL = "google/flan-t5-large"
KB_BUILD_TOP_N = 50
KB_EXAMPLES_PER_MISCON = 3
KB_PATH = Path("misconception_kb.json")
```

### Fix 2: Add Missing Imports (Cell 1)

Add at the top with other imports:
```python
import json
import re
from collections import Counter
from pathlib import Path
```

### Fix 3: Restructure to Share Variables

**Option A**: Remove `main()` wrapper, run at module level
- Simpler but less clean
- Variables become global

**Option B**: Move explainability code into `main()`
- Keeps encapsulation
- Preferred approach

**Option C**: Return needed variables from `main()` and capture them
```python
# At end of main():
return train, test, le, device, correct

# After main() definition:
train_df, test_df, le, device, correct_df = main()
```

### Fix 4: Fix Variable Name Consistency

Change all `train_df` references to `train` OR:
```python
train_df = train  # Add alias after loading
```

### Fix 5: Add Missing Columns for Demo

Before the demo section:
```python
# Create placeholder columns if running on training data
if 'pred_top3' not in train_df.columns:
    train_df['pred_top3'] = ''
if 'label_str' not in train_df.columns:
    train_df['label_str'] = train_df['target']  # or appropriate column
```

### Fix 6: Fix Deprecation Warning

Change:
```python
torch_dtype=torch.float16
```
To:
```python
dtype=torch.float16
```

### Fix 7: Add Cleanup for Explainer Model

At the end:
```python
if explainer_model is not None:
    del explainer_model
    del explainer_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

---

## Recommended Approach

The cleanest fix is to:

1. **Move explainability code inside `main()`** (after inference, before cleanup)
2. **Add all missing imports at the top**
3. **Add all missing configuration constants**
4. **Rename `train` to `train_df`** for consistency (or use alias)
5. **Create the `correct_by_q_full` dictionary** from the `correct` DataFrame
6. **Add proper cleanup** for all loaded models

This maintains the single-responsibility structure while fixing all dependency issues.
