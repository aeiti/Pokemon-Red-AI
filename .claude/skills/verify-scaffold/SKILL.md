---
name: verify-scaffold
description: After staging or committing a multi-file scaffold (new package directory, new feature surface, or any batch of new files added at once), verify that every file you intended to add was actually tracked by git. .gitignore patterns can silently drop files without warning, and the resulting "missing __init__.py" or "missing module" failures surface much later — often during import or test runs on a fresh clone. Triggers when files appear to be added en masse, after a scaffolding/init-style commit, or whenever a previously-believed-tracked file turns out to be missing.
---

# Verify staged/committed files match what's on disk

Git's `add` silently respects `.gitignore`. If a pattern like `env/` is
unanchored, it will match `any/path/env/` — including a freshly-created
package directory you meant to commit. `git add` emits no warning. The
commit summary will show "7 files changed" when you expected 8. Nobody
notices until a later branch tries to put a module in the
silently-dropped directory and `git add` rejects it.

This is most likely when:
- You just created a new top-level package directory (e.g. `pkg/env/`,
  `pkg/build/`, `pkg/tests/`, `pkg/dist/`) whose name matches a
  conventional ignore pattern.
- You staged many files at once and didn't read the staged-file list.
- A teammate's `.gitignore` has unanchored patterns that worked fine
  for them but conflict with your new layout.

## Recipe

After staging or committing a scaffold, run **both** of these:

```sh
# 1. What files did git ACTUALLY just stage / commit?
git diff --cached --stat              # for staged but uncommitted
git show --stat HEAD                  # for the last commit

# 2. What files exist in the new directory(ies) on disk?
ls <dir>
find <dir> -type f -not -path '*/__pycache__/*'
```

Cross-reference. Any file on disk that isn't in git's list is a
silently-dropped file.

For a more targeted check after committing — does git track the file
you expect?

```sh
git ls-files <path>                   # empty = not tracked
git check-ignore -v <path>            # shows which gitignore rule
                                      # (if any) is filtering it
```

`git check-ignore` is the smoking gun: it tells you exactly which
pattern, on which line of which `.gitignore`, is ignoring the file.

## Fixing unanchored patterns

Once you find the culprit pattern, anchor it to the repo root with a
leading slash:

```diff
- env/        # matches any env/ directory anywhere
+ /env/       # matches only the top-level env/
```

Then re-stage the dropped file(s) in a **new commit** (per project
convention — never amend). A commit titled something like
`fix: anchor virtualenv ignore patterns to repo root` reads cleanly.

## When to invoke

- Right after a `chore: scaffold ...` or `feat: add package ...` style
  commit.
- Before merging a feature branch, when the diff looks suspiciously
  light on new files.
- Whenever someone reports "I can't import foo.bar even though
  bar/__init__.py exists on disk."
