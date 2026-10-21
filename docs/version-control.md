# Owner-controlled private repository

This workspace is already a local Git repository on branch `main`, with no
commits yet. The challenge is a pinned public submodule. No Git command runs on
the GPU host; no agent performs Git network/remote-management actions here.

Create an **empty private repository** in your Git hosting provider's UI. Do not
initialize it with a README, license, or gitignore, because those files already
exist locally. Then review the staged and unstaged changes on this machine:

```sh
git status --short
git diff
git diff --cached
git check-ignore -v .agents/skills/emmy-private-runtime/SKILL.md
```

Choose which files to retain and stage them yourself. Private runtime skills,
compiler prototypes and generated artifacts are ignored; do not force-add them.
After reviewing, run the following yourself (replace the URL):

```sh
git add -A
git diff --cached --check
git diff --cached
git commit -m "Add baseline compiler and local CUDA evaluator"
git remote add origin git@github.com:YOUR_ACCOUNT/YOUR_PRIVATE_REPOSITORY.git
git push -u origin main
```

The push publishes the selected parent-repository files and the submodule
reference. It does not make the public challenge submodule private. Do not
change the parent repository to public while it contains unpublished work.
Future clones need `git submodule update --init --recursive`, run by you.

No agent has committed, added a remote, fetched, pulled, or pushed this work.
