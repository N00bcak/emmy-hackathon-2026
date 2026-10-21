# Owner-controlled private repository

This workspace is a local Git repository with the challenge pinned as a public
submodule. The requested implementation is committed locally after verification.
No Git commands run on the GPU host. Agents perform no Git network or remote
management actions here; the owner controls publication.

Create an **empty private repository** in GitHub’s UI if you do not already have
one. Leave README, license and gitignore initialization off. Review locally:

```sh
git status --short
git log -1 --stat
git check-ignore -v .agents/skills/emmy-private-runtime/SKILL.md
```

Private runtime instructions and generated artifacts remain ignored. Do not
force-add them. Then, **you** run the appropriate commands, replacing the URL:

```sh
# If origin has not been configured:
git remote add origin git@github.com:YOUR_ACCOUNT/YOUR_PRIVATE_REPOSITORY.git
# Or, if origin already exists and you intend to replace it:
# git remote set-url origin git@github.com:YOUR_ACCOUNT/YOUR_PRIVATE_REPOSITORY.git
git push -u origin main
```

The push publishes committed parent-repository files and the submodule reference;
it does not make the challenge’s public submodule private. Keep the parent private
while it contains unpublished work. Future clones need
`git submodule update --init --recursive`, run by you.

A local commit is authorized only when requested. It never authorizes an agent to
push, fetch, manage remotes, or publish. No agent networked Git action was needed
for this implementation.
