# AGENTS.md — installed by `just-github-actions-n-workflows init`

> **Owned by the toolkit.** This file is generated and overwritten by `init` and `update` from the `docs/AGENTS.template.md` of the [just-github-actions-n-workflows](https://github.com/justAnArthur/just-github-actions-n-workflows) toolkit. To customize agent-facing instructions for this specific repo, create a separate `CONTRIBUTING.md` or repo-level instructions — do NOT hand-edit this file.

## TL;DR

This repo's release machinery is wired by workflows from `justAnArthur/just-github-actions-n-workflows`. As an AI agent you need three things to be useful here:

1. **Use Conventional Commits with a scope** — missing scope = silent no-bump.
2. **The scope maps to a package** via `properties.gitCommitScopeRelatedNames` in that package's `package.json`. Read it BEFORE editing.
3. **Tags carry JSON annotations** that control which publish/deploy workflows fire. Inspect with `git tag -l --format='%(contents)' <tag>` if anything looks skipped.

## 1. Repo layout after install

- `.github/workflows/*.yml` — workflows owned by the toolkit. Do not hand-edit.
- `.github/workflows/.toolkit-lock.json` — tracks installed `{name, file, ref, installedAt}`. `update` reads this; deleting it forces a full re-install.
- `.github/AGENTS.md` — this file (symlinked to `./AGENTS.md` at repo root).
- `.justactions.yml` — deploy config (only used by `deploy-docker-compose.yml` and `deploy-vercel-on-tag.yml`).

The `# toolkit-ref: <ref>` comment at the top of each workflow YAML is injected by `init`. The `uses: justAnArthur/just-github-actions-n-workflows/...@main` references inside the workflow bodies are rewritten to `@<sha>` at install time.

## 2. Commit format — the part you MUST get right

Conventional Commits. Scope is mandatory.

```
<type>(<scope>[,<scope>,...]): <subject>
```

Type → version bump:

| Type    | Bump  |
|---------|-------|
| `feat`  | minor |
| `fix`   | patch |
| `perf`  | minor |
| `feat!` | major |
| `BREAKING CHANGE:` in body | major |
| anything else | patch |

Multiple scopes (comma-separated) bump multiple packages at once. To suppress a bump entirely, include `[skip bump]` anywhere in the commit message.

## 3. Scope → package mapping

The mapping lives in each package's manifest, not in any single file. Example:

```jsonc
// packages/api/package.json
{
  "name": "@myorg/api",
  "properties": {
    "gitCommitScopeRelatedNames": "api,backend"
  }
}
```

Then `feat(api): ...` AND `feat(backend): ...` both bump `@myorg/api`. Match logic is in the toolkit: a scope matches if it equals the manifest `name` or any value in `gitCommitScopeRelatedNames` (see `lib/src/manifests/discovery.ts` and `lib/src/modules/index.ts`).

**Before editing code in this repo:** read every `package.json` (or `pom.xml`) under the path you'll touch. Check its `properties.gitCommitScopeRelatedNames`. Use one of those values as your commit scope.

If you use a scope that isn't in any manifest, your commit is **silently dropped** — no version bump occurs and no error surfaces.

## 4. Tags and deploy targets

When the `bump-version.yml` workflow runs (push to `main`, or via `workflow_dispatch`), it:

1. Parses commits since the last tag, finds the highest-bump-type per scope.
2. Updates `package.json` / `pom.xml` versions.
3. Commits and pushes the version bump as a `chore[skip bump]: bumping ...` commit.
4. Creates **annotated git tags** in the form `@scope/name@version` (e.g. `@myorg/api@0.2.0`).
5. The tag's annotation message is **JSON**:

   ```json
   {"deployTargets":["npm","docker"]}
   ```

   This JSON controls which publish/deploy workflows fire.
6. **Dispatches the tag workflows** for each new tag. The tags are pushed with the default `GITHUB_TOKEN`, and GitHub never starts workflows from pushes made with that token — so `bump-version.yml` itself starts every workflow in `.github/workflows/` whose `push.tags` filter matches the tag, via `workflow_dispatch` on the tag ref with `tag=<tag>` (step "Dispatch tag-triggered workflows"; needs `actions: write`, already in the file). No PAT is involved.

`deployTargets` is inferred from each manifest's properties:

| Property                              | Adds target |
|---------------------------------------|-------------|
| `private` not `true`                  | `npm`       |
| `DockerfilePath` (PascalCase) or `dockerfilePath` | `docker` |
| `vercelProjectId`                     | `vercel`    |
| `deployTargets` (explicit, comma-separated) | overrides the above |

## 5. Why a publish or deploy silently skipped

**No run at all for a new tag** → open the `bump-version` run that created it and check the "Dispatch tag-triggered workflows" step. It names every workflow it started and fails if a dispatch was rejected (most often: `actions: write` missing because a reusable-workflow caller didn't grant it). A tag workflow without a `workflow_dispatch` `tag` input is skipped with a warning. To recover, run the tag workflow from the Actions tab with the tag as `tag` input.

Each of `publish-npm-on-tag.yml`, `publish-docker-on-tag.yml`, `deploy-vercel-on-tag.yml` runs `resolve-tag-meta` first, then guards the main job with `if:`. The job skips when the tag annotation doesn't list its target.

To inspect a tag's annotation:

```sh
git fetch --tags
git tag -l --format='%(contents)' '@myorg/api@0.2.0'
```

- **Empty output or non-JSON** → legacy tag. Re-dispatch the workflow manually to fall back to runtime detection (`check-publishable` for npm, `get-dockerfile-path` for docker).
- **JSON without the target you want** → fix `properties.deployTargets` (or the inferred source property) and trigger a new bump.
- **JSON with the target, but workflow still skipped** → check `resolve-tag-meta.outputs.has_annotation` and the matching `publish_*` output in the workflow run logs.

## 6. Manual dispatch inputs

| Workflow                       | Inputs |
|--------------------------------|--------|
| `bump-version.yml`             | `bump_type` (e.g. `minor:canary`), `bump_to_calculated_stable`, `bump_manifest_names` (comma-separated), `bump_prerelease_channel` |
| `publish-npm-on-tag.yml`       | `tag` (e.g. `@scope/pkg@1.2.3`) |
| `publish-docker-on-tag.yml`    | `tag` |
| `deploy-vercel-on-tag.yml`     | `tag` |
| `release-on-tag.yml`           | `tag` |
| `deploy-docker-compose.yml`    | `environment`, `module_versions` (JSON), `ssh_target_path`, `compose_file` |

Re-dispatch is the right tool when a tag annotation is missing or the workflow errored non-deterministically.

## 7. Required secrets (set in repo Settings → Secrets → Actions)

| Secret | Used by |
|--------|---------|
| `NPM_TOKEN` | `publish-npm-on-tag.yml`; optional in `publish-docker-on-tag.yml` (docker build-arg) |
| `VERCEL_DEPLOY_HOOK_URL` | `deploy-vercel-on-tag.yml` |
| `SSH_PRIVATE_KEY`, `SERVER_USERNAME`, `DOCKER_USERNAME`, `DOCKER_PASSWORD` | `deploy-docker-compose.yml` |

All workflows use the auto-provided `GITHUB_TOKEN` — no PAT required. If you need a PAT with elevated scopes for a specific run, set `GH_TOKEN` in secrets and forward it via `workflow_call`.

## 8. Updating the toolkit

```sh
just-github-actions-n-workflows status    # show installed vs latest
just-github-actions-n-workflows update    # upgrade to latest
just-github-actions-n-workflows update --ref v1.2.3   # pin to version
```

All three commands accept `--cwd <path>` (added in CLI v1.0.2), so you can run the CLI from anywhere — `--cwd` points at this repo:

```sh
npx @justanarthur/just-github-actions-n-workflows-cli@1.0.2 update --cwd /path/to/this-repo --yes
```

If you can't run the CLI for some reason (e.g. CI), re-curl the workflow files manually from the new tag and re-commit. Since CLI v1.0.2, `npx` works from any directory — the lib dep is rewritten at publish time, so no local clone is needed.

## 9. Don't edit the workflows or this file directly

`.github/workflows/*.yml` and `.github/AGENTS.md` are owned by the toolkit. Local edits get blown away on the next `update` or `init --force`. Adding `# local-edit:` or `# toolkit-ref:` comments does not protect them — the next refresh rewrites the whole file.

**If you need a different publish behavior, do NOT edit `publish-npm-on-tag.yml`.** Override at the package level instead:

| need | where to put it |
|------|-----------------|
| publish to a different npm registry (`npm.pkg.github.com`, private registry, etc.) | `publishConfig.registry` in the package's `package.json` |
| publish to a non-default dist-tag (`next`, `beta`, …) | `publishConfig.tag` in the package's `package.json` |
| per-package auth token (e.g. GH Packages PAT) | `.npmrc` in the package's directory |
| extra pre-publish steps (codegen, asset build) | the `Build` step in `publish-npm-on-tag.yml` already runs `bun run --silent build` — add a `build` script to `package.json`. don't insert new steps into the workflow |
| bump logic / tag format / which deploy targets fire | `properties` in `package.json` (`gitCommitScopeRelatedNames`, `deployTargets`, `dockerfilePath`, `vercelProjectId`) |
| deploy config (ssh target, compose profiles) | `.justactions.yml` at the repo root |

The relevant publish step in `publish-npm-on-tag.yml` is:

```yaml
- name: Publish to npm
  if: steps.gate.outputs.skip != 'true'
  working-directory: ${{ steps.pkg.outputs.dir }}
  run: bun publish -p --access public --tag ${{ needs.resolve-meta.outputs.npm_tag }}
  env:
    NPM_CONFIG_TOKEN: ${{ secrets.NPM_TOKEN }}
```

`bun publish` reads `publishConfig` and `.npmrc` from the package directory on every invocation. **That is the supported override surface.**

Things you must NOT do, even if they look like the right call:

- Replace the `bun publish` step with a `.npmrc` swap + second `bun publish` call to hit a second registry. The toolkit's design for dual-registry is to copy `publish-npm-on-tag.yml` to `publish-gh-packages.yml` in `.github/workflows/` (both fire on `**@*`); do not patch the upstream copy.
- Comment out or remove the `Create GitHub release` step from `publish-npm-on-tag.yml`. That release is bundled by design. If you don't want it, either accept it, or stop using `publish-npm-on-tag.yml` and write your own (then you own the maintenance — updates won't help you).
- Add `# toolkit-ref: <sha>` / `# local-edit:` banner comments to track your changes. These are deleted on next `update`. The lockfile at `.github/workflows/.toolkit-lock.json` already tracks the ref — read it.

If you genuinely need behavior the toolkit doesn't support (e.g. you need to *skip* the bundled release, or run a second publish that the existing dual-workflow pattern can't express), open an issue upstream with a concrete use case. Don't patch locally and ship.

Adjusting the workflow's `push.branches` / `push.tags` pattern after install is expected and survives updates. Keep `**` before the `@` in a tag pattern: GitHub's `*` doesn't match `/`, so `*@*` never fires for a scoped tag like `@scope/name@1.2.3` — `init` does NOT customize triggers, so the user is responsible for them.

For repo-specific agent instructions, write a separate `CONTRIBUTING.md` or root-level `AGENTS.md` outside the toolkit's managed path.

## 10. Common agent mistakes in this repo

- `feat: add endpoint` — missing scope, silent no-bump.
- `feat(API): ...` — uppercase scope; manifests use lowercase by convention.
- Adding a new package without setting `properties.gitCommitScopeRelatedNames` — bumps for it will never fire.
- Pushing a tag like `v1.2.3` instead of `@scope/name@1.2.3` — none of the `**@*` triggers will match.
- Running `npm version patch` directly — bypasses the conventional-commit-driven bump and produces a tag without the JSON annotation, which downstream workflows then fall back to legacy detection for.
- Hand-editing `bump-version.yml` to change the bump logic — overwritten by `update`.
- Adding a `workflow_run` / `gh workflow run` workflow to "publish after bump" — `bump-version.yml` already dispatches the tag workflows, so this publishes twice. Delete such a workaround after updating.
- Copying a tag workflow (e.g. `publish-npm-on-tag.yml` → `publish-gh-packages.yml`) and removing its `workflow_dispatch` `tag` input — bump-version can then no longer start it.
- Hand-editing `publish-npm-on-tag.yml` to swap `.npmrc` for dual-registry, comment out the release step, or hard-code a registry — overwritten by `update`. Use `publishConfig` / `.npmrc` / `.justactions.yml` instead (see section 9).
- Adding `# toolkit-ref:` / `# local-edit:` comments to track manual workflow edits — deleted on next `update`. The lockfile already tracks the ref.
