# Release Bootstrap Checklist

This document describes the one-time manual setup required to enable the tag-triggered release workflow for this project.

## Prerequisites

Before completing this checklist, ensure:
- The GitHub repository exists and is public
- `main` branch has been pushed
- You have admin access to the GitHub repository
- You have access to PyPI (as the package owner or maintainer)

## Setup Steps

### 1. Create and Push the GitHub Repository

- [ ] Create a new public GitHub repository under the org (e.g., `InfiniteRoomLabs/claudesync-index`)
- [ ] Push the `main` branch to GitHub
- [ ] Verify the repository is accessible at the expected URL

### 2. Verify Package Name on PyPI

- [ ] Visit https://pypi.org and search for `claudesync-index`
- [ ] If the package name is unclaimed, proceed to step 3
- [ ] If the package name is already claimed, **STOP** — the spec requires a decision on renaming before proceeding

### 3. Configure PyPI Trusted Publishing

- [ ] Log in to https://pypi.org with your account
- [ ] Navigate to **Account Settings** → **Publishing**
- [ ] Click **Add a new pending publisher**
- [ ] Configure:
  - **PyPI Project Name**: `claudesync-index`
  - **GitHub Repository Name**: `InfiniteRoomLabs/claudesync-index`
  - **Workflow Name**: `release.yml`
  - **Environment Name**: `pypi`
- [ ] Save and verify the trusted publisher entry appears in the list

### 4. Create the `pypi` Environment on GitHub

- [ ] Go to your GitHub repository → **Settings** → **Environments**
- [ ] Click **New environment**
- [ ] Name it `pypi`
- [ ] (Optional) Configure deployment protection rules if desired (e.g., require specific branch)
- [ ] Save the environment

### 5. Verify Organization Package Settings

- [ ] Go to your organization settings (if applicable) or your GitHub personal account settings
- [ ] Navigate to **Package Settings** or similar (varies by account type)
- [ ] Confirm that **Actions** is permitted to publish packages to **Container registry (GHCR)**
- [ ] Ensure no registry-blocking policies are in place

### 6. Create Internal Pull-Mirror (After First Release)

After you publish the first release and confirm it appears on PyPI and GHCR, create an internal pull-mirror of the GitHub repository:

- [ ] Document the mirror location and setup (this is infrastructure-specific; refer to your internal pull-mirror procedures)
- [ ] Verify that commits from the public GitHub repository synchronize to the internal mirror

## Testing the Release Workflow

Once all steps are complete:

1. **Create a test tag locally**:
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

2. **Monitor the workflow**:
   - Go to GitHub repository → **Actions**
   - Locate the `release` workflow run
   - Verify all jobs pass (`check`, `test`, `pypi`, `docker`)

3. **Verify outputs**:
   - Check PyPI: https://pypi.org/project/claudesync-index/
   - Check GHCR: `ghcr.io/infiniteroomlabs/claudesync-index`
   - Confirm both `0.1.0` and `latest` tags are present (GHCR tags strip the `v` prefix)

## Troubleshooting

### PyPI Publish Fails

- **Issue**: "Publisher not recognized" or "Authentication failed"
- **Fix**: Verify the trusted publisher settings on PyPI match the GitHub repo name, workflow file name, and environment name exactly

### GHCR Push Fails

- **Issue**: "Unauthorized" or "Permission denied"
- **Fix**: Confirm that the organization (or personal account) allows Actions to push to GHCR

### Tag Validation Fails

- **Issue**: "tag v0.1.0 != pyproject 0.1.0"
- **Fix**: Ensure the version in `pyproject.toml` matches the tag (without the `v` prefix). Update `pyproject.toml` and retag if needed

## See Also

- [Workflow Definition](.github/workflows/release.yml)
- [PyPI Trusted Publishing Docs](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions Environments](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment)
