<#
Commit the v4 tree into an existing repository.

The token is read from the GITHUB_TOKEN environment variable. It is passed to
git through an askpass helper for the duration of the push only, so it is not
written into .git/config, not placed on a command line, and not echoed.

Self-push, from the unpacked bundle directory:

    $env:GITHUB_TOKEN = "<paste here, in your own terminal>"
    powershell -File scripts/push_to_github.ps1 -RepoUrl https://github.com/OWNER/REPO.git

Preview without committing:

    powershell -File scripts/push_to_github.ps1 -RepoUrl ... -DryRun

With the GitHub CLI already authenticated, no token is needed:

    gh auth setup-git
    powershell -File scripts/push_to_github.ps1 -RepoUrl ... -UseGitCredentialHelper
#>
param(
  [Parameter(Mandatory = $true)][string]$RepoUrl,
  [string]$SourceDir = ".",
  [string]$Branch = "",
  [string]$Message = "v4 rebuild: evidence-based genomics, corrected GEM, gastric bioenergetics, calibrated kinetics, competition layer, interface",
  [switch]$UseGitCredentialHelper,
  [switch]$DryRun,
  [string]$WorkDir = ""
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if (-not $UseGitCredentialHelper -and -not $env:GITHUB_TOKEN) {
  throw "Set `$env:GITHUB_TOKEN, or pass -UseGitCredentialHelper if git credentials are already configured."
}

$src = (Resolve-Path $SourceDir).Path
# Default the scratch directory next to the current location rather than the
# system temp dir: TEMP is not writable under some sandboxed/AppContainer
# shells, and the failure there is a bare "Permission denied" from git.
if (-not $WorkDir) { $WorkDir = Join-Path (Get-Location).Path ".pushwork" }
$work = Join-Path $WorkDir ("repo-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
$repo = Join-Path $work "repo"
New-Item -ItemType Directory -Force -Path $work | Out-Null
$savedAskpass = $env:GIT_ASKPASS

try {
  if (-not $UseGitCredentialHelper) {
    # git calls this helper and reads the answer from stdout; the value comes
    # from the environment, so it never appears as an argument anywhere.
    $helper = Join-Path $work "askpass.ps1"
    $lines = @(
      'param([string]$prompt)',
      'if ($prompt -match "[Uu]sername") { Write-Output "x-access-token" }',
      'else { Write-Output $env:GITHUB_TOKEN }'
    )
    Set-Content -Path $helper -Value $lines -Encoding ascii
    $shim = Join-Path $work "askpass.cmd"
    Set-Content -Path $shim -Encoding ascii `
      -Value ('@powershell -NoProfile -ExecutionPolicy Bypass -File "' + $helper + '" %*')
    $env:GIT_ASKPASS = $shim
    $env:GIT_TERMINAL_PROMPT = "0"
  }

  Write-Host "cloning $RepoUrl"
  git clone --quiet $RepoUrl $repo
  if ($LASTEXITCODE -ne 0) { throw "clone failed (check the URL and that the token has repo scope)" }

  Push-Location $repo
  if (-not $Branch) { $Branch = (git symbolic-ref --short HEAD).Trim() }
  git checkout --quiet $Branch
  $before = @(git ls-files)
  Write-Host ("default branch: {0}; tracked files already present: {1}" -f $Branch, $before.Count)

  # Copy the v4 tree in. Paths that already exist are updated in place; files
  # the v4 work does not touch are left exactly as they are.
  $exclDirs = @(".git", "bundle", "__pycache__", "logs", "articles",
                (Join-Path $src "data\genomes"), (Join-Path $src "data\geo"),
                (Join-Path $src "data\blast"), (Join-Path $src "data\blastwork"))
  robocopy $src $repo /E /XD $exclDirs /XF "*.zip" /NFL /NDL /NJH /NJS /NP | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "robocopy failed with code $LASTEXITCODE" }

  git add -A
  $status = @(git status --porcelain)
  if ($status.Count -eq 0) {
    Write-Host "nothing to commit - the repository already matches this tree"
    return
  }

  $added    = @($status | Where-Object { $_ -match '^A' }).Count
  $modified = @($status | Where-Object { $_ -match '^M' }).Count
  $deleted  = @($status | Where-Object { $_ -match '^D' }).Count
  Write-Host ""
  Write-Host ("added {0}, modified {1}, deleted {2}" -f $added, $modified, $deleted)

  if ($modified -gt 0) {
    Write-Host "your existing files that this commit REPLACES:" -ForegroundColor Yellow
    $status | Where-Object { $_ -match '^M' } | ForEach-Object { "  " + $_.Substring(3) }
  }
  if ($deleted -gt 0) {
    Write-Host "reported as deleted - review before continuing:" -ForegroundColor Red
    $status | Where-Object { $_ -match '^D' } | ForEach-Object { "  " + $_.Substring(3) }
  }
  Write-Host ""
  git diff --cached --stat | Select-Object -Last 45

  if ($DryRun) {
    Write-Host ""
    Write-Host "-DryRun set: nothing was committed or pushed."
    return
  }

  git -c user.name="cncm-i745-digital-twin" `
      -c user.email="noreply@users.noreply.github.com" `
      commit --quiet -m $Message
  git push --quiet origin $Branch
  if ($LASTEXITCODE -ne 0) { throw "push failed" }
  Write-Host ("pushed {0} to {1} ({2})" -f (git rev-parse --short HEAD), $Branch, $RepoUrl)
}
finally {
  Pop-Location -ErrorAction SilentlyContinue
  if ($savedAskpass) { $env:GIT_ASKPASS = $savedAskpass }
  else { Remove-Item Env:\GIT_ASKPASS -ErrorAction SilentlyContinue }
  Remove-Item Env:\GIT_TERMINAL_PROMPT -ErrorAction SilentlyContinue
  if (Test-Path $work) { Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue }
}
