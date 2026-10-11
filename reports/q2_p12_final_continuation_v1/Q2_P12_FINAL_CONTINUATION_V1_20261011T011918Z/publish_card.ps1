param([Parameter(Mandatory=$true)][ValidatePattern('^[a-z][a-z0-9_-]{0,31}$')][string]$Phase)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$gitDir='C:\Users\jackx\Documents\ChatGPT\CP-DISR\reports\git_publication_20261010\git_bridge.git'
$branch='codex/cp-disr-q2-p12-final-continuation-v1'
$relative='reports/q2_p12_final_continuation_v1/'+(Split-Path -Leaf $root)
$last=Join-Path $root 'last_publication_commit.txt'
$parent=if(Test-Path -LiteralPath $last){(Get-Content -LiteralPath $last -Raw).Trim()}else{'fabfa1341b057a4c2b67c0b3acb88e35c43b45ac'}
$registration=Get-Content -LiteralPath (Join-Path $root 'preregistration.json') -Raw -Encoding UTF8 | ConvertFrom-Json
if($registration.freeze_status-ne'FROZEN_BEFORE_FORMAL_RUN'){throw 'Registration and code/resource identities must be frozen before publication'}
# Explicitly reviewed plain-text files only. No raw gzip/binary or recursive collection.
$include=@(
 'experiment_card.md','preregistration.json','input_manifest.json','freeze_notes.md','publish_card.ps1',
 'core_semantics.py','worker.py','orchestrator.py','analyze.py','log_codec.py','log_codec_selftest.py','identity.py',
 'implementation_diff.patch','implementation_change_inventory.json','REPRODUCE.md','checkpoint_extraction.json',
 'legacy_suffix_reference.json','observed_initial.json','log_codec_selftest.json','syntax_check.json','orchestrator_selftest.json','analyzer_selftest.json','local_freeze_receipt.json',
 'preflight_identity.json','identity_before.json','identity_after.json','resource_preflight.json',
 'checkpoints/A_108.json','checkpoints/B_108.json','checkpoints/B_repeat_108.json','checkpoints/A_repeat_108.json',
 'branch_results.json','branch_ledger.jsonl','run_receipt.json','results.csv','event_results.json',
 'repeatability_checks.json','repeatability_report.md','coalescence_checks.json','coalescence_report.md',
 'score_ledger_verification.json','score_consistency.json','legacy_suffix_checks.json','startup_suffix_verify.json','score_ledger_checks.json','resource_accounting.json','runner_receipt.json',
 'research_decision.md','analysis_receipt.json','final_receipt.json','artifact_manifest.json','resource_summary.json',
 'server_final_check.json','local_verification.json','transfer_verification.json','ssh_cleanup.json',
 'branches/A/receipt.json','branches/B/receipt.json','branches/B_repeat/receipt.json','branches/A_repeat/receipt.json',
 'plans/A.json','plans/B.json','plans/B_repeat.json','plans/A_repeat.json'
)
$files=@(foreach($name in $include){$path=Join-Path $root $name;if(Test-Path -LiteralPath $path){Get-Item -LiteralPath $path}})
$utf8Strict=[Text.UTF8Encoding]::new($false,$true)
$payload=@{}
$totalBytes=0L
foreach($f in $files){
 if($f.Length-gt16MB){throw ('Whitelisted text file unexpectedly exceeds 16 MiB: '+$f.Name)}
 $text=$utf8Strict.GetString([IO.File]::ReadAllBytes($f.FullName))
 if($text.Contains([string][char]0)){throw ('Binary/NUL content in publication file: '+$f.Name)}
 $totalBytes+=$f.Length
 $rel=[IO.Path]::GetRelativePath($root,$f.FullName).Replace('\','/')
 $payload[$rel]=@{bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
if($totalBytes-gt128MB){throw 'Publication text unexpectedly exceeds 128 MiB; inspect before publishing'}
$manifestPath=Join-Path $root 'archive_manifest.json'
[IO.File]::WriteAllText($manifestPath,(@{phase=$Phase;branch=$branch;parent=$parent;files=$payload;payload_count=$payload.Count;plain_bytes=$totalBytes;binary_evidence='Stored separately with actual byte-hash indices'}|ConvertTo-Json -Depth 10).Replace("`r`n","`n")+"`n",[Text.UTF8Encoding]::new($false))
$env:GIT_NO_LAZY_FETCH='1'
$env:GIT_INDEX_FILE=Join-Path $root ('publication_'+[Guid]::NewGuid().ToString('N')+'.index')
try {
 git --git-dir=$gitDir read-tree $parent
 if($LASTEXITCODE-ne0){throw 'read-tree failed'}
 foreach($f in (@($files.FullName)+@($manifestPath))){
  $rel=[IO.Path]::GetRelativePath($root,$f).Replace('\','/')
  $blob=git --git-dir=$gitDir hash-object -w --no-filters -- $f
  if($LASTEXITCODE-ne0){throw 'hash-object failed'}
  git --git-dir=$gitDir update-index --add --cacheinfo "100644,$blob,$relative/$rel"
  if($LASTEXITCODE-ne0){throw 'update-index failed'}
 }
 $changed=@(git --git-dir=$gitDir diff --cached --name-only $parent)
 if($LASTEXITCODE-ne0 -or @($changed|Where-Object{!$_.StartsWith($relative+'/')}).Count-gt0){throw 'Non-card file changed'}
 git --git-dir=$gitDir -c core.whitespace=cr-at-eol diff --cached --check $parent -- "$relative/core_semantics.py" "$relative/worker.py" "$relative/orchestrator.py" "$relative/analyze.py" "$relative/log_codec.py" "$relative/publish_card.ps1" "$relative/research_decision.md"
 if($LASTEXITCODE-ne0){throw 'Whitespace check failed'}
 $tree=git --git-dir=$gitDir write-tree --missing-ok
 if($LASTEXITCODE-ne0){throw 'write-tree failed'}
 $body=Join-Path $root ('commit_body_'+$Phase+'.txt')
 [IO.File]::WriteAllText($body,"Q2-P12-FINAL-CONTINUATION-V1: $Phase`n",[Text.UTF8Encoding]::new($false))
 $sha=git --git-dir=$gitDir -c user.name=rollinpioneer -c user.email=1287282018@qq.com commit-tree $tree -p $parent -F $body
 if($LASTEXITCODE-ne0){throw 'commit-tree failed'}
 $old=if(Test-Path -LiteralPath $last){$parent}else{'0000000000000000000000000000000000000000'}
 git --git-dir=$gitDir update-ref "refs/heads/$branch" $sha $old
 if($LASTEXITCODE-ne0){throw 'update-ref failed; do not overwrite an unexpected prior card branch'}
 foreach($f in (@($files.FullName)+@($manifestPath))){
  $rel=[IO.Path]::GetRelativePath($root,$f).Replace('\','/')
  $expect=git --git-dir=$gitDir hash-object --no-filters -- $f
  $stored=git --git-dir=$gitDir rev-parse "${sha}:$relative/$rel"
  if($LASTEXITCODE-ne0 -or $expect-ne$stored){throw 'Blob verification failed'}
 }
} finally {
 Remove-Item Env:\GIT_INDEX_FILE -ErrorAction SilentlyContinue
 Remove-Item Env:\GIT_NO_LAZY_FETCH -ErrorAction SilentlyContinue
}
git --git-dir=$gitDir -c http.sslverify=true -c credential.username=rollinpioneer -c credential.interactive=false push origin "refs/heads/${branch}:refs/heads/$branch"
if($LASTEXITCODE-ne0){throw 'push failed; preserve created commit for an authorized retry'}
$remote=git --git-dir=$gitDir -c http.sslverify=true -c credential.username=rollinpioneer -c credential.interactive=false ls-remote --heads origin "refs/heads/$branch"
if($LASTEXITCODE-ne0 -or ($remote-split'\s+')[0]-ne$sha){throw 'Remote verification failed'}
[IO.File]::WriteAllText($last,$sha+"`n",[Text.UTF8Encoding]::new($false))
$receipt=@{phase=$Phase;status='PUSHED_VERIFIED';commit=$sha;parent=$parent;branch=$branch;archive_path=$relative;payload_files=$payload.Count;plain_bytes=$totalBytes;updated_utc=[DateTime]::UtcNow.ToString('o')}
[IO.File]::WriteAllText((Join-Path $root ('publication_'+$Phase+'.json')),($receipt|ConvertTo-Json).Replace("`r`n","`n")+"`n",[Text.UTF8Encoding]::new($false))
Write-Output ($receipt|ConvertTo-Json -Compress)
