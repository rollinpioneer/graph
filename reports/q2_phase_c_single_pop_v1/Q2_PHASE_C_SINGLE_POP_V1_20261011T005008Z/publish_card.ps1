param([Parameter(Mandatory=$true)][string]$Phase)
$ErrorActionPreference='Stop'
$root=$PSScriptRoot
$gitDir='C:\Users\jackx\Documents\ChatGPT\CP-DISR\reports\git_publication_20261010\git_bridge.git'
$branch='codex/cp-disr-q2-phase-c-single-pop-v1'
$relative='reports/q2_phase_c_single_pop_v1/'+(Split-Path -Leaf $root)
$last=Join-Path $root 'last_publication_commit.txt'
$parent=if(Test-Path -LiteralPath $last){(Get-Content -LiteralPath $last -Raw).Trim()}else{'f3d8d655d172b338bb38ba7f638aa0fda355dc55'}
$env:GIT_NO_LAZY_FETCH='1'
$env:GIT_INDEX_FILE=Join-Path $root ('publication_'+[Guid]::NewGuid().ToString('N')+'.index')
# Explicitly reviewed publication list. Original evidence and all binary data stay on disk.
$include=@('experiment_card.md','phase_c.py','analyze.py','publish_card.ps1','preregistration.json','input_manifest.json','implementation_diff.patch','REPRODUCE.md','identity_before.json','identity_after.json','branch_results.json','run_receipt.json','repeatability_checks.json','repeatability_report.md','coalescence_checks.json','coalescence_report.md','score_ledger_verification.json','event_results.json','results.csv','research_decision.md','analysis_receipt.json','final_receipt.json','artifact_manifest.json','server_final_check.json','transfer_verification.json','plans/ipc_p12_k002_A.json','plans/ipc_p12_k002_B.json','plans/ipc_p12_k002_A_repeat.json','plans/ipc_p12_k002_B_repeat.json','plans/struct_n4_004_k008_A.json','plans/struct_n4_004_k008_B.json','plans/struct_n4_004_k008_A_repeat.json','plans/struct_n4_004_k008_B_repeat.json')
$files=@(foreach($name in $include){$path=Join-Path $root $name;if(Test-Path -LiteralPath $path){Get-Item -LiteralPath $path}})
$payload=@{}
foreach($f in $files){$rel=[IO.Path]::GetRelativePath($root,$f.FullName).Replace('\','/');$payload[$rel]=@{bytes=$f.Length;sha256=(Get-FileHash -LiteralPath $f.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}}
$manifestPath=Join-Path $root 'archive_manifest.json'
[IO.File]::WriteAllText($manifestPath,(@{phase=$Phase;branch=$branch;parent=$parent;files=$payload;payload_count=$payload.Count}|ConvertTo-Json -Depth 10).Replace("`r`n","`n")+"`n",[Text.UTF8Encoding]::new($false))
git --git-dir=$gitDir read-tree $parent
if($LASTEXITCODE-ne0){throw 'read-tree failed'}
foreach($f in (@($files.FullName)+@($manifestPath))){$rel=[IO.Path]::GetRelativePath($root,$f).Replace('\','/');$blob=git --git-dir=$gitDir hash-object -w --no-filters -- $f;if($LASTEXITCODE-ne0){throw 'hash-object failed'};git --git-dir=$gitDir update-index --add --cacheinfo "100644,$blob,$relative/$rel";if($LASTEXITCODE-ne0){throw 'update-index failed'}}
$changed=@(git --git-dir=$gitDir diff --cached --name-only $parent)
if($LASTEXITCODE-ne0 -or @($changed|Where-Object{!$_.StartsWith($relative+'/')}).Count-gt0){throw 'Non-card file changed'}
# Old evidence bytes and pasted Markdown are preserved, including original hard breaks.
git --git-dir=$gitDir -c core.whitespace=cr-at-eol diff --cached --check $parent -- "$relative/phase_c.py" "$relative/analyze.py" "$relative/publish_card.ps1" "$relative/research_decision.md"
if($LASTEXITCODE-ne0){throw 'Whitespace check failed'}
$tree=git --git-dir=$gitDir write-tree --missing-ok
if($LASTEXITCODE-ne0){throw 'write-tree failed'}
$body=Join-Path $root ('commit_body_'+$Phase+'.txt')
[IO.File]::WriteAllText($body,"Q2-PHASE-C-SINGLE-POP-V1: $Phase`n",[Text.UTF8Encoding]::new($false))
$sha=git --git-dir=$gitDir -c user.name=rollinpioneer -c user.email=1287282018@qq.com commit-tree $tree -p $parent -F $body
if($LASTEXITCODE-ne0){throw 'commit-tree failed'}
$old=if(Test-Path -LiteralPath $last){$parent}else{'0000000000000000000000000000000000000000'}
git --git-dir=$gitDir update-ref "refs/heads/$branch" $sha $old
if($LASTEXITCODE-ne0){throw 'update-ref failed'}
foreach($f in (@($files.FullName)+@($manifestPath))){$rel=[IO.Path]::GetRelativePath($root,$f).Replace('\','/');$expect=git --git-dir=$gitDir hash-object --no-filters -- $f;$stored=git --git-dir=$gitDir rev-parse "${sha}:$relative/$rel";if($LASTEXITCODE-ne0 -or $expect-ne$stored){throw 'Blob verification failed'}}
Remove-Item Env:\GIT_INDEX_FILE
Remove-Item Env:\GIT_NO_LAZY_FETCH
git --git-dir=$gitDir -c http.sslverify=true -c credential.username=rollinpioneer -c credential.interactive=false push origin "refs/heads/${branch}:refs/heads/$branch"
if($LASTEXITCODE-ne0){throw 'push failed'}
$remote=git --git-dir=$gitDir -c http.sslverify=true -c credential.username=rollinpioneer -c credential.interactive=false ls-remote --heads origin "refs/heads/$branch"
if($LASTEXITCODE-ne0 -or ($remote-split'\s+')[0]-ne$sha){throw 'Remote verification failed'}
[IO.File]::WriteAllText($last,$sha+"`n",[Text.UTF8Encoding]::new($false))
$receipt=@{phase=$Phase;status='PUSHED_VERIFIED';commit=$sha;parent=$parent;branch=$branch;archive_path=$relative;payload_files=$payload.Count;updated_utc=[DateTime]::UtcNow.ToString('o')}
[IO.File]::WriteAllText((Join-Path $root ('publication_'+$Phase+'.json')),($receipt|ConvertTo-Json).Replace("`r`n","`n")+"`n",[Text.UTF8Encoding]::new($false))
Write-Output ($receipt|ConvertTo-Json -Compress)
