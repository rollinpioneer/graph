param([Parameter(Mandatory=$true)][string]$Phase)
$ErrorActionPreference='Stop'
$cardRoot=$PSScriptRoot
$bridgeGit='C:\Users\jackx\Documents\ChatGPT\CP-DISR\reports\git_publication_20261010\git_bridge.git'
$branch='codex/cp-disr-local-reuse-screen-v1'
$archiveRel='reports/local_reuse_screen_v1/'+(Split-Path -Leaf $cardRoot)
$lastFile=Join-Path $cardRoot 'last_publication_commit.txt'
$parentSha=if(Test-Path -LiteralPath $lastFile){(Get-Content -LiteralPath $lastFile -Raw).Trim()}else{'cc7d79d6fcb053bf0c16033e765a2e808de2aaac'}
$env:GIT_NO_LAZY_FETCH='1'
$env:GIT_INDEX_FILE=Join-Path $cardRoot ('publish_'+$Phase+'_'+[Guid]::NewGuid().ToString('N')+'.index')
if(Test-Path -LiteralPath $env:GIT_INDEX_FILE){throw 'Publication index already exists'}
$include=@('experiment_card.md','screen.py','prototype.py','measure.py','timing.py','timing_registration.json','timing_components.csv','search_screen.py','manifest.json','parents.json','prepare_receipt.json','identity_before.json','prepare.log','method_correspondence.md','dependency_audit.md','affected_work.csv','group_work.csv','G1.json','audit_receipt.json','audit.log','phase2_registration.json','correctness_report.md','correctness.json','correctness.log','correctness_receipt.json','G2.json','timing_results.csv','timing_summary.json','timing.log','timing_receipt.json','G3.json','search_results.csv','trajectory_difference_report.md','research_decision.md','final_receipt.json','registration.json','.gitattributes')
$payload=@{}
$include+=@('publish_card.ps1','REPRODUCE.md','nearest_source_identity.json','affected_summary.csv','integrity_after.json')
foreach($name in $include){
  $path=Join-Path $cardRoot $name
  if(Test-Path -LiteralPath $path){$payload[$name]=@{bytes=(Get-Item -LiteralPath $path).Length;sha256=(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}}
}
$manifest=@{phase=$Phase;publication_branch=$branch;parent_commit=$parentSha;files=$payload;count=$payload.Count;new_experiment_computation_during_publication=0}
$manifestPath=Join-Path $cardRoot 'archive_manifest.json'
[IO.File]::WriteAllText($manifestPath,($manifest|ConvertTo-Json -Depth 8).Replace("`r`n","`n")+"`n",[Text.UTF8Encoding]::new($false))
git --git-dir=$bridgeGit read-tree $parentSha
if($LASTEXITCODE-ne0){throw 'read-tree failed'}
foreach($name in (@($payload.Keys)+@('archive_manifest.json'))){
  $path=Join-Path $cardRoot $name
  $blob=git --git-dir=$bridgeGit hash-object -w --no-filters -- $path
  if($LASTEXITCODE-ne0){throw 'hash-object failed'}
  git --git-dir=$bridgeGit update-index --add --cacheinfo "100644,$blob,$archiveRel/$name"
  if($LASTEXITCODE-ne0){throw 'update-index failed'}
}
$paths=@(git --git-dir=$bridgeGit diff --cached --name-only $parentSha)
if($LASTEXITCODE-ne0 -or @($paths|Where-Object{!$_.StartsWith($archiveRel+'/')}).Count-gt0){throw 'Unexpected non-card change'}
git --git-dir=$bridgeGit -c core.whitespace=cr-at-eol diff --cached --check $parentSha -- '.' ":(exclude)$archiveRel/experiment_card.md"
if($LASTEXITCODE-ne0){throw 'Whitespace check failed'}
$tree=git --git-dir=$bridgeGit write-tree --missing-ok
if($LASTEXITCODE-ne0){throw 'write-tree failed'}
$bodyPath=Join-Path $cardRoot ('commit_body_'+$Phase+'.txt')
[IO.File]::WriteAllText($bodyPath,"LOCAL-REUSE-SCREEN-V1: $Phase`n",[Text.UTF8Encoding]::new($false))
$newSha=git --git-dir=$bridgeGit -c user.name=rollinpioneer -c user.email=1287282018@qq.com commit-tree $tree -p $parentSha -F $bodyPath
if($LASTEXITCODE-ne0){throw 'commit-tree failed'}
$oldRef=if(Test-Path -LiteralPath $lastFile){$parentSha}else{'0000000000000000000000000000000000000000'}
git --git-dir=$bridgeGit update-ref "refs/heads/$branch" $newSha $oldRef
if($LASTEXITCODE-ne0){throw 'update-ref failed'}
foreach($name in (@($payload.Keys)+@('archive_manifest.json'))){
  $expected=git --git-dir=$bridgeGit hash-object --no-filters -- (Join-Path $cardRoot $name)
  $stored=git --git-dir=$bridgeGit rev-parse "${newSha}:$archiveRel/$name"
  if($LASTEXITCODE-ne0 -or $expected-ne$stored){throw 'Published blob differs'}
}
Remove-Item Env:\GIT_INDEX_FILE
Remove-Item Env:\GIT_NO_LAZY_FETCH
git --git-dir=$bridgeGit -c http.sslverify=true -c credential.username=rollinpioneer -c credential.interactive=false push origin "refs/heads/${branch}:refs/heads/$branch"
if($LASTEXITCODE-ne0){throw 'push failed'}
$remote=git --git-dir=$bridgeGit -c http.sslverify=true -c credential.username=rollinpioneer -c credential.interactive=false ls-remote --heads origin "refs/heads/$branch"
if($LASTEXITCODE-ne0 -or ($remote-split'\s+')[0]-ne$newSha){throw 'Remote SHA verification failed'}
[IO.File]::WriteAllText($lastFile,$newSha+"`n",[Text.UTF8Encoding]::new($false))
$receipt=@{phase=$Phase;status='PUSHED_VERIFIED';commit=$newSha;parent=$parentSha;branch=$branch;archive_path=$archiveRel;payload_files=$payload.Count;verified_payload_blobs=$payload.Count+1;updated_utc=[DateTime]::UtcNow.ToString('o')}
[IO.File]::WriteAllText((Join-Path $cardRoot ('publication_'+$Phase+'.json')),($receipt|ConvertTo-Json).Replace("`r`n","`n")+"`n",[Text.UTF8Encoding]::new($false))
Write-Output ($receipt|ConvertTo-Json -Compress)
