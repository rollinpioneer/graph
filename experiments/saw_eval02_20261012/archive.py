"""Explicit lightweight archive, final guards and parent preservation verification."""
import subprocess
from eval_core import *
repo=ROOT.parent.parent
branch=subprocess.check_output(['git','-C',str(repo),'branch','--show-current'],text=True).strip()
assert branch=='codex/saw-eval-02-20261012'
for k,h in EXPECTED.items():assert sha(OLD/'runs'/f'formal_{k}'/'params_1000000.pkl')==h
old_manifest=json.load(open(OLD/'LIGHT_MANIFEST.json'))
for x in old_manifest['files']:
    p=OLD/x['relative'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
assert subprocess.check_output(['git','-C',str(OLD/'vendor/saw'),'rev-parse','HEAD'],text=True).strip()=='3535a16c901e5de1233ac5a586401b3ddfa5e30d'
assert not subprocess.check_output(['git','-C',str(OLD/'vendor/saw'),'status','--porcelain','--untracked-files=no'],text=True).strip()
raw=json.load(open(ROOT/'EVIDENCE_INDEX.json'))
for x in raw['raw_files']:
    p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
assert json.load(open(ROOT/'results/A_AUDIT.json'))['status']=='PASS'
assert all(v==0 for v in json.load(open(ROOT/'receipts/EXIT_CODES.json'))['exits'].values())
assert json.load(open(ROOT/'results/B_SUMMARY.json'))['total_formal_episodes']==2000
assert len(list((ROOT/'results').glob('A_steps_*.csv')))==50
for k in ('navigate','stitch'):
    assert json.load(open(ROOT/'receipts'/f'B_{k}_DONE.json'))['formal_episodes']==1000
    for c in ('O','U'):
        for s in (55001,65001):
            rs=list(csv.DictReader(open(ROOT/'results'/f'B_{k}_{c}_{s}.csv')))
            assert len(rs)==250 and all(int(r['steps'])<=1000 for r in rs)
assert time.time()<DEADLINE
write(ROOT/'receipts/FINAL_VALIDATION.json',dict(status='PASS',parent_lightweight_files_unchanged=len(old_manifest['files']),
       readonly_weight_hashes_unchanged=True,official_tracked_source_unchanged=True,raw_identities_verified=len(raw['raw_files']),
       A_cases=50,B_episodes=2000,all_exit_codes=0,gpu_seconds=0,training_updates=0,
       figure_visual_review='PASS: protocol score figure inspected',statistics_note='Primary frozen-cell bootstrap unchanged; supplementary score bounds explicitly labeled. No retrospective seed/size changes.'))
files=[]
for p in ROOT.iterdir():
    if p.is_file() and (p.suffix in ('.py','.md','.json') or p.name in ('.gitignore','.gitattributes','requirements.lock.txt')) and p.name!='LIGHT_MANIFEST.json':files.append(p)
for d in ('results','receipts'):
    files.extend(p for p in (ROOT/d).iterdir() if p.is_file() and p.suffix in ('.csv','.json','.png'))
for p in files:
    assert p.stat().st_size<1048576
    if p.suffix=='.csv':p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    if p.suffix=='.json':json.loads(p.read_text(),parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
write(ROOT/'LIGHT_MANIFEST.json',dict(files=[dict(relative=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)],
       note='Excludes itself and post-push local receipt. Raw weights/scenes/trajectories/logs excluded.'))
files.append(ROOT/'LIGHT_MANIFEST.json');staged=[str(p.relative_to(repo)) for p in sorted(files)]
subprocess.run(['git','-C',str(repo),'add','--',*staged],check=True)
actual=set(subprocess.check_output(['git','-C',str(repo),'diff','--cached','--name-only'],text=True).splitlines())
assert actual<=set(staged)
subprocess.run(['git','-C',str(repo),'diff','--cached','--check'],check=True)
print('ARCHIVE_PASS',len(files),sum(p.stat().st_size for p in files),flush=True)
