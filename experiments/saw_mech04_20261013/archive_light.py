"""Explicit own-experiment allowlist; heavy artifacts stay remote with SHA index."""
import pathlib,json,hashlib,subprocess,tarfile,time
P=pathlib.Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    raw=[]
    for directory in ['runs','logs']:
        for p in sorted((P/directory).rglob('*')):
            if p.is_file():raw.append(dict(path=str(p),relative=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=sha(p)))
    (P/'RAW_ARTIFACT_INDEX.json').write_text(json.dumps(dict(server_root=str(P),files=raw),indent=2)+'\n')
    rootnames=['.gitignore','.gitattributes','requirements.lock','EXPERIMENT_CARD_FROZEN.md','REQUESTED_CARD.md','FROZEN_RUN_APPENDIX.md','R_AUDIT.md','T_FROZEN_PROTOCOL.md','T_INTERPRETATION.md','REPRODUCTION_REPORT.md','HYPOTHESIS_VERDICT.md','NEXT_DECISION.md','RAW_ARTIFACT_INDEX.json']
    files=[P/n for n in rootnames if (P/n).exists()]+sorted(P.glob('*.py'))
    for directory,extensions in [('results',{'.csv','.json'}),('receipts',{'.json'}),('figures',{'.png','.pdf'})]:
        files += [p for p in sorted((P/directory).glob('*')) if p.is_file() and p.suffix in extensions and not p.name.endswith('_active.json')]
    assert not any('/runs/' in str(p) or '/logs/' in str(p) for p in files)
    assert all(p.stat().st_size<10*1024**2 for p in files),'UNEXPECTED_LARGE_LIGHT_FILE'
    manifest=dict(files=[dict(relative=str(p.relative_to(P)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(set(files))],raw_index_sha256=sha(P/'RAW_ARTIFACT_INDEX.json'),note='Generated after all own experiment processes exited. Git publication/SSH exit receipt saved separately to avoid circular commit identity.')
    (P/'LIGHT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');files.append(P/'LIGHT_MANIFEST.json')
    subprocess.run(['git','add','--']+[str(p.relative_to(P)) for p in sorted(set(files))],cwd=P,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=P,check=True,stdout=subprocess.DEVNULL)
    staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=P,text=True).splitlines();prefix='experiments/saw_mech04_20261013/'
    assert all(x.startswith(prefix) for x in staged);assert not any('/runs/' in x or '/logs/' in x or '.pkl' in x or '.npz' in x or '.npy' in x for x in staged)
    print('LIGHT_ALLOWLIST_READY',len(files),'staged',len(staged),'bytes',sum(p.stat().st_size for p in files),'rawfiles',len(raw))
if __name__=='__main__':main()
