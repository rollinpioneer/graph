"""Wait for full validated extraction, then independently validate raw data."""
import json,subprocess,time,sys,fcntl
from pathlib import Path
R=Path(__file__).resolve().parent
lock=(R/'raw_validation_driver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
status=R/'RAW_VALIDATION_DRIVER.json'
try:
 while True:
  s=json.loads((R/'DATA_ACQUISITION_STATUS.json').read_text())
  if s['status']=='FAILED':raise RuntimeError(s)
  if s['status']=='RAW_ARCHIVES_EXTRACTED_PENDING_VALIDATION':break
  status.write_text(json.dumps({'status':'WAITING_FOR_RAW_ARCHIVES','updated_unix':time.time()}));time.sleep(20)
 for tag in ['check_official_crc.py','validate_raw.py']:
  status.write_text(json.dumps({'status':'RUNNING','step':tag,'updated_unix':time.time()}))
  subprocess.run([sys.executable,str(R/tag)],check=True)
 status.write_text(json.dumps({'status':'PASS','updated_unix':time.time(),'formal_training_started':False}))
except Exception as e:
 status.write_text(json.dumps({'status':'FAILED','error':repr(e),'updated_unix':time.time()}));raise
