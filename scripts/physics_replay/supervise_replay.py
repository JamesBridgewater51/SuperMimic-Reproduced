"""Detached owner of the authorized bounded Warp sanity and original-eleven replay."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/liuminghao/artifacts/supermimic/x2/mjcf-physics-audit')
CODE = Path('/data/liuminghao/supermimic/repo-x2-physics-audit/scripts/physics_replay')
PYTHON = '/data/liuminghao/supermimic/repo-x2/.venv/bin/python'
LOGS = Path('/data/liuminghao/logs/supermimic/x2/mjcf-physics-audit')


def write_receipt(name, record):
    target = ROOT / name
    temporary = target.with_name('.'+target.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(record, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(target)


def execute(phase, command):
    started = time.time()
    with (LOGS/(phase+'.log')).open('ab') as stream:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
        write_receipt(phase+'-process.json', {'pid':child.pid,'supervisor_pid':os.getpid(), 'started_epoch':started, 'command':command})
        returncode = child.wait()
    write_receipt(phase+'-exit.json', {'returncode':returncode,'success':returncode==0,'started_epoch':started,'finished_epoch':time.time()})
    if returncode:raise SystemExit(returncode)


def main():
    ROOT.mkdir(parents=True,exist_ok=True); LOGS.mkdir(parents=True,exist_ok=True)
    target = ROOT/'execution-owner.json'
    with target.open('x') as stream:
        json.dump({'supervisor_pid':os.getpid(),'session_id':os.getsid(0),'started_epoch':time.time(),'scope':'bounded mobility plus original11 real Warp physics; no training jobs'},stream,indent=2);stream.flush();os.fsync(stream.fileno())
    execute('warp-sanity',[PYTHON,'-I',str(CODE/'sanity_warp.py'),'--model',str(ROOT/'prepared-corrected/model/compiled-scene.mjb'),'--output',str(ROOT/'sanity-warp')])
    execute('warp-eleven',[PYTHON,'-I',str(CODE/'record_warp_replay.py'),'--repo','/data/liuminghao/supermimic/repo-x2-physics-audit','--base-xml','/data/liuminghao/supermimic/repo-x2-physics-audit/assets/x2/agibot_x2ultra_physics_corrected.xml','--expected-model-sha256','4b5ef94ee6e8b073aed50dd2c5ecb1eb27edf32c5d3a5bac50179f7908d60197','--output',str(ROOT/'replay')])
    write_receipt('execution-exit.json',{'returncode':0,'success':True,'finished_epoch':time.time()})


if __name__=='__main__':
    try:
        main()
    except SystemExit as failure:
        write_receipt('execution-exit.json', {'returncode':int(failure.code),'success':False,'finished_epoch':time.time()})
        raise
