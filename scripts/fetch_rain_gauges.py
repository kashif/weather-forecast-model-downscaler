"""Cache validated Météo-France RR1 observations for predeclared test hours.

Archives are department/two-year bulk files. Only requested columns/hours are
retained. Raw zero rain remains zero; missing or unvalidated values stay missing.
"""
from __future__ import annotations
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import yaml
from downscale.rain_resume import atomic_json

COLUMNS=['NUM_POSTE','NOM_USUEL','LAT','LON','ALTI','AAAAMMJJHH','RR1','QRR1']


def select_gauge_rows(frame,hours,quality_codes):
    """RR1 is already decimal mm: 'mm et 1/10' describes resolution, not scaling."""
    frame=frame.copy()
    stamps=pd.to_datetime(frame['AAAAMMJJHH'].astype(str),format='%Y%m%d%H',errors='coerce')
    rain=pd.to_numeric(frame['RR1'],errors='coerce')
    quality=pd.to_numeric(frame['QRR1'],errors='coerce')
    chosen=stamps.isin(hours)&quality.isin(quality_codes)&np.isfinite(rain)&(rain>=0)
    result=frame.loc[chosen,['NUM_POSTE','NOM_USUEL','LAT','LON','ALTI']].copy()
    result['time']=stamps[chosen].values;result['rain_mm']=rain[chosen].astype('float32').values
    result['quality']=quality[chosen].astype('int8').values
    result=result.rename(columns={'NUM_POSTE':'station','NOM_USUEL':'name','LAT':'lat','LON':'lon','ALTI':'elevation_m'})
    result['station']=result['station'].astype(str).str.zfill(8)
    for column in ['lat','lon','elevation_m']:result[column]=pd.to_numeric(result[column],errors='coerce')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config_rain_gauges.yaml')
    parser.add_argument('--out',default='data/precipitation/gauge_validation')
    parser.add_argument('--archive-cache',help='Reuse an existing source directory of bulk archives')
    args=parser.parse_args();cfg=yaml.safe_load(Path(args.config).read_text())
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    source=Path(args.archive_cache) if args.archive_cache else out/'source'
    source.mkdir(parents=True,exist_ok=True)
    lock=(out/'.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('gauge collection is already running') from None
    dates=pd.DatetimeIndex(cfg['splits']['test'])
    # Extra endpoints support an explicitly reported +/-1h timing sensitivity.
    hours=pd.DatetimeIndex(np.concatenate([pd.date_range(d,d+pd.Timedelta(days=1,hours=1),freq='h').values for d in dates])).unique()
    signature=dict(dates=cfg['splits']['test'],quality_codes=cfg['gauge_quality_codes'],columns=COLUMNS)
    marker=out/'selection_manifest.json'
    if marker.exists() and json.loads(marker.read_text())!=signature:raise ValueError('gauge cache selection differs')
    atomic_json(marker,signature)
    resource_path=source/'resources.json'
    if not resource_path.exists():resource_path.write_bytes(urllib.request.urlopen(cfg['gauge_source_dataset'],timeout=60).read())
    metadata=json.loads(resource_path.read_text())
    resources=[r for r in metadata['resources'] if re.fullmatch(r'HOR_departement_(\d{2}|2[AB])_periode_2025-2026',r['title'])]
    if not resources:raise ValueError('no metropolitan 2025-2026 archives listed')
    def fetch(resource):
        department=resource['title'].split('_')[2]
        raw=source/(department+'.csv.gz');filtered=out/(department+'.parquet');record=out/(department+'.json')
        if filtered.exists() and record.exists():return json.loads(record.read_text())
        if not raw.exists():
            temporary=raw.with_suffix('.tmp')
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(resource['url'],timeout=120) as response,temporary.open('wb') as stream:
                        shutil.copyfileobj(response,stream)
                    temporary.replace(raw);break
                except Exception:
                    if attempt==4:raise
                    time.sleep(2**attempt)
        frames=[]
        for frame in pd.read_csv(raw,sep=';',compression='gzip',encoding='utf-8-sig',usecols=COLUMNS,
                                 dtype={'NUM_POSTE':str,'AAAAMMJJHH':str},chunksize=150000,low_memory=False):
            selected=select_gauge_rows(frame,hours,cfg['gauge_quality_codes'])
            if len(selected):frames.append(selected)
        result=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['station','name','lat','lon','elevation_m','time','rain_mm','quality'])
        temporary=filtered.with_suffix('.tmp');result.to_parquet(temporary,index=False);temporary.replace(filtered)
        info=dict(department=department,url=resource['url'],resource_id=resource['id'],compressed_bytes=raw.stat().st_size,
                  raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),retained_rows=len(result))
        atomic_json(record,info);return info
    completed=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch,r) for r in resources]):
            completed.append(future.result())
            progress=dict(stage='gauge_archives',completed=len(completed),total=len(resources))
            atomic_json(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    frames=[pd.read_parquet(out/(r['department']+'.parquet')) for r in completed]
    result=pd.concat(frames,ignore_index=True)
    result.to_parquet(out/'observations.parquet',index=False)
    atomic_json(out/'source_manifest.json',dict(source='Météo-France hourly climatological RR1 archive',
                licence='Etalab Open Licence 2.0',quality_codes=cfg['gauge_quality_codes'],
                units='mm, already decimal values',time_zone='UTC, metropolitan archives only',
                interval_note='RR1 interpreted as preceding-hour accumulation; +/-1h sensitivity reported separately',
                signature=signature,resources=sorted(completed,key=lambda r:r['department'])))
    atomic_json(out/'progress.json',dict(stage='complete',rows=len(result),stations=int(result.station.nunique()),departments=len(resources)))


if __name__=='__main__':main()
