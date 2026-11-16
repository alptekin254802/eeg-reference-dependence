"""Execute generated explanation configurations in a deterministic sequence."""
import argparse,json
from pathlib import Path
from batch import run

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--configuration');a=p.parse_args();index=json.loads(a.index.read_text())
 for name,path in index.items():
  if a.configuration and name!=a.configuration:continue
  output=a.index.parent/name;output.mkdir(parents=True,exist_ok=True);run(json.loads(Path(path).read_text()),output)
if __name__=='__main__':main()
