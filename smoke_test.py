"""Quick installation check; no student implementation is executed."""
from pathlib import Path
import tempfile
from transformer_labkit import generate,verify

def main():
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'variant';generate(1,1,p,n=4);verify(p)
    print('Installation, generation and verification: OK')

if __name__=='__main__':main()
