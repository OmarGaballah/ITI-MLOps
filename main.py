import numpy as np 

if __name__=="__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from mlops import cli

    cli.main()