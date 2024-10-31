import gedi.config
from datetime import datetime as dt
from gedi.run import gedi, run
from utils.default_argparse import ArgParser
from utils.param_keys import *

if __name__=='__main__':
    start_gedi = dt.now()
    print(f'INFO: GEDI starting {start_gedi}')
    args = ArgParser().parse('GEDI main')
    gedi(args.alg_params_json)
    print(f'SUCCESS: GEDI took {dt.now()-start_gedi} sec.')