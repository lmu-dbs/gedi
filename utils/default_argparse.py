import argparse


class ArgParser(object):
    @staticmethod
    def parse(description):
        parser = argparse.ArgumentParser(description=description)

        parser.add_argument(
            '-o',
            '--options',
            dest='run_params_json',
            help='a path to the parameter configuration of the program',
        )
        parser.add_argument(
            '-a',
            '--algorithm-configs',
            dest='alg_params_json',
            help='a path to the configurations of the algorithms'
        )
        parser.add_argument(
            '-l',
            '--load',
            dest='result_load_files',
            nargs='+',
            help='the list of paths to already saved results'
        )
        return parser.parse_args()