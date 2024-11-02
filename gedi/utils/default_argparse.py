import argparse


class ArgParser(object):
    @staticmethod
    def parse(description):
        parser = argparse.ArgumentParser(description=description)

        parser.add_argument(
            '-a',
            '--algorithm-configs',
            dest='alg_params_json',
            help='a path to the configurations of the algorithms'
        )
        return parser.parse_args()