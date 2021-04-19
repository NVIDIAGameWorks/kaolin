#!/usr/bin/env python3

# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import shutil
import subprocess
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
KAOLIN_ROOT = os.path.realpath(
    os.path.join(THIS_DIR, os.pardir, os.pardir, os.pardir, os.pardir))

logger = logging.getLogger(__name__)


def obj_paths():
    samples_dir = os.path.join(KAOLIN_ROOT, 'tests', 'samples')
    return [os.path.join(samples_dir, 'rocket.obj'),
            os.path.join(samples_dir, 'model.obj')]


def timelapse_path():
    return os.path.realpath(
        os.path.join(KAOLIN_ROOT, 'tests', 'samples', 'timelapse', 'notexture'))


def golden_screenshots_path():
    return os.path.join(THIS_DIR, 'cypress', 'fixtures')


def cypress_config_path():
    # Important: must be relative
    return os.path.join('tests', 'integration', 'experimental', 'dash3d', 'cypress.json')


def port():
    return 8008


def generate_timelapse_input():
    objs = ','.join(obj_paths())
    out_dir = timelapse_path()
    script = os.path.realpath(
        os.path.join(KAOLIN_ROOT, 'examples', 'tutorial', 'visualize_main.py'))

    args = f'--skip_normalization --test_objs={objs} --output_dir={out_dir}'
    command = f'python {script} {args}'
    logger.info(f'Re-generating timelapse input here: {out_dir}\n by running {command}')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ret = os.system(command)
    if ret != 0:
        raise RuntimeError('Creation of timelapse failed')


def start_dash3d():
    script = os.path.realpath(os.path.join(THIS_DIR, 'start_dash3d.sh'))
    logdir = timelapse_path()
    _port = port()

    command = f'{script} {logdir} {_port}'
    logger.info(f'Starting dash3d server in the background by running {command}')
    ret = os.system(command)

    if ret != 0:
        raise RuntimeError('Failed to start Dash3D')


def run_cypress():
    command = 'npx cypress run --config-file {}'.format(cypress_config_path())
    logger.info(f'Starting cypress by running {command}')
    os.chdir(KAOLIN_ROOT)
    ret = os.system(command)
    if ret != 0:
        raise RuntimeError('Failed cypress integration test')


def run_end_to_end_integration_tests():
    print('END 2 END INTEGRATION TEST FOR DASH 3D-------------------------------')
    print('Timelapse input: {}'.format(timelapse_path()))
    print('Server: http://localhost:{}'.format(port))
    print('Golden screenshot files: {}'.format(golden_screenshots_path()))
    print('Visual comparison results: ')


def run_main(regenerate_timelapse_input,
             skip_start_dash3d):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s|%(levelname)8s|%(name)15s| %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

    if regenerate_timelapse_input:
        generate_timelapse_input()

    if not skip_start_dash3d:
        start_dash3d()

    run_cypress()


class TestBinaryEncoding:
    def test_server_client_binary_compatibility(self):
        run_main(regenerate_timelapse_input=False,
                 skip_start_dash3d=False)


if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--regenerate_timelapse_input', action='store_true',
                         help='If set, will regenerate timelapse input in {}')
    aparser.add_argument('--skip_start_dash3d', action='store_true',
                         help='If set, will skip starting dash3d, which may already be running.')
    args = aparser.parse_args()

    run_main(regenerate_timelapse_input=args.regenerate_timelapse_input,
             skip_start_dash3d=args.skip_start_dash3d)
