from smrtrpy import *
from script_read import *
from para_init import *
import argparse
import os
import shutil

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", type=str,
                    help="path to input road map")

parser.add_argument("-c", "--config", type=str,
                    help="path to config file", default='config.ini')

args = vars(parser.parse_args())

para_init(args["config"])

#para_init("config.ini")


if os.path.exists(gl.get_value('scratch_path')):

    with os.scandir(gl.get_value('scratch_path')) as entries:

        for entry in entries:

            if entry.is_file() or entry.is_symlink():

                os.remove(entry.path)

            elif entry.is_dir():

                shutil.rmtree(entry.path)

else:

    os.mkdir(gl.get_value('scratch_path'))

if not os.path.exists(gl.get_value('icon_path')):

    os.mkdir(gl.get_value('icon_path'))


script_input = script_process(args["input"])

#script_input = script_process("wrkflow_script.rmp")

AreaDivide().region_init_sep()

start_0 = time.time()

total = 0

for i in range(len(script_input)):

    print(script_input[i])

    start = time.time()

    smrtr_engine(script_input[i])

    end = time.time()

    print('Time Elapsed = ', end - start, 's.')

    total = total + end - start

    print('Total Time Elapsed = ', total, 's.')

    time.sleep(0.5)
    
end_0 = time.time()

print('Total Time =', end_0 - start_0, 's.')