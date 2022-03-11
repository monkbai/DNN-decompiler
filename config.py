import logging

pin_home = "/export/d1/zliudc/TOOL/pin-3.14-98223-gb010a12c6-gcc-linux/"
pintool_dir = "/export/d1/zliudc/TOOL/pin-3.14-98223-gb010a12c6-gcc-linux/source/tools/MyPinTool/"
# pin_home = "/home/lifter/Downloads/pin-3.14-98223-gb010a12c6-gcc-linux/"
# pintool_dir = "/home/lifter/Downloads/pin-3.14-98223-gb010a12c6-gcc-linux/source/tools/MyPinTool/"


logging.basicConfig(filename='overall_time.log',
                    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S %p',
                    level=logging.DEBUG)
