import logging

pin_home = "/home/lifter/pin-3.14-98223-gb010a12c6-gcc-linux/"
pintool_dir = "/home/lifter/pin-3.14-98223-gb010a12c6-gcc-linux/source/tools/MyPinTool/"



logging.basicConfig(filename='overall_time.log',
                    format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S %p',
                    level=logging.INFO)
