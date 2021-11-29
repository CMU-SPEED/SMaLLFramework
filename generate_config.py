import os
import sys

config_template = ''' 
#define config_kernel_size {:}
#define config_stride {:}

#define config_G {:}
#define config_C_o_1 {:}

#define config_C_i  {:}
#define config_C_o  {:}

#define config_channel_stride config_C_o_1
'''

config_file = open("../config.h", "w")

config_txt = config_template.format(
                                    sys.argv[1], 
                                    sys.argv[2], 

                                    sys.argv[3],
                                    sys.argv[4],

                                    sys.argv[5],
                                    sys.argv[6]
                                    )
print(config_txt, file=config_file)
config_file.close()
os.system("make torch_{:}.x".format(sys.argv[7]))
os.system("cp torch_{:}.x  torch_{:}_{:}_{:}.x".format(sys.argv[7], sys.argv[7], sys.argv[5], sys.argv[6]))