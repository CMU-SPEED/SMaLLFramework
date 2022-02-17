import os
import sys

config_template = ''' 
#define config_kernel_size {:}
#define config_stride {:}

#define config_channel_stride {:}
'''

config_file = open("./config.h", "w")

config_txt = config_template.format(
                                    sys.argv[1], 
                                    sys.argv[2], 

                                    sys.argv[3]
                                    )
print(config_txt, file=config_file)
config_file.close()
# os.system("make torch_{:}.x".format(sys.argv[7]))
# os.system("cp torch_{:}.x  torch_{:}_{:}_{:}.x".format(sys.argv[7], sys.argv[7], sys.argv[5], sys.argv[6]))


vars = ["config_kernel_size", "config_stride", "config_channel_stride"]

bash_vars = "\n"
for i in range(len(vars)):
    bash_vars += "export {:}={:}\n".format(vars[i], sys.argv[i+1])


file = "Experiments/script_VGG_almost.sh"
index = 3
with open(file, 'r+') as fd:
    contents = fd.readlines()
    # print(contents)
    fd.seek(0)  # readlines consumes the iterator, so we need to start over
    i = 0
    print(len(contents))
    # c = input()
    while i < len(contents):
        print(contents[i])
        print("export config" in contents[i])
        if "export config" in contents[i]:
            i+=2
            fd.write(bash_vars)
        else:
            fd.write(contents[i])
            print("not modified", contents[i])
        i += 1
        print(i)

    # fd.writelines(contents) 