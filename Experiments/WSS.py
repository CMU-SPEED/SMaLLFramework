#calculating the active volumes of Intermediate and Output Arrays\

def block_fusion(size):
  pool_size = (size-1)//2
  ip_size = (size+2)
  # print(pool_size)
  conv_volume = ((size**2)*16*4)/1024
  pool_volume = ((pool_size**2)*16*4)/1024
  ip_volume = ((ip_size**2)*16*4)/1024
  return conv_volume, pool_volume, ip_volume

def l_P_fusion(size):
  pool_size = (size-1)//2
  ip_size = (size+2)
  conv_volume = (3*(size)*16*4)/1024
  pool_volume = ((pool_size)*16*4)/1024
  ip_volume = (5*(ip_size)*16*4)/1024
  return conv_volume, pool_volume, ip_volume

def l_C_fusion(size):
  ip_size = (size+2)
  ip_volume = (3*(ip_size)*16*4)/1024
  pool_size = (size-1)//2
  conv_volume = ((size)*16*4)/1024
  pool_volume = (2*(pool_size)*16*4)/1024

  return conv_volume, pool_volume, ip_volume


def k_fusion():
  ip_size = (6+2)
  ip_volume = (3*(ip_size)*16*4)/1024
  conv_volume = 6*16*4/1024
  pool_volume = (6*(4)*16*4)/1024
  return conv_volume, pool_volume, ip_volume


def sweep(size):
    for j in [block_fusion, l_P_fusion, l_C_fusion]:
        conv_volume, pool_volume, ip_volume = j(size)
        total = conv_volume+pool_volume+ip_volume
        print("{:.2f},".format(
                                               total
                                              ), end=" ")
    conv_volume, pool_volume,ip_volume = k_fusion()
    if size > 12:
        total = conv_volume+pool_volume+ip_volume
        print("{:.2f}".format(
                                               total
                                              )
                                              )
    else:
        pool_volume = (4*(3.5)*16*4)/1024
        total = conv_volume+pool_volume+ip_volume
        print("{:.2f}".format(
                                               total
                                              )
                                              )

for i in [12,30,54,114,222,504]:
 sweep(i)
