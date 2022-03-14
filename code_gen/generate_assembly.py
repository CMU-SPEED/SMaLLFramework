
W_ob = 6
C_ob = 16
SIMD = 4
stride = 1

a_byte_offset = C_ob*stride*4
b_byte_offset = SIMD*4
microkernel = '''#define fma_reg_broadcast({inputs})\
{\
__asm__ volatile \
(\
{load_scalars}\
{load_a_block} \
{load_b_block} \
{fma_block}\
"{load_b_block} \n\t"\
"{fma_block}  \n\t"\
\
"{load_b_block} \n\t"\
"{fma_block}  \n\t"\
"{load_b_block} \n\t"\
"{fma_block}  \n\t"\
\
"{load_b_block} \n\t"\
"{fma_block}  \n\t"\
"{load_b_block} \n\t"\
"{fma_block}  \n\t"\
\
"{load_b_block} \n\t"\
"{fma_block}  \n\t"\
"{load_b_block} \n\t"\
"{fma_block}  \n\t"\
\
:{output_names}\
:{input_names}\
:{clobber_list}\
);\
}'''


fma_reg_broadcast_instr = '''fmla %[c_reg].4s, %[b_reg].4s, %[a_reg].s[" #offset "] '''

scalar_load_instr = '''"ldr {reg_name},%[{obj}addr] \\n\\t"\\\n'''
vector_load_instr = '''"ldr q{reg_num}, [{base_reg}, #{offset}]  \\n\\t"\\\n'''



scalar_register_count = 0

a_addr_reg = "x{:}".format(scalar_register_count)
scalar_register_count += 1
b_addr_reg = "x{:}".format(scalar_register_count)


a_scalar_reg = scalar_load_instr.format(reg_name=a_addr_reg, obj='a')
b_scalar_reg = scalar_load_instr.format(reg_name=b_addr_reg, obj='b')

load_scalars = a_scalar_reg + b_scalar_reg

print(load_scalars)


load_a_block = ''' '''
vector_register_count = 0
for i in range(W_ob):
    load_a_block +=  vector_load_instr.format(reg_num=vector_register_count, base_reg=a_addr_reg , offset=i*a_byte_offset)
    vector_register_count += 1
print(load_a_block)


oad_a_block = ''' '''
vector_register_count = 0
for i in range(W_ob):
    load_a_block +=  vector_load_instr.format(reg_num=vector_register_count, base_reg=a_addr_reg , offset=i*a_byte_offset)
    vector_register_count += 1
print(load_a_block)