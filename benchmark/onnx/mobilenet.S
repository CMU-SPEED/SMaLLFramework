
./tiny/mobilenet.so:     file format elf64-x86-64


Disassembly of section .init:

0000000000002000 <_init>:
    2000:	f3 0f 1e fa          	endbr64 
    2004:	48 83 ec 08          	sub    $0x8,%rsp
    2008:	48 8b 05 91 cf 0d 00 	mov    0xdcf91(%rip),%rax        # defa0 <__gmon_start__@Base>
    200f:	48 85 c0             	test   %rax,%rax
    2012:	74 02                	je     2016 <_init+0x16>
    2014:	ff d0                	call   *%rax
    2016:	48 83 c4 08          	add    $0x8,%rsp
    201a:	c3                   	ret    

Disassembly of section .plt:

0000000000002020 <printf@plt-0x10>:
    2020:	ff 35 e2 cf 0d 00    	push   0xdcfe2(%rip)        # df008 <_GLOBAL_OFFSET_TABLE_+0x8>
    2026:	ff 25 e4 cf 0d 00    	jmp    *0xdcfe4(%rip)        # df010 <_GLOBAL_OFFSET_TABLE_+0x10>
    202c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000002030 <printf@plt>:
    2030:	ff 25 e2 cf 0d 00    	jmp    *0xdcfe2(%rip)        # df018 <printf@GLIBC_2.2.5>
    2036:	68 00 00 00 00       	push   $0x0
    203b:	e9 e0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002040 <memset@plt>:
    2040:	ff 25 da cf 0d 00    	jmp    *0xdcfda(%rip)        # df020 <memset@GLIBC_2.2.5>
    2046:	68 01 00 00 00       	push   $0x1
    204b:	e9 d0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002050 <MatMul@plt>:
    2050:	ff 25 d2 cf 0d 00    	jmp    *0xdcfd2(%rip)        # df028 <MatMul@Base>
    2056:	68 02 00 00 00       	push   $0x2
    205b:	e9 c0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002060 <puts@plt>:
    2060:	ff 25 ca cf 0d 00    	jmp    *0xdcfca(%rip)        # df030 <puts@GLIBC_2.2.5>
    2066:	68 03 00 00 00       	push   $0x3
    206b:	e9 b0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002070 <omTensorGetStrides@plt>:
    2070:	ff 25 c2 cf 0d 00    	jmp    *0xdcfc2(%rip)        # df038 <omTensorGetStrides@@Base+0xd21c0>
    2076:	68 04 00 00 00       	push   $0x4
    207b:	e9 a0 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002080 <__assert_fail@plt>:
    2080:	ff 25 ba cf 0d 00    	jmp    *0xdcfba(%rip)        # df040 <__assert_fail@GLIBC_2.2.5>
    2086:	68 05 00 00 00       	push   $0x5
    208b:	e9 90 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002090 <omTensorListGetOmtArray@plt>:
    2090:	ff 25 b2 cf 0d 00    	jmp    *0xdcfb2(%rip)        # df048 <omTensorListGetOmtArray@@Base+0xd148c>
    2096:	68 06 00 00 00       	push   $0x6
    209b:	e9 80 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020a0 <omTensorSetDataType@plt>:
    20a0:	ff 25 aa cf 0d 00    	jmp    *0xdcfaa(%rip)        # df050 <omTensorSetDataType@@Base+0xd20d0>
    20a6:	68 07 00 00 00       	push   $0x7
    20ab:	e9 70 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020b0 <putchar@plt>:
    20b0:	ff 25 a2 cf 0d 00    	jmp    *0xdcfa2(%rip)        # df058 <putchar@GLIBC_2.2.5>
    20b6:	68 08 00 00 00       	push   $0x8
    20bb:	e9 60 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020c0 <strncmp@plt>:
    20c0:	ff 25 9a cf 0d 00    	jmp    *0xdcf9a(%rip)        # df060 <strncmp@GLIBC_2.2.5>
    20c6:	68 09 00 00 00       	push   $0x9
    20cb:	e9 50 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020d0 <malloc@plt>:
    20d0:	ff 25 92 cf 0d 00    	jmp    *0xdcf92(%rip)        # df068 <malloc@GLIBC_2.2.5>
    20d6:	68 0a 00 00 00       	push   $0xa
    20db:	e9 40 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020e0 <omTensorListCreate@plt>:
    20e0:	ff 25 8a cf 0d 00    	jmp    *0xdcf8a(%rip)        # df070 <omTensorListCreate@@Base+0xd15ea>
    20e6:	68 0b 00 00 00       	push   $0xb
    20eb:	e9 30 ff ff ff       	jmp    2020 <_init+0x20>

00000000000020f0 <omTensorSetDataPtr@plt>:
    20f0:	ff 25 82 cf 0d 00    	jmp    *0xdcf82(%rip)        # df078 <omTensorSetDataPtr@@Base+0xd22e7>
    20f6:	68 0c 00 00 00       	push   $0xc
    20fb:	e9 20 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002100 <omTensorGetDataPtr@plt>:
    2100:	ff 25 7a cf 0d 00    	jmp    *0xdcf7a(%rip)        # df080 <omTensorGetDataPtr@@Base+0xd2305>
    2106:	68 0d 00 00 00       	push   $0xd
    210b:	e9 10 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002110 <free@plt>:
    2110:	ff 25 72 cf 0d 00    	jmp    *0xdcf72(%rip)        # df088 <free@GLIBC_2.2.5>
    2116:	68 0e 00 00 00       	push   $0xe
    211b:	e9 00 ff ff ff       	jmp    2020 <_init+0x20>

0000000000002120 <strlen@plt>:
    2120:	ff 25 6a cf 0d 00    	jmp    *0xdcf6a(%rip)        # df090 <strlen@GLIBC_2.2.5>
    2126:	68 0f 00 00 00       	push   $0xf
    212b:	e9 f0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002130 <expf@plt>:
    2130:	ff 25 62 cf 0d 00    	jmp    *0xdcf62(%rip)        # df098 <expf@GLIBC_2.27>
    2136:	68 10 00 00 00       	push   $0x10
    213b:	e9 e0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002140 <om_f16_to_f32@plt>:
    2140:	ff 25 5a cf 0d 00    	jmp    *0xdcf5a(%rip)        # df0a0 <om_f16_to_f32@@Base+0xd1429>
    2146:	68 11 00 00 00       	push   $0x11
    214b:	e9 d0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002150 <om_f32_to_f16@plt>:
    2150:	ff 25 52 cf 0d 00    	jmp    *0xdcf52(%rip)        # df0a8 <om_f32_to_f16@@Base+0xd1371>
    2156:	68 12 00 00 00       	push   $0x12
    215b:	e9 c0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002160 <Conv2D_bias@plt>:
    2160:	ff 25 4a cf 0d 00    	jmp    *0xdcf4a(%rip)        # df0b0 <Conv2D_bias@Base>
    2166:	68 13 00 00 00       	push   $0x13
    216b:	e9 b0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002170 <main_graph_mobilenet@plt>:
    2170:	ff 25 42 cf 0d 00    	jmp    *0xdcf42(%rip)        # df0b8 <main_graph_mobilenet@@Base+0xdcde8>
    2176:	68 14 00 00 00       	push   $0x14
    217b:	e9 a0 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002180 <__stack_chk_fail@plt>:
    2180:	ff 25 3a cf 0d 00    	jmp    *0xdcf3a(%rip)        # df0c0 <__stack_chk_fail@GLIBC_2.4>
    2186:	68 15 00 00 00       	push   $0x15
    218b:	e9 90 fe ff ff       	jmp    2020 <_init+0x20>

0000000000002190 <AveragePool@plt>:
    2190:	ff 25 32 cf 0d 00    	jmp    *0xdcf32(%rip)        # df0c8 <AveragePool@Base>
    2196:	68 16 00 00 00       	push   $0x16
    219b:	e9 80 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021a0 <omTensorCreateUntyped@plt>:
    21a0:	ff 25 2a cf 0d 00    	jmp    *0xdcf2a(%rip)        # df0d0 <omTensorCreateUntyped@@Base+0xd253b>
    21a6:	68 17 00 00 00       	push   $0x17
    21ab:	e9 70 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021b0 <omTensorDestroy@plt>:
    21b0:	ff 25 22 cf 0d 00    	jmp    *0xdcf22(%rip)        # df0d8 <omTensorDestroy@@Base+0xd23c1>
    21b6:	68 18 00 00 00       	push   $0x18
    21bb:	e9 60 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021c0 <run_main_graph_mobilenet@plt>:
    21c0:	ff 25 1a cf 0d 00    	jmp    *0xdcf1a(%rip)        # df0e0 <run_main_graph_mobilenet@@Base+0xd2ef0>
    21c6:	68 19 00 00 00       	push   $0x19
    21cb:	e9 50 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021d0 <omTensorGetShape@plt>:
    21d0:	ff 25 12 cf 0d 00    	jmp    *0xdcf12(%rip)        # df0e8 <omTensorGetShape@@Base+0xd22e4>
    21d6:	68 1a 00 00 00       	push   $0x1a
    21db:	e9 40 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021e0 <memcpy@plt>:
    21e0:	ff 25 0a cf 0d 00    	jmp    *0xdcf0a(%rip)        # df0f0 <memcpy@GLIBC_2.14>
    21e6:	68 1b 00 00 00       	push   $0x1b
    21eb:	e9 30 fe ff ff       	jmp    2020 <_init+0x20>

00000000000021f0 <Relu@plt>:
    21f0:	ff 25 02 cf 0d 00    	jmp    *0xdcf02(%rip)        # df0f8 <Relu@Base>
    21f6:	68 1c 00 00 00       	push   $0x1c
    21fb:	e9 20 fe ff ff       	jmp    2020 <_init+0x20>

Disassembly of section .plt.got:

0000000000002200 <__cxa_finalize@plt>:
    2200:	ff 25 b2 cd 0d 00    	jmp    *0xdcdb2(%rip)        # defb8 <__cxa_finalize@GLIBC_2.2.5>
    2206:	66 90                	xchg   %ax,%ax

Disassembly of section .text:

0000000000002210 <deregister_tm_clones>:
    2210:	48 8d 3d 91 cf 0d 00 	lea    0xdcf91(%rip),%rdi        # df1a8 <completed.0>
    2217:	48 8d 05 8a cf 0d 00 	lea    0xdcf8a(%rip),%rax        # df1a8 <completed.0>
    221e:	48 39 f8             	cmp    %rdi,%rax
    2221:	74 15                	je     2238 <deregister_tm_clones+0x28>
    2223:	48 8b 05 7e cd 0d 00 	mov    0xdcd7e(%rip),%rax        # defa8 <_ITM_deregisterTMCloneTable@Base>
    222a:	48 85 c0             	test   %rax,%rax
    222d:	74 09                	je     2238 <deregister_tm_clones+0x28>
    222f:	ff e0                	jmp    *%rax
    2231:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2238:	c3                   	ret    
    2239:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002240 <register_tm_clones>:
    2240:	48 8d 3d 61 cf 0d 00 	lea    0xdcf61(%rip),%rdi        # df1a8 <completed.0>
    2247:	48 8d 35 5a cf 0d 00 	lea    0xdcf5a(%rip),%rsi        # df1a8 <completed.0>
    224e:	48 29 fe             	sub    %rdi,%rsi
    2251:	48 89 f0             	mov    %rsi,%rax
    2254:	48 c1 ee 3f          	shr    $0x3f,%rsi
    2258:	48 c1 f8 03          	sar    $0x3,%rax
    225c:	48 01 c6             	add    %rax,%rsi
    225f:	48 d1 fe             	sar    %rsi
    2262:	74 14                	je     2278 <register_tm_clones+0x38>
    2264:	48 8b 05 45 cd 0d 00 	mov    0xdcd45(%rip),%rax        # defb0 <_ITM_registerTMCloneTable@Base>
    226b:	48 85 c0             	test   %rax,%rax
    226e:	74 08                	je     2278 <register_tm_clones+0x38>
    2270:	ff e0                	jmp    *%rax
    2272:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2278:	c3                   	ret    
    2279:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002280 <__do_global_dtors_aux>:
    2280:	f3 0f 1e fa          	endbr64 
    2284:	80 3d 1d cf 0d 00 00 	cmpb   $0x0,0xdcf1d(%rip)        # df1a8 <completed.0>
    228b:	75 2b                	jne    22b8 <__do_global_dtors_aux+0x38>
    228d:	55                   	push   %rbp
    228e:	48 83 3d 22 cd 0d 00 	cmpq   $0x0,0xdcd22(%rip)        # defb8 <__cxa_finalize@GLIBC_2.2.5>
    2295:	00 
    2296:	48 89 e5             	mov    %rsp,%rbp
    2299:	74 0c                	je     22a7 <__do_global_dtors_aux+0x27>
    229b:	48 8b 3d 5e ce 0d 00 	mov    0xdce5e(%rip),%rdi        # df100 <__dso_handle>
    22a2:	e8 59 ff ff ff       	call   2200 <__cxa_finalize@plt>
    22a7:	e8 64 ff ff ff       	call   2210 <deregister_tm_clones>
    22ac:	c6 05 f5 ce 0d 00 01 	movb   $0x1,0xdcef5(%rip)        # df1a8 <completed.0>
    22b3:	5d                   	pop    %rbp
    22b4:	c3                   	ret    
    22b5:	0f 1f 00             	nopl   (%rax)
    22b8:	c3                   	ret    
    22b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000022c0 <frame_dummy>:
    22c0:	f3 0f 1e fa          	endbr64 
    22c4:	e9 77 ff ff ff       	jmp    2240 <register_tm_clones>
    22c9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000022d0 <main_graph_mobilenet>:
    22d0:	55                   	push   %rbp
    22d1:	41 57                	push   %r15
    22d3:	41 56                	push   %r14
    22d5:	41 55                	push   %r13
    22d7:	41 54                	push   %r12
    22d9:	53                   	push   %rbx
    22da:	48 81 ec b8 02 00 00 	sub    $0x2b8,%rsp
    22e1:	4c 89 c3             	mov    %r8,%rbx
    22e4:	49 89 d6             	mov    %rdx,%r14
    22e7:	48 89 bc 24 28 02 00 	mov    %rdi,0x228(%rsp)
    22ee:	00 
    22ef:	49 69 f8 00 b0 01 00 	imul   $0x1b000,%r8,%rdi
    22f6:	48 83 cf 10          	or     $0x10,%rdi
    22fa:	e8 d1 fd ff ff       	call   20d0 <malloc@plt>
    22ff:	48 89 c5             	mov    %rax,%rbp
    2302:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    2307:	48 83 c5 0f          	add    $0xf,%rbp
    230b:	48 83 e5 f0          	and    $0xfffffffffffffff0,%rbp
    230f:	48 89 5c 24 20       	mov    %rbx,0x20(%rsp)
    2314:	48 85 db             	test   %rbx,%rbx
    2317:	0f 8e 80 01 00 00    	jle    249d <main_graph_mobilenet+0x1cd>
    231d:	4c 8d 9d 00 20 01 00 	lea    0x12000(%rbp),%r11
    2324:	4d 8d 56 08          	lea    0x8(%r14),%r10
    2328:	31 d2                	xor    %edx,%edx
    232a:	4c 89 f3             	mov    %r14,%rbx
    232d:	eb 2d                	jmp    235c <main_graph_mobilenet+0x8c>
    232f:	90                   	nop
    2330:	48 ff c2             	inc    %rdx
    2333:	48 81 c3 00 b0 01 00 	add    $0x1b000,%rbx
    233a:	4c 8b 5c 24 10       	mov    0x10(%rsp),%r11
    233f:	49 81 c3 00 b0 01 00 	add    $0x1b000,%r11
    2346:	4c 8b 14 24          	mov    (%rsp),%r10
    234a:	49 81 c2 00 b0 01 00 	add    $0x1b000,%r10
    2351:	48 3b 54 24 20       	cmp    0x20(%rsp),%rdx
    2356:	0f 84 41 01 00 00    	je     249d <main_graph_mobilenet+0x1cd>
    235c:	48 69 fa 00 b0 01 00 	imul   $0x1b000,%rdx,%rdi
    2363:	4c 8d 87 80 21 01 00 	lea    0x12180(%rdi),%r8
    236a:	4c 8d 8f 80 04 00 00 	lea    0x480(%rdi),%r9
    2371:	4c 89 14 24          	mov    %r10,(%rsp)
    2375:	4c 89 5c 24 10       	mov    %r11,0x10(%rsp)
    237a:	48 89 de             	mov    %rbx,%rsi
    237d:	45 31 e4             	xor    %r12d,%r12d
    2380:	eb 2c                	jmp    23ae <main_graph_mobilenet+0xde>
    2382:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    2389:	00 00 00 
    238c:	0f 1f 40 00          	nopl   0x0(%rax)
    2390:	49 ff c4             	inc    %r12
    2393:	48 81 c6 80 04 00 00 	add    $0x480,%rsi
    239a:	49 81 c3 80 01 00 00 	add    $0x180,%r11
    23a1:	49 81 c2 80 04 00 00 	add    $0x480,%r10
    23a8:	49 83 fc 60          	cmp    $0x60,%r12
    23ac:	74 82                	je     2330 <main_graph_mobilenet+0x60>
    23ae:	4d 89 e5             	mov    %r12,%r13
    23b1:	49 c1 e5 07          	shl    $0x7,%r13
    23b5:	4f 8d 7c ed 00       	lea    0x0(%r13,%r13,8),%r15
    23ba:	4b 8d 04 0e          	lea    (%r14,%r9,1),%rax
    23be:	4c 01 f8             	add    %r15,%rax
    23c1:	4f 8d 6c 6d 00       	lea    0x0(%r13,%r13,2),%r13
    23c6:	48 8d 4c 3d 00       	lea    0x0(%rbp,%rdi,1),%rcx
    23cb:	4c 01 e9             	add    %r13,%rcx
    23ce:	48 39 c1             	cmp    %rax,%rcx
    23d1:	73 5d                	jae    2430 <main_graph_mobilenet+0x160>
    23d3:	4d 01 f7             	add    %r14,%r15
    23d6:	49 01 ff             	add    %rdi,%r15
    23d9:	49 01 ed             	add    %rbp,%r13
    23dc:	4d 01 c5             	add    %r8,%r13
    23df:	4d 39 ef             	cmp    %r13,%r15
    23e2:	73 4c                	jae    2430 <main_graph_mobilenet+0x160>
    23e4:	49 c7 c7 80 fe ff ff 	mov    $0xfffffffffffffe80,%r15
    23eb:	4d 89 d5             	mov    %r10,%r13
    23ee:	66 90                	xchg   %ax,%ax
    23f0:	f3 41 0f 10 45 f8    	movss  -0x8(%r13),%xmm0
    23f6:	f3 43 0f 11 84 3b 80 	movss  %xmm0,-0x11e80(%r11,%r15,1)
    23fd:	e1 fe ff 
    2400:	f3 41 0f 10 45 fc    	movss  -0x4(%r13),%xmm0
    2406:	f3 43 0f 11 84 3b 80 	movss  %xmm0,-0x8e80(%r11,%r15,1)
    240d:	71 ff ff 
    2410:	f3 41 0f 10 45 00    	movss  0x0(%r13),%xmm0
    2416:	f3 43 0f 11 84 3b 80 	movss  %xmm0,0x180(%r11,%r15,1)
    241d:	01 00 00 
    2420:	49 83 c5 0c          	add    $0xc,%r13
    2424:	49 83 c7 04          	add    $0x4,%r15
    2428:	75 c6                	jne    23f0 <main_graph_mobilenet+0x120>
    242a:	e9 61 ff ff ff       	jmp    2390 <main_graph_mobilenet+0xc0>
    242f:	90                   	nop
    2430:	49 c7 c5 a0 ff ff ff 	mov    $0xffffffffffffffa0,%r13
    2437:	49 89 f7             	mov    %rsi,%r15
    243a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2440:	41 0f 10 07          	movups (%r15),%xmm0
    2444:	41 0f 10 4f 10       	movups 0x10(%r15),%xmm1
    2449:	41 0f 10 57 20       	movups 0x20(%r15),%xmm2
    244e:	0f 28 da             	movaps %xmm2,%xmm3
    2451:	0f c6 d9 21          	shufps $0x21,%xmm1,%xmm3
    2455:	0f 28 e2             	movaps %xmm2,%xmm4
    2458:	0f c6 e1 f6          	shufps $0xf6,%xmm1,%xmm4
    245c:	0f 28 e8             	movaps %xmm0,%xmm5
    245f:	0f c6 e9 01          	shufps $0x1,%xmm1,%xmm5
    2463:	0f c6 c8 21          	shufps $0x21,%xmm0,%xmm1
    2467:	0f c6 c3 2c          	shufps $0x2c,%xmm3,%xmm0
    246b:	0f c6 ec 28          	shufps $0x28,%xmm4,%xmm5
    246f:	0f c6 ca c2          	shufps $0xc2,%xmm2,%xmm1
    2473:	43 0f 29 84 ab 80 e1 	movaps %xmm0,-0x11e80(%r11,%r13,4)
    247a:	fe ff 
    247c:	43 0f 29 ac ab 80 71 	movaps %xmm5,-0x8e80(%r11,%r13,4)
    2483:	ff ff 
    2485:	43 0f 29 8c ab 80 01 	movaps %xmm1,0x180(%r11,%r13,4)
    248c:	00 00 
    248e:	49 83 c7 30          	add    $0x30,%r15
    2492:	49 83 c5 04          	add    $0x4,%r13
    2496:	75 a8                	jne    2440 <main_graph_mobilenet+0x170>
    2498:	e9 f3 fe ff ff       	jmp    2390 <main_graph_mobilenet+0xc0>
    249d:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
    24a2:	48 89 d0             	mov    %rdx,%rax
    24a5:	48 c1 e0 0b          	shl    $0xb,%rax
    24a9:	48 8d 0c c0          	lea    (%rax,%rax,8),%rcx
    24ad:	48 8d 04 d5 00 00 00 	lea    0x0(,%rdx,8),%rax
    24b4:	00 
    24b5:	48 83 fa ff          	cmp    $0xffffffffffffffff,%rdx
    24b9:	48 89 d7             	mov    %rdx,%rdi
    24bc:	be 08 00 00 00       	mov    $0x8,%esi
    24c1:	48 0f 45 f0          	cmovne %rax,%rsi
    24c5:	48 89 f0             	mov    %rsi,%rax
    24c8:	48 f7 d0             	not    %rax
    24cb:	48 c1 f8 3f          	sar    $0x3f,%rax
    24cf:	48 83 c8 01          	or     $0x1,%rax
    24d3:	48 29 c8             	sub    %rcx,%rax
    24d6:	48 99                	cqto   
    24d8:	48 f7 fe             	idiv   %rsi
    24db:	48 89 c3             	mov    %rax,%rbx
    24de:	48 f7 d3             	not    %rbx
    24e1:	48 89 c8             	mov    %rcx,%rax
    24e4:	48 99                	cqto   
    24e6:	48 f7 fe             	idiv   %rsi
    24e9:	48 85 f6             	test   %rsi,%rsi
    24ec:	48 89 c2             	mov    %rax,%rdx
    24ef:	48 0f 48 d3          	cmovs  %rbx,%rdx
    24f3:	48 85 c9             	test   %rcx,%rcx
    24f6:	48 0f 4e d0          	cmovle %rax,%rdx
    24fa:	48 85 f6             	test   %rsi,%rsi
    24fd:	48 0f 4e da          	cmovle %rdx,%rbx
    2501:	48 85 c9             	test   %rcx,%rcx
    2504:	48 0f 49 da          	cmovns %rdx,%rbx
    2508:	48 83 ff ff          	cmp    $0xffffffffffffffff,%rdi
    250c:	48 89 f8             	mov    %rdi,%rax
    250f:	48 0f 44 c3          	cmove  %rbx,%rax
    2513:	48 89 84 24 e8 00 00 	mov    %rax,0xe8(%rsp)
    251a:	00 
    251b:	48 89 f8             	mov    %rdi,%rax
    251e:	49 89 fe             	mov    %rdi,%r14
    2521:	48 c1 e0 0d          	shl    $0xd,%rax
    2525:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    252a:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    252f:	e8 9c fb ff ff       	call   20d0 <malloc@plt>
    2534:	49 89 c7             	mov    %rax,%r15
    2537:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    253b:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    2540:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    2544:	4c 89 64 24 60       	mov    %r12,0x60(%rsp)
    2549:	bf 04 00 00 00       	mov    $0x4,%edi
    254e:	e8 4d fc ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    2553:	49 89 c5             	mov    %rax,%r13
    2556:	48 89 04 24          	mov    %rax,(%rsp)
    255a:	48 89 c7             	mov    %rax,%rdi
    255d:	31 f6                	xor    %esi,%esi
    255f:	4c 89 fa             	mov    %r15,%rdx
    2562:	4c 89 e1             	mov    %r12,%rcx
    2565:	e8 86 fb ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    256a:	be 01 00 00 00       	mov    $0x1,%esi
    256f:	4c 89 ef             	mov    %r13,%rdi
    2572:	e8 29 fb ff ff       	call   20a0 <omTensorSetDataType@plt>
    2577:	4c 89 ef             	mov    %r13,%rdi
    257a:	e8 51 fc ff ff       	call   21d0 <omTensorGetShape@plt>
    257f:	49 89 c7             	mov    %rax,%r15
    2582:	4c 89 ef             	mov    %r13,%rdi
    2585:	e8 e6 fa ff ff       	call   2070 <omTensorGetStrides@plt>
    258a:	4d 89 37             	mov    %r14,(%r15)
    258d:	4d 89 f4             	mov    %r14,%r12
    2590:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    2597:	49 c7 47 08 08 00 00 	movq   $0x8,0x8(%r15)
    259e:	00 
    259f:	48 c7 40 08 00 09 00 	movq   $0x900,0x8(%rax)
    25a6:	00 
    25a7:	49 c7 47 10 30 00 00 	movq   $0x30,0x10(%r15)
    25ae:	00 
    25af:	48 c7 40 10 30 00 00 	movq   $0x30,0x10(%rax)
    25b6:	00 
    25b7:	49 c7 47 18 30 00 00 	movq   $0x30,0x18(%r15)
    25be:	00 
    25bf:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    25c6:	00 
    25c7:	bf 04 00 00 00       	mov    $0x4,%edi
    25cc:	e8 cf fb ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    25d1:	49 89 c5             	mov    %rax,%r13
    25d4:	48 89 c7             	mov    %rax,%rdi
    25d7:	31 f6                	xor    %esi,%esi
    25d9:	4c 8b 74 24 30       	mov    0x30(%rsp),%r14
    25de:	4c 89 f2             	mov    %r14,%rdx
    25e1:	48 89 e9             	mov    %rbp,%rcx
    25e4:	e8 07 fb ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    25e9:	be 01 00 00 00       	mov    $0x1,%esi
    25ee:	4c 89 ef             	mov    %r13,%rdi
    25f1:	e8 aa fa ff ff       	call   20a0 <omTensorSetDataType@plt>
    25f6:	4c 89 ef             	mov    %r13,%rdi
    25f9:	e8 d2 fb ff ff       	call   21d0 <omTensorGetShape@plt>
    25fe:	49 89 c7             	mov    %rax,%r15
    2601:	4c 89 ef             	mov    %r13,%rdi
    2604:	e8 67 fa ff ff       	call   2070 <omTensorGetStrides@plt>
    2609:	4d 89 27             	mov    %r12,(%r15)
    260c:	48 c7 00 00 6c 00 00 	movq   $0x6c00,(%rax)
    2613:	49 c7 47 08 03 00 00 	movq   $0x3,0x8(%r15)
    261a:	00 
    261b:	48 c7 40 08 00 24 00 	movq   $0x2400,0x8(%rax)
    2622:	00 
    2623:	49 c7 47 10 60 00 00 	movq   $0x60,0x10(%r15)
    262a:	00 
    262b:	48 c7 40 10 60 00 00 	movq   $0x60,0x10(%rax)
    2632:	00 
    2633:	49 c7 47 18 60 00 00 	movq   $0x60,0x18(%r15)
    263a:	00 
    263b:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    2642:	00 
    2643:	4c 8d 3d 06 35 01 00 	lea    0x13506(%rip),%r15        # 15b50 <constant_58_mobilenet>
    264a:	bf 04 00 00 00       	mov    $0x4,%edi
    264f:	e8 4c fb ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    2654:	48 89 c5             	mov    %rax,%rbp
    2657:	48 89 c7             	mov    %rax,%rdi
    265a:	31 f6                	xor    %esi,%esi
    265c:	4c 89 fa             	mov    %r15,%rdx
    265f:	4c 89 f9             	mov    %r15,%rcx
    2662:	e8 89 fa ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    2667:	be 01 00 00 00       	mov    $0x1,%esi
    266c:	48 89 ef             	mov    %rbp,%rdi
    266f:	e8 2c fa ff ff       	call   20a0 <omTensorSetDataType@plt>
    2674:	48 89 ef             	mov    %rbp,%rdi
    2677:	e8 54 fb ff ff       	call   21d0 <omTensorGetShape@plt>
    267c:	49 89 c7             	mov    %rax,%r15
    267f:	48 89 ef             	mov    %rbp,%rdi
    2682:	e8 e9 f9 ff ff       	call   2070 <omTensorGetStrides@plt>
    2687:	49 c7 07 08 00 00 00 	movq   $0x8,(%r15)
    268e:	48 c7 00 1b 00 00 00 	movq   $0x1b,(%rax)
    2695:	49 c7 47 08 03 00 00 	movq   $0x3,0x8(%r15)
    269c:	00 
    269d:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    26a4:	00 
    26a5:	49 c7 47 10 03 00 00 	movq   $0x3,0x10(%r15)
    26ac:	00 
    26ad:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    26b4:	00 
    26b5:	49 c7 47 18 03 00 00 	movq   $0x3,0x18(%r15)
    26bc:	00 
    26bd:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    26c4:	00 
    26c5:	4c 8d 25 c4 75 01 00 	lea    0x175c4(%rip),%r12        # 19c90 <constant_52_mobilenet>
    26cc:	bf 01 00 00 00       	mov    $0x1,%edi
    26d1:	e8 ca fa ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    26d6:	49 89 c7             	mov    %rax,%r15
    26d9:	48 89 c7             	mov    %rax,%rdi
    26dc:	31 f6                	xor    %esi,%esi
    26de:	4c 89 e2             	mov    %r12,%rdx
    26e1:	4c 89 e1             	mov    %r12,%rcx
    26e4:	e8 07 fa ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    26e9:	be 01 00 00 00       	mov    $0x1,%esi
    26ee:	4c 89 ff             	mov    %r15,%rdi
    26f1:	e8 aa f9 ff ff       	call   20a0 <omTensorSetDataType@plt>
    26f6:	4c 89 ff             	mov    %r15,%rdi
    26f9:	e8 d2 fa ff ff       	call   21d0 <omTensorGetShape@plt>
    26fe:	49 89 c4             	mov    %rax,%r12
    2701:	4c 89 ff             	mov    %r15,%rdi
    2704:	e8 67 f9 ff ff       	call   2070 <omTensorGetStrides@plt>
    2709:	49 c7 04 24 08 00 00 	movq   $0x8,(%r12)
    2710:	00 
    2711:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    2718:	4c 8b 24 24          	mov    (%rsp),%r12
    271c:	4c 89 e7             	mov    %r12,%rdi
    271f:	4c 89 ee             	mov    %r13,%rsi
    2722:	48 89 ea             	mov    %rbp,%rdx
    2725:	4c 89 f9             	mov    %r15,%rcx
    2728:	45 31 c0             	xor    %r8d,%r8d
    272b:	45 31 c9             	xor    %r9d,%r9d
    272e:	6a 02                	push   $0x2
    2730:	6a 02                	push   $0x2
    2732:	6a 01                	push   $0x1
    2734:	6a 01                	push   $0x1
    2736:	e8 25 fa ff ff       	call   2160 <Conv2D_bias@plt>
    273b:	48 83 c4 20          	add    $0x20,%rsp
    273f:	4c 89 e7             	mov    %r12,%rdi
    2742:	e8 69 fa ff ff       	call   21b0 <omTensorDestroy@plt>
    2747:	4c 89 ef             	mov    %r13,%rdi
    274a:	e8 61 fa ff ff       	call   21b0 <omTensorDestroy@plt>
    274f:	48 89 ef             	mov    %rbp,%rdi
    2752:	e8 59 fa ff ff       	call   21b0 <omTensorDestroy@plt>
    2757:	4c 89 ff             	mov    %r15,%rdi
    275a:	e8 51 fa ff ff       	call   21b0 <omTensorDestroy@plt>
    275f:	4c 89 f7             	mov    %r14,%rdi
    2762:	e8 a9 f9 ff ff       	call   2110 <free@plt>
    2767:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    276c:	e8 5f f9 ff ff       	call   20d0 <malloc@plt>
    2771:	49 89 c7             	mov    %rax,%r15
    2774:	4c 8d 70 0f          	lea    0xf(%rax),%r14
    2778:	48 89 04 24          	mov    %rax,(%rsp)
    277c:	49 83 e6 f0          	and    $0xfffffffffffffff0,%r14
    2780:	4c 89 74 24 08       	mov    %r14,0x8(%rsp)
    2785:	bf 04 00 00 00       	mov    $0x4,%edi
    278a:	e8 11 fa ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    278f:	49 89 c5             	mov    %rax,%r13
    2792:	48 89 c7             	mov    %rax,%rdi
    2795:	31 f6                	xor    %esi,%esi
    2797:	4c 89 fa             	mov    %r15,%rdx
    279a:	4c 89 f1             	mov    %r14,%rcx
    279d:	e8 4e f9 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    27a2:	be 01 00 00 00       	mov    $0x1,%esi
    27a7:	4c 89 ef             	mov    %r13,%rdi
    27aa:	e8 f1 f8 ff ff       	call   20a0 <omTensorSetDataType@plt>
    27af:	4c 89 ef             	mov    %r13,%rdi
    27b2:	e8 19 fa ff ff       	call   21d0 <omTensorGetShape@plt>
    27b7:	49 89 c7             	mov    %rax,%r15
    27ba:	4c 89 ef             	mov    %r13,%rdi
    27bd:	e8 ae f8 ff ff       	call   2070 <omTensorGetStrides@plt>
    27c2:	4c 8b 74 24 20       	mov    0x20(%rsp),%r14
    27c7:	4d 89 37             	mov    %r14,(%r15)
    27ca:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    27d1:	49 c7 47 08 08 00 00 	movq   $0x8,0x8(%r15)
    27d8:	00 
    27d9:	48 c7 40 08 00 09 00 	movq   $0x900,0x8(%rax)
    27e0:	00 
    27e1:	49 c7 47 10 30 00 00 	movq   $0x30,0x10(%r15)
    27e8:	00 
    27e9:	48 c7 40 10 30 00 00 	movq   $0x30,0x10(%rax)
    27f0:	00 
    27f1:	49 c7 47 18 30 00 00 	movq   $0x30,0x18(%r15)
    27f8:	00 
    27f9:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    2800:	00 
    2801:	bf 04 00 00 00       	mov    $0x4,%edi
    2806:	e8 95 f9 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    280b:	49 89 c7             	mov    %rax,%r15
    280e:	48 89 c7             	mov    %rax,%rdi
    2811:	31 f6                	xor    %esi,%esi
    2813:	4c 8b 64 24 40       	mov    0x40(%rsp),%r12
    2818:	4c 89 e2             	mov    %r12,%rdx
    281b:	48 8b 4c 24 60       	mov    0x60(%rsp),%rcx
    2820:	e8 cb f8 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    2825:	be 01 00 00 00       	mov    $0x1,%esi
    282a:	4c 89 ff             	mov    %r15,%rdi
    282d:	e8 6e f8 ff ff       	call   20a0 <omTensorSetDataType@plt>
    2832:	4c 89 ff             	mov    %r15,%rdi
    2835:	e8 96 f9 ff ff       	call   21d0 <omTensorGetShape@plt>
    283a:	48 89 c5             	mov    %rax,%rbp
    283d:	4c 89 ff             	mov    %r15,%rdi
    2840:	e8 2b f8 ff ff       	call   2070 <omTensorGetStrides@plt>
    2845:	4c 89 75 00          	mov    %r14,0x0(%rbp)
    2849:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    2850:	48 c7 45 08 08 00 00 	movq   $0x8,0x8(%rbp)
    2857:	00 
    2858:	48 c7 40 08 00 09 00 	movq   $0x900,0x8(%rax)
    285f:	00 
    2860:	48 c7 45 10 30 00 00 	movq   $0x30,0x10(%rbp)
    2867:	00 
    2868:	48 c7 40 10 30 00 00 	movq   $0x30,0x10(%rax)
    286f:	00 
    2870:	48 c7 45 18 30 00 00 	movq   $0x30,0x18(%rbp)
    2877:	00 
    2878:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    287f:	00 
    2880:	4c 89 ef             	mov    %r13,%rdi
    2883:	4c 89 fe             	mov    %r15,%rsi
    2886:	e8 65 f9 ff ff       	call   21f0 <Relu@plt>
    288b:	4c 89 ef             	mov    %r13,%rdi
    288e:	e8 1d f9 ff ff       	call   21b0 <omTensorDestroy@plt>
    2893:	4c 89 ff             	mov    %r15,%rdi
    2896:	e8 15 f9 ff ff       	call   21b0 <omTensorDestroy@plt>
    289b:	4c 89 e7             	mov    %r12,%rdi
    289e:	e8 6d f8 ff ff       	call   2110 <free@plt>
    28a3:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    28a8:	e8 23 f8 ff ff       	call   20d0 <malloc@plt>
    28ad:	49 89 c6             	mov    %rax,%r14
    28b0:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    28b4:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    28b9:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    28bd:	4c 89 7c 24 30       	mov    %r15,0x30(%rsp)
    28c2:	bf 04 00 00 00       	mov    $0x4,%edi
    28c7:	e8 d4 f8 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    28cc:	49 89 c4             	mov    %rax,%r12
    28cf:	48 89 c7             	mov    %rax,%rdi
    28d2:	31 f6                	xor    %esi,%esi
    28d4:	4c 89 f2             	mov    %r14,%rdx
    28d7:	4c 89 f9             	mov    %r15,%rcx
    28da:	e8 11 f8 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    28df:	be 01 00 00 00       	mov    $0x1,%esi
    28e4:	4c 89 64 24 40       	mov    %r12,0x40(%rsp)
    28e9:	4c 89 e7             	mov    %r12,%rdi
    28ec:	e8 af f7 ff ff       	call   20a0 <omTensorSetDataType@plt>
    28f1:	4c 89 e7             	mov    %r12,%rdi
    28f4:	e8 d7 f8 ff ff       	call   21d0 <omTensorGetShape@plt>
    28f9:	49 89 c5             	mov    %rax,%r13
    28fc:	4c 89 e7             	mov    %r12,%rdi
    28ff:	e8 6c f7 ff ff       	call   2070 <omTensorGetStrides@plt>
    2904:	4c 8b 74 24 20       	mov    0x20(%rsp),%r14
    2909:	4d 89 75 00          	mov    %r14,0x0(%r13)
    290d:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    2914:	49 c7 45 08 08 00 00 	movq   $0x8,0x8(%r13)
    291b:	00 
    291c:	48 c7 40 08 00 09 00 	movq   $0x900,0x8(%rax)
    2923:	00 
    2924:	49 c7 45 10 30 00 00 	movq   $0x30,0x10(%r13)
    292b:	00 
    292c:	48 c7 40 10 30 00 00 	movq   $0x30,0x10(%rax)
    2933:	00 
    2934:	49 c7 45 18 30 00 00 	movq   $0x30,0x18(%r13)
    293b:	00 
    293c:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    2943:	00 
    2944:	bf 04 00 00 00       	mov    $0x4,%edi
    2949:	e8 52 f8 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    294e:	49 89 c5             	mov    %rax,%r13
    2951:	48 89 c7             	mov    %rax,%rdi
    2954:	31 f6                	xor    %esi,%esi
    2956:	48 8b 14 24          	mov    (%rsp),%rdx
    295a:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    295f:	e8 8c f7 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    2964:	be 01 00 00 00       	mov    $0x1,%esi
    2969:	4c 89 ef             	mov    %r13,%rdi
    296c:	e8 2f f7 ff ff       	call   20a0 <omTensorSetDataType@plt>
    2971:	4c 89 ef             	mov    %r13,%rdi
    2974:	e8 57 f8 ff ff       	call   21d0 <omTensorGetShape@plt>
    2979:	49 89 c4             	mov    %rax,%r12
    297c:	4c 89 ef             	mov    %r13,%rdi
    297f:	e8 ec f6 ff ff       	call   2070 <omTensorGetStrides@plt>
    2984:	4d 89 34 24          	mov    %r14,(%r12)
    2988:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    298f:	49 c7 44 24 08 08 00 	movq   $0x8,0x8(%r12)
    2996:	00 00 
    2998:	48 c7 40 08 00 09 00 	movq   $0x900,0x8(%rax)
    299f:	00 
    29a0:	49 c7 44 24 10 30 00 	movq   $0x30,0x10(%r12)
    29a7:	00 00 
    29a9:	48 c7 40 10 30 00 00 	movq   $0x30,0x10(%rax)
    29b0:	00 
    29b1:	49 c7 44 24 18 30 00 	movq   $0x30,0x18(%r12)
    29b8:	00 00 
    29ba:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    29c1:	00 
    29c2:	48 8d 2d a7 71 01 00 	lea    0x171a7(%rip),%rbp        # 19b70 <constant_53_mobilenet>
    29c9:	bf 04 00 00 00       	mov    $0x4,%edi
    29ce:	e8 cd f7 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    29d3:	49 89 c4             	mov    %rax,%r12
    29d6:	48 89 c7             	mov    %rax,%rdi
    29d9:	31 f6                	xor    %esi,%esi
    29db:	48 89 ea             	mov    %rbp,%rdx
    29de:	48 89 e9             	mov    %rbp,%rcx
    29e1:	e8 0a f7 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    29e6:	be 01 00 00 00       	mov    $0x1,%esi
    29eb:	4c 89 e7             	mov    %r12,%rdi
    29ee:	e8 ad f6 ff ff       	call   20a0 <omTensorSetDataType@plt>
    29f3:	4c 89 e7             	mov    %r12,%rdi
    29f6:	e8 d5 f7 ff ff       	call   21d0 <omTensorGetShape@plt>
    29fb:	48 89 c5             	mov    %rax,%rbp
    29fe:	4c 89 e7             	mov    %r12,%rdi
    2a01:	e8 6a f6 ff ff       	call   2070 <omTensorGetStrides@plt>
    2a06:	48 c7 45 00 08 00 00 	movq   $0x8,0x0(%rbp)
    2a0d:	00 
    2a0e:	48 c7 00 09 00 00 00 	movq   $0x9,(%rax)
    2a15:	48 c7 45 08 01 00 00 	movq   $0x1,0x8(%rbp)
    2a1c:	00 
    2a1d:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    2a24:	00 
    2a25:	48 c7 45 10 03 00 00 	movq   $0x3,0x10(%rbp)
    2a2c:	00 
    2a2d:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    2a34:	00 
    2a35:	48 c7 45 18 03 00 00 	movq   $0x3,0x18(%rbp)
    2a3c:	00 
    2a3d:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    2a44:	00 
    2a45:	4c 8d 3d 64 72 01 00 	lea    0x17264(%rip),%r15        # 19cb0 <constant_51_mobilenet>
    2a4c:	bf 01 00 00 00       	mov    $0x1,%edi
    2a51:	e8 4a f7 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    2a56:	48 89 c5             	mov    %rax,%rbp
    2a59:	48 89 c7             	mov    %rax,%rdi
    2a5c:	31 f6                	xor    %esi,%esi
    2a5e:	4c 89 fa             	mov    %r15,%rdx
    2a61:	4c 89 f9             	mov    %r15,%rcx
    2a64:	e8 87 f6 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    2a69:	be 01 00 00 00       	mov    $0x1,%esi
    2a6e:	48 89 ef             	mov    %rbp,%rdi
    2a71:	e8 2a f6 ff ff       	call   20a0 <omTensorSetDataType@plt>
    2a76:	48 89 ef             	mov    %rbp,%rdi
    2a79:	e8 52 f7 ff ff       	call   21d0 <omTensorGetShape@plt>
    2a7e:	49 89 c7             	mov    %rax,%r15
    2a81:	48 89 ef             	mov    %rbp,%rdi
    2a84:	e8 e7 f5 ff ff       	call   2070 <omTensorGetStrides@plt>
    2a89:	49 c7 07 08 00 00 00 	movq   $0x8,(%r15)
    2a90:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    2a97:	4c 8b 7c 24 40       	mov    0x40(%rsp),%r15
    2a9c:	4c 89 ff             	mov    %r15,%rdi
    2a9f:	4c 89 ee             	mov    %r13,%rsi
    2aa2:	4c 89 e2             	mov    %r12,%rdx
    2aa5:	48 89 e9             	mov    %rbp,%rcx
    2aa8:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    2aae:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    2ab4:	6a 01                	push   $0x1
    2ab6:	6a 01                	push   $0x1
    2ab8:	6a 01                	push   $0x1
    2aba:	6a 01                	push   $0x1
    2abc:	e8 9f f6 ff ff       	call   2160 <Conv2D_bias@plt>
    2ac1:	48 83 c4 20          	add    $0x20,%rsp
    2ac5:	4c 89 ff             	mov    %r15,%rdi
    2ac8:	e8 e3 f6 ff ff       	call   21b0 <omTensorDestroy@plt>
    2acd:	4c 89 ef             	mov    %r13,%rdi
    2ad0:	e8 db f6 ff ff       	call   21b0 <omTensorDestroy@plt>
    2ad5:	4c 89 e7             	mov    %r12,%rdi
    2ad8:	e8 d3 f6 ff ff       	call   21b0 <omTensorDestroy@plt>
    2add:	48 89 ef             	mov    %rbp,%rdi
    2ae0:	e8 cb f6 ff ff       	call   21b0 <omTensorDestroy@plt>
    2ae5:	48 8b 3c 24          	mov    (%rsp),%rdi
    2ae9:	e8 22 f6 ff ff       	call   2110 <free@plt>
    2aee:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    2af3:	e8 d8 f5 ff ff       	call   20d0 <malloc@plt>
    2af8:	49 89 c7             	mov    %rax,%r15
    2afb:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    2aff:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    2b04:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    2b08:	4c 89 24 24          	mov    %r12,(%rsp)
    2b0c:	bf 04 00 00 00       	mov    $0x4,%edi
    2b11:	e8 8a f6 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    2b16:	49 89 c5             	mov    %rax,%r13
    2b19:	48 89 c7             	mov    %rax,%rdi
    2b1c:	31 f6                	xor    %esi,%esi
    2b1e:	4c 89 fa             	mov    %r15,%rdx
    2b21:	4c 89 e1             	mov    %r12,%rcx
    2b24:	e8 c7 f5 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    2b29:	be 01 00 00 00       	mov    $0x1,%esi
    2b2e:	4c 89 ef             	mov    %r13,%rdi
    2b31:	e8 6a f5 ff ff       	call   20a0 <omTensorSetDataType@plt>
    2b36:	4c 89 ef             	mov    %r13,%rdi
    2b39:	e8 92 f6 ff ff       	call   21d0 <omTensorGetShape@plt>
    2b3e:	49 89 c7             	mov    %rax,%r15
    2b41:	4c 89 ef             	mov    %r13,%rdi
    2b44:	e8 27 f5 ff ff       	call   2070 <omTensorGetStrides@plt>
    2b49:	4d 89 37             	mov    %r14,(%r15)
    2b4c:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    2b53:	49 c7 47 08 08 00 00 	movq   $0x8,0x8(%r15)
    2b5a:	00 
    2b5b:	48 c7 40 08 00 09 00 	movq   $0x900,0x8(%rax)
    2b62:	00 
    2b63:	49 c7 47 10 30 00 00 	movq   $0x30,0x10(%r15)
    2b6a:	00 
    2b6b:	48 c7 40 10 30 00 00 	movq   $0x30,0x10(%rax)
    2b72:	00 
    2b73:	49 c7 47 18 30 00 00 	movq   $0x30,0x18(%r15)
    2b7a:	00 
    2b7b:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    2b82:	00 
    2b83:	bf 04 00 00 00       	mov    $0x4,%edi
    2b88:	e8 13 f6 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    2b8d:	49 89 c7             	mov    %rax,%r15
    2b90:	48 89 c7             	mov    %rax,%rdi
    2b93:	31 f6                	xor    %esi,%esi
    2b95:	4c 8b 64 24 60       	mov    0x60(%rsp),%r12
    2b9a:	4c 89 e2             	mov    %r12,%rdx
    2b9d:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    2ba2:	e8 49 f5 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    2ba7:	be 01 00 00 00       	mov    $0x1,%esi
    2bac:	4c 89 ff             	mov    %r15,%rdi
    2baf:	e8 ec f4 ff ff       	call   20a0 <omTensorSetDataType@plt>
    2bb4:	4c 89 ff             	mov    %r15,%rdi
    2bb7:	e8 14 f6 ff ff       	call   21d0 <omTensorGetShape@plt>
    2bbc:	48 89 c5             	mov    %rax,%rbp
    2bbf:	4c 89 ff             	mov    %r15,%rdi
    2bc2:	e8 a9 f4 ff ff       	call   2070 <omTensorGetStrides@plt>
    2bc7:	4c 89 75 00          	mov    %r14,0x0(%rbp)
    2bcb:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    2bd2:	48 c7 45 08 08 00 00 	movq   $0x8,0x8(%rbp)
    2bd9:	00 
    2bda:	48 c7 40 08 00 09 00 	movq   $0x900,0x8(%rax)
    2be1:	00 
    2be2:	48 c7 45 10 30 00 00 	movq   $0x30,0x10(%rbp)
    2be9:	00 
    2bea:	48 c7 40 10 30 00 00 	movq   $0x30,0x10(%rax)
    2bf1:	00 
    2bf2:	48 c7 45 18 30 00 00 	movq   $0x30,0x18(%rbp)
    2bf9:	00 
    2bfa:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    2c01:	00 
    2c02:	4c 89 ef             	mov    %r13,%rdi
    2c05:	4c 89 fe             	mov    %r15,%rsi
    2c08:	e8 e3 f5 ff ff       	call   21f0 <Relu@plt>
    2c0d:	4c 89 ef             	mov    %r13,%rdi
    2c10:	e8 9b f5 ff ff       	call   21b0 <omTensorDestroy@plt>
    2c15:	4c 89 ff             	mov    %r15,%rdi
    2c18:	e8 93 f5 ff ff       	call   21b0 <omTensorDestroy@plt>
    2c1d:	4c 89 e7             	mov    %r12,%rdi
    2c20:	e8 eb f4 ff ff       	call   2110 <free@plt>
    2c25:	bf 18 00 00 00       	mov    $0x18,%edi
    2c2a:	e8 a1 f4 ff ff       	call   20d0 <malloc@plt>
    2c2f:	48 89 84 24 18 02 00 	mov    %rax,0x218(%rsp)
    2c36:	00 
    2c37:	48 83 c0 0f          	add    $0xf,%rax
    2c3b:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    2c3f:	48 89 84 24 10 02 00 	mov    %rax,0x210(%rsp)
    2c46:	00 
    2c47:	4c 89 30             	mov    %r14,(%rax)
    2c4a:	bf 28 00 00 00       	mov    $0x28,%edi
    2c4f:	e8 7c f4 ff ff       	call   20d0 <malloc@plt>
    2c54:	48 8d 48 0f          	lea    0xf(%rax),%rcx
    2c58:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
    2c5c:	4c 89 31             	mov    %r14,(%rcx)
    2c5f:	48 c7 41 08 08 00 00 	movq   $0x8,0x8(%rcx)
    2c66:	00 
    2c67:	48 c7 41 10 ff ff ff 	movq   $0xffffffffffffffff,0x10(%rcx)
    2c6e:	ff 
    2c6f:	48 89 c7             	mov    %rax,%rdi
    2c72:	e8 99 f4 ff ff       	call   2110 <free@plt>
    2c77:	48 8d 04 dd 00 00 00 	lea    0x0(,%rbx,8),%rax
    2c7e:	00 
    2c7f:	48 89 84 24 e0 00 00 	mov    %rax,0xe0(%rsp)
    2c86:	00 
    2c87:	49 89 df             	mov    %rbx,%r15
    2c8a:	49 c1 e7 04          	shl    $0x4,%r15
    2c8e:	4c 8b ac 24 e8 00 00 	mov    0xe8(%rsp),%r13
    2c95:	00 
    2c96:	4c 89 e8             	mov    %r13,%rax
    2c99:	49 0f af c7          	imul   %r15,%rax
    2c9d:	48 8d 3c 85 10 00 00 	lea    0x10(,%rax,4),%rdi
    2ca4:	00 
    2ca5:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
    2caa:	e8 21 f4 ff ff       	call   20d0 <malloc@plt>
    2caf:	49 89 c6             	mov    %rax,%r14
    2cb2:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    2cb6:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    2cba:	bf 03 00 00 00       	mov    $0x3,%edi
    2cbf:	e8 dc f4 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    2cc4:	48 89 c5             	mov    %rax,%rbp
    2cc7:	48 89 c7             	mov    %rax,%rdi
    2cca:	31 f6                	xor    %esi,%esi
    2ccc:	4c 89 b4 24 08 02 00 	mov    %r14,0x208(%rsp)
    2cd3:	00 
    2cd4:	4c 89 f2             	mov    %r14,%rdx
    2cd7:	4c 89 a4 24 00 02 00 	mov    %r12,0x200(%rsp)
    2cde:	00 
    2cdf:	4c 89 e1             	mov    %r12,%rcx
    2ce2:	e8 09 f4 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    2ce7:	be 01 00 00 00       	mov    $0x1,%esi
    2cec:	48 89 ef             	mov    %rbp,%rdi
    2cef:	e8 ac f3 ff ff       	call   20a0 <omTensorSetDataType@plt>
    2cf4:	48 89 ef             	mov    %rbp,%rdi
    2cf7:	e8 d4 f4 ff ff       	call   21d0 <omTensorGetShape@plt>
    2cfc:	49 89 c6             	mov    %rax,%r14
    2cff:	48 89 ef             	mov    %rbp,%rdi
    2d02:	e8 69 f3 ff ff       	call   2070 <omTensorGetStrides@plt>
    2d07:	4d 89 2e             	mov    %r13,(%r14)
    2d0a:	4c 89 bc 24 38 01 00 	mov    %r15,0x138(%rsp)
    2d11:	00 
    2d12:	4c 89 38             	mov    %r15,(%rax)
    2d15:	49 c7 46 08 10 00 00 	movq   $0x10,0x8(%r14)
    2d1c:	00 
    2d1d:	48 89 58 08          	mov    %rbx,0x8(%rax)
    2d21:	49 89 5e 10          	mov    %rbx,0x10(%r14)
    2d25:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    2d2c:	00 
    2d2d:	bf 02 00 00 00       	mov    $0x2,%edi
    2d32:	e8 69 f4 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    2d37:	49 89 c6             	mov    %rax,%r14
    2d3a:	48 8d 0d cf 91 0d 00 	lea    0xd91cf(%rip),%rcx        # dbf10 <constant_11_mobilenet>
    2d41:	48 89 c7             	mov    %rax,%rdi
    2d44:	31 f6                	xor    %esi,%esi
    2d46:	48 89 ca             	mov    %rcx,%rdx
    2d49:	e8 a2 f3 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    2d4e:	be 01 00 00 00       	mov    $0x1,%esi
    2d53:	4c 89 f7             	mov    %r14,%rdi
    2d56:	e8 45 f3 ff ff       	call   20a0 <omTensorSetDataType@plt>
    2d5b:	4c 89 f7             	mov    %r14,%rdi
    2d5e:	e8 6d f4 ff ff       	call   21d0 <omTensorGetShape@plt>
    2d63:	49 89 c7             	mov    %rax,%r15
    2d66:	4c 89 f7             	mov    %r14,%rdi
    2d69:	e8 02 f3 ff ff       	call   2070 <omTensorGetStrides@plt>
    2d6e:	49 c7 07 10 00 00 00 	movq   $0x10,(%r15)
    2d75:	48 c7 00 08 00 00 00 	movq   $0x8,(%rax)
    2d7c:	49 c7 47 08 08 00 00 	movq   $0x8,0x8(%r15)
    2d83:	00 
    2d84:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    2d8b:	00 
    2d8c:	bf 03 00 00 00       	mov    $0x3,%edi
    2d91:	e8 0a f4 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    2d96:	49 89 c7             	mov    %rax,%r15
    2d99:	48 89 c7             	mov    %rax,%rdi
    2d9c:	31 f6                	xor    %esi,%esi
    2d9e:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    2da3:	48 8b 0c 24          	mov    (%rsp),%rcx
    2da7:	e8 44 f3 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    2dac:	be 01 00 00 00       	mov    $0x1,%esi
    2db1:	4c 89 ff             	mov    %r15,%rdi
    2db4:	e8 e7 f2 ff ff       	call   20a0 <omTensorSetDataType@plt>
    2db9:	4c 89 ff             	mov    %r15,%rdi
    2dbc:	e8 0f f4 ff ff       	call   21d0 <omTensorGetShape@plt>
    2dc1:	49 89 c4             	mov    %rax,%r12
    2dc4:	4c 89 ff             	mov    %r15,%rdi
    2dc7:	e8 a4 f2 ff ff       	call   2070 <omTensorGetStrides@plt>
    2dcc:	4d 89 2c 24          	mov    %r13,(%r12)
    2dd0:	48 8b 8c 24 e0 00 00 	mov    0xe0(%rsp),%rcx
    2dd7:	00 
    2dd8:	48 89 08             	mov    %rcx,(%rax)
    2ddb:	49 c7 44 24 08 08 00 	movq   $0x8,0x8(%r12)
    2de2:	00 00 
    2de4:	48 89 58 08          	mov    %rbx,0x8(%rax)
    2de8:	49 89 5c 24 10       	mov    %rbx,0x10(%r12)
    2ded:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    2df4:	00 
    2df5:	48 89 ef             	mov    %rbp,%rdi
    2df8:	4c 89 f6             	mov    %r14,%rsi
    2dfb:	4c 89 fa             	mov    %r15,%rdx
    2dfe:	e8 4d f2 ff ff       	call   2050 <MatMul@plt>
    2e03:	48 89 ef             	mov    %rbp,%rdi
    2e06:	e8 a5 f3 ff ff       	call   21b0 <omTensorDestroy@plt>
    2e0b:	4c 89 f7             	mov    %r14,%rdi
    2e0e:	e8 9d f3 ff ff       	call   21b0 <omTensorDestroy@plt>
    2e13:	4c 89 ff             	mov    %r15,%rdi
    2e16:	e8 95 f3 ff ff       	call   21b0 <omTensorDestroy@plt>
    2e1b:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    2e20:	e8 eb f2 ff ff       	call   2110 <free@plt>
    2e25:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    2e2a:	e8 a1 f2 ff ff       	call   20d0 <malloc@plt>
    2e2f:	48 89 84 24 20 02 00 	mov    %rax,0x220(%rsp)
    2e36:	00 
    2e37:	48 83 c0 0f          	add    $0xf,%rax
    2e3b:	48 89 84 24 f8 01 00 	mov    %rax,0x1f8(%rsp)
    2e42:	00 
    2e43:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    2e47:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
    2e4e:	00 
    2e4f:	4d 85 ed             	test   %r13,%r13
    2e52:	0f 8e f6 12 00 00    	jle    414e <main_graph_mobilenet+0x1e7e>
    2e58:	48 8d 04 9d 00 00 00 	lea    0x0(,%rbx,4),%rax
    2e5f:	00 
    2e60:	48 8d 0c 40          	lea    (%rax,%rax,2),%rcx
    2e64:	48 89 8c 24 30 01 00 	mov    %rcx,0x130(%rsp)
    2e6b:	00 
    2e6c:	48 6b cb 3c          	imul   $0x3c,%rbx,%rcx
    2e70:	48 8b b4 24 20 01 00 	mov    0x120(%rsp),%rsi
    2e77:	00 
    2e78:	48 8d 14 0e          	lea    (%rsi,%rcx,1),%rdx
    2e7c:	48 89 94 24 f0 01 00 	mov    %rdx,0x1f0(%rsp)
    2e83:	00 
    2e84:	48 89 df             	mov    %rbx,%rdi
    2e87:	48 c1 e7 06          	shl    $0x6,%rdi
    2e8b:	4c 8b 84 24 00 02 00 	mov    0x200(%rsp),%r8
    2e92:	00 
    2e93:	48 89 8c 24 78 02 00 	mov    %rcx,0x278(%rsp)
    2e9a:	00 
    2e9b:	4c 01 c1             	add    %r8,%rcx
    2e9e:	48 89 8c 24 e8 01 00 	mov    %rcx,0x1e8(%rsp)
    2ea5:	00 
    2ea6:	48 6b cb 38          	imul   $0x38,%rbx,%rcx
    2eaa:	48 8d 14 0e          	lea    (%rsi,%rcx,1),%rdx
    2eae:	48 89 94 24 e0 01 00 	mov    %rdx,0x1e0(%rsp)
    2eb5:	00 
    2eb6:	48 89 8c 24 70 02 00 	mov    %rcx,0x270(%rsp)
    2ebd:	00 
    2ebe:	4c 01 c1             	add    %r8,%rcx
    2ec1:	48 89 8c 24 d8 01 00 	mov    %rcx,0x1d8(%rsp)
    2ec8:	00 
    2ec9:	48 6b d3 34          	imul   $0x34,%rbx,%rdx
    2ecd:	48 8d 0c 16          	lea    (%rsi,%rdx,1),%rcx
    2ed1:	48 89 8c 24 d0 01 00 	mov    %rcx,0x1d0(%rsp)
    2ed8:	00 
    2ed9:	4c 8d 24 c0          	lea    (%rax,%rax,8),%r12
    2edd:	49 89 df             	mov    %rbx,%r15
    2ee0:	49 c1 e7 05          	shl    $0x5,%r15
    2ee4:	48 8d 0c db          	lea    (%rbx,%rbx,8),%rcx
    2ee8:	4c 8d 34 49          	lea    (%rcx,%rcx,2),%r14
    2eec:	49 01 de             	add    %rbx,%r14
    2eef:	48 8d 04 80          	lea    (%rax,%rax,4),%rax
    2ef3:	48 89 84 24 28 01 00 	mov    %rax,0x128(%rsp)
    2efa:	00 
    2efb:	49 89 db             	mov    %rbx,%r11
    2efe:	49 83 e3 f8          	and    $0xfffffffffffffff8,%r11
    2f02:	f3 0f 10 0d 16 b2 00 	movss  0xb216(%rip),%xmm1        # e120 <_fini+0x1a4>
    2f09:	00 
    2f0a:	f3 0f 10 15 16 b2 00 	movss  0xb216(%rip),%xmm2        # e128 <_fini+0x1ac>
    2f11:	00 
    2f12:	f3 0f 10 25 12 b2 00 	movss  0xb212(%rip),%xmm4        # e12c <_fini+0x1b0>
    2f19:	00 
    2f1a:	f3 44 0f 10 0d 1d b2 	movss  0xb21d(%rip),%xmm9        # e140 <_fini+0x1c4>
    2f21:	00 00 
    2f23:	f3 44 0f 10 15 1c b2 	movss  0xb21c(%rip),%xmm10        # e148 <_fini+0x1cc>
    2f2a:	00 00 
    2f2c:	f3 44 0f 10 1d 17 b2 	movss  0xb217(%rip),%xmm11        # e14c <_fini+0x1d0>
    2f33:	00 00 
    2f35:	f3 44 0f 10 25 12 b2 	movss  0xb212(%rip),%xmm12        # e150 <_fini+0x1d4>
    2f3c:	00 00 
    2f3e:	f3 44 0f 10 2d 0d b2 	movss  0xb20d(%rip),%xmm13        # e154 <_fini+0x1d8>
    2f45:	00 00 
    2f47:	f3 44 0f 10 35 08 b2 	movss  0xb208(%rip),%xmm14        # e158 <_fini+0x1dc>
    2f4e:	00 00 
    2f50:	48 89 94 24 68 02 00 	mov    %rdx,0x268(%rsp)
    2f57:	00 
    2f58:	49 8d 04 10          	lea    (%r8,%rdx,1),%rax
    2f5c:	48 89 84 24 c8 01 00 	mov    %rax,0x1c8(%rsp)
    2f63:	00 
    2f64:	48 8b 94 24 38 01 00 	mov    0x138(%rsp),%rdx
    2f6b:	00 
    2f6c:	4c 8d 14 52          	lea    (%rdx,%rdx,2),%r10
    2f70:	48 8b 8c 24 f8 01 00 	mov    0x1f8(%rsp),%rcx
    2f77:	00 
    2f78:	4e 8d 0c 11          	lea    (%rcx,%r10,1),%r9
    2f7c:	4c 89 8c 24 b0 02 00 	mov    %r9,0x2b0(%rsp)
    2f83:	00 
    2f84:	4b 8d 04 10          	lea    (%r8,%r10,1),%rax
    2f88:	48 89 84 24 c0 01 00 	mov    %rax,0x1c0(%rsp)
    2f8f:	00 
    2f90:	4c 6b cb 2c          	imul   $0x2c,%rbx,%r9
    2f94:	4a 8d 04 0e          	lea    (%rsi,%r9,1),%rax
    2f98:	48 89 84 24 b8 01 00 	mov    %rax,0x1b8(%rsp)
    2f9f:	00 
    2fa0:	4c 89 8c 24 40 02 00 	mov    %r9,0x240(%rsp)
    2fa7:	00 
    2fa8:	4b 8d 04 08          	lea    (%r8,%r9,1),%rax
    2fac:	48 89 84 24 b0 01 00 	mov    %rax,0x1b0(%rsp)
    2fb3:	00 
    2fb4:	4c 8b 8c 24 e0 00 00 	mov    0xe0(%rsp),%r9
    2fbb:	00 
    2fbc:	4b 8d 04 89          	lea    (%r9,%r9,4),%rax
    2fc0:	4c 8d 2c 06          	lea    (%rsi,%rax,1),%r13
    2fc4:	4c 89 ac 24 a8 01 00 	mov    %r13,0x1a8(%rsp)
    2fcb:	00 
    2fcc:	48 89 84 24 38 02 00 	mov    %rax,0x238(%rsp)
    2fd3:	00 
    2fd4:	49 8d 04 00          	lea    (%r8,%rax,1),%rax
    2fd8:	48 89 84 24 a0 01 00 	mov    %rax,0x1a0(%rsp)
    2fdf:	00 
    2fe0:	4a 8d 04 26          	lea    (%rsi,%r12,1),%rax
    2fe4:	48 89 84 24 98 01 00 	mov    %rax,0x198(%rsp)
    2feb:	00 
    2fec:	4c 89 a4 24 60 02 00 	mov    %r12,0x260(%rsp)
    2ff3:	00 
    2ff4:	4f 8d 24 20          	lea    (%r8,%r12,1),%r12
    2ff8:	4a 8d 2c 39          	lea    (%rcx,%r15,1),%rbp
    2ffc:	48 89 ac 24 a0 02 00 	mov    %rbp,0x2a0(%rsp)
    3003:	00 
    3004:	4f 8d 2c 38          	lea    (%r8,%r15,1),%r13
    3008:	4a 8d 04 36          	lea    (%rsi,%r14,1),%rax
    300c:	48 89 84 24 90 01 00 	mov    %rax,0x190(%rsp)
    3013:	00 
    3014:	4c 89 b4 24 50 02 00 	mov    %r14,0x250(%rsp)
    301b:	00 
    301c:	4f 8d 34 30          	lea    (%r8,%r14,1),%r14
    3020:	4f 8d 0c 49          	lea    (%r9,%r9,2),%r9
    3024:	4a 8d 04 0e          	lea    (%rsi,%r9,1),%rax
    3028:	48 89 84 24 88 01 00 	mov    %rax,0x188(%rsp)
    302f:	00 
    3030:	4c 89 8c 24 30 02 00 	mov    %r9,0x230(%rsp)
    3037:	00 
    3038:	4f 8d 0c 08          	lea    (%r8,%r9,1),%r9
    303c:	48 8b ac 24 28 01 00 	mov    0x128(%rsp),%rbp
    3043:	00 
    3044:	48 8d 04 2e          	lea    (%rsi,%rbp,1),%rax
    3048:	48 89 84 24 80 01 00 	mov    %rax,0x180(%rsp)
    304f:	00 
    3050:	49 8d 04 28          	lea    (%r8,%rbp,1),%rax
    3054:	48 89 84 24 78 01 00 	mov    %rax,0x178(%rsp)
    305b:	00 
    305c:	48 8d 0c 11          	lea    (%rcx,%rdx,1),%rcx
    3060:	48 89 8c 24 80 02 00 	mov    %rcx,0x280(%rsp)
    3067:	00 
    3068:	49 8d 04 10          	lea    (%r8,%rdx,1),%rax
    306c:	48 89 84 24 70 01 00 	mov    %rax,0x170(%rsp)
    3073:	00 
    3074:	48 8b 8c 24 30 01 00 	mov    0x130(%rsp),%rcx
    307b:	00 
    307c:	48 8d 2c 0e          	lea    (%rsi,%rcx,1),%rbp
    3080:	48 89 ac 24 68 01 00 	mov    %rbp,0x168(%rsp)
    3087:	00 
    3088:	49 8d 0c 08          	lea    (%r8,%rcx,1),%rcx
    308c:	48 89 8c 24 60 01 00 	mov    %rcx,0x160(%rsp)
    3093:	00 
    3094:	48 8d 0c de          	lea    (%rsi,%rbx,8),%rcx
    3098:	48 89 8c 24 58 01 00 	mov    %rcx,0x158(%rsp)
    309f:	00 
    30a0:	49 8d 0c d8          	lea    (%r8,%rbx,8),%rcx
    30a4:	48 89 8c 24 50 01 00 	mov    %rcx,0x150(%rsp)
    30ab:	00 
    30ac:	48 8d 0c 9e          	lea    (%rsi,%rbx,4),%rcx
    30b0:	48 89 8c 24 48 01 00 	mov    %rcx,0x148(%rsp)
    30b7:	00 
    30b8:	49 8d 0c 98          	lea    (%r8,%rbx,4),%rcx
    30bc:	48 89 8c 24 40 01 00 	mov    %rcx,0x140(%rsp)
    30c3:	00 
    30c4:	48 8d 4e 10          	lea    0x10(%rsi),%rcx
    30c8:	49 8d 68 10          	lea    0x10(%r8),%rbp
    30cc:	48 8d 14 16          	lea    (%rsi,%rdx,1),%rdx
    30d0:	4c 89 bc 24 58 02 00 	mov    %r15,0x258(%rsp)
    30d7:	00 
    30d8:	4e 8d 3c 3e          	lea    (%rsi,%r15,1),%r15
    30dc:	4c 89 7c 24 30       	mov    %r15,0x30(%rsp)
    30e1:	4c 89 94 24 48 02 00 	mov    %r10,0x248(%rsp)
    30e8:	00 
    30e9:	4e 8d 14 16          	lea    (%rsi,%r10,1),%r10
    30ed:	4c 89 54 24 60       	mov    %r10,0x60(%rsp)
    30f2:	4c 8b 94 24 e8 01 00 	mov    0x1e8(%rsp),%r10
    30f9:	00 
    30fa:	4c 89 54 24 10       	mov    %r10,0x10(%rsp)
    30ff:	4c 8b 94 24 f0 01 00 	mov    0x1f0(%rsp),%r10
    3106:	00 
    3107:	4c 89 14 24          	mov    %r10,(%rsp)
    310b:	4c 8b 94 24 d8 01 00 	mov    0x1d8(%rsp),%r10
    3112:	00 
    3113:	4c 89 94 24 18 01 00 	mov    %r10,0x118(%rsp)
    311a:	00 
    311b:	4c 8b 94 24 e0 01 00 	mov    0x1e0(%rsp),%r10
    3122:	00 
    3123:	4c 89 94 24 10 01 00 	mov    %r10,0x110(%rsp)
    312a:	00 
    312b:	48 8b 84 24 c8 01 00 	mov    0x1c8(%rsp),%rax
    3132:	00 
    3133:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    3138:	4c 8b 94 24 d0 01 00 	mov    0x1d0(%rsp),%r10
    313f:	00 
    3140:	4c 89 54 24 08       	mov    %r10,0x8(%rsp)
    3145:	48 8b 84 24 c0 01 00 	mov    0x1c0(%rsp),%rax
    314c:	00 
    314d:	48 89 84 24 08 01 00 	mov    %rax,0x108(%rsp)
    3154:	00 
    3155:	48 8b 84 24 b0 01 00 	mov    0x1b0(%rsp),%rax
    315c:	00 
    315d:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    3162:	48 8b 84 24 b8 01 00 	mov    0x1b8(%rsp),%rax
    3169:	00 
    316a:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    316f:	48 8b 84 24 a0 01 00 	mov    0x1a0(%rsp),%rax
    3176:	00 
    3177:	48 89 84 24 00 01 00 	mov    %rax,0x100(%rsp)
    317e:	00 
    317f:	48 8b 84 24 a8 01 00 	mov    0x1a8(%rsp),%rax
    3186:	00 
    3187:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    318c:	4c 89 a4 24 a8 02 00 	mov    %r12,0x2a8(%rsp)
    3193:	00 
    3194:	4c 89 64 24 70       	mov    %r12,0x70(%rsp)
    3199:	48 8b 84 24 98 01 00 	mov    0x198(%rsp),%rax
    31a0:	00 
    31a1:	48 89 84 24 98 00 00 	mov    %rax,0x98(%rsp)
    31a8:	00 
    31a9:	4c 89 ac 24 98 02 00 	mov    %r13,0x298(%rsp)
    31b0:	00 
    31b1:	4c 89 b4 24 90 02 00 	mov    %r14,0x290(%rsp)
    31b8:	00 
    31b9:	4c 89 b4 24 90 00 00 	mov    %r14,0x90(%rsp)
    31c0:	00 
    31c1:	49 89 d6             	mov    %rdx,%r14
    31c4:	48 8b 84 24 90 01 00 	mov    0x190(%rsp),%rax
    31cb:	00 
    31cc:	48 89 84 24 88 00 00 	mov    %rax,0x88(%rsp)
    31d3:	00 
    31d4:	4c 89 8c 24 88 02 00 	mov    %r9,0x288(%rsp)
    31db:	00 
    31dc:	4c 89 c8             	mov    %r9,%rax
    31df:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    31e5:	48 8b 94 24 88 01 00 	mov    0x188(%rsp),%rdx
    31ec:	00 
    31ed:	48 89 94 24 d8 00 00 	mov    %rdx,0xd8(%rsp)
    31f4:	00 
    31f5:	48 8b 94 24 78 01 00 	mov    0x178(%rsp),%rdx
    31fc:	00 
    31fd:	48 89 94 24 d0 00 00 	mov    %rdx,0xd0(%rsp)
    3204:	00 
    3205:	48 8b 94 24 80 01 00 	mov    0x180(%rsp),%rdx
    320c:	00 
    320d:	48 89 94 24 c8 00 00 	mov    %rdx,0xc8(%rsp)
    3214:	00 
    3215:	4c 8b 94 24 70 01 00 	mov    0x170(%rsp),%r10
    321c:	00 
    321d:	48 8b 94 24 60 01 00 	mov    0x160(%rsp),%rdx
    3224:	00 
    3225:	48 89 94 24 c0 00 00 	mov    %rdx,0xc0(%rsp)
    322c:	00 
    322d:	48 8b 94 24 68 01 00 	mov    0x168(%rsp),%rdx
    3234:	00 
    3235:	48 89 94 24 b8 00 00 	mov    %rdx,0xb8(%rsp)
    323c:	00 
    323d:	48 8b 94 24 50 01 00 	mov    0x150(%rsp),%rdx
    3244:	00 
    3245:	4c 8b bc 24 58 01 00 	mov    0x158(%rsp),%r15
    324c:	00 
    324d:	4c 89 bc 24 b0 00 00 	mov    %r15,0xb0(%rsp)
    3254:	00 
    3255:	4c 8b bc 24 40 01 00 	mov    0x140(%rsp),%r15
    325c:	00 
    325d:	4c 89 bc 24 a8 00 00 	mov    %r15,0xa8(%rsp)
    3264:	00 
    3265:	4c 8b bc 24 48 01 00 	mov    0x148(%rsp),%r15
    326c:	00 
    326d:	4c 89 bc 24 a0 00 00 	mov    %r15,0xa0(%rsp)
    3274:	00 
    3275:	48 89 b4 24 f8 00 00 	mov    %rsi,0xf8(%rsp)
    327c:	00 
    327d:	4c 89 c6             	mov    %r8,%rsi
    3280:	e9 eb 00 00 00       	jmp    3370 <main_graph_mobilenet+0x10a0>
    3285:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    328c:	00 00 00 
    328f:	90                   	nop
    3290:	49 ff c1             	inc    %r9
    3293:	48 01 f9             	add    %rdi,%rcx
    3296:	48 01 fd             	add    %rdi,%rbp
    3299:	48 01 bc 24 f8 00 00 	add    %rdi,0xf8(%rsp)
    32a0:	00 
    32a1:	48 01 fe             	add    %rdi,%rsi
    32a4:	48 01 bc 24 a0 00 00 	add    %rdi,0xa0(%rsp)
    32ab:	00 
    32ac:	48 01 bc 24 a8 00 00 	add    %rdi,0xa8(%rsp)
    32b3:	00 
    32b4:	48 01 bc 24 b0 00 00 	add    %rdi,0xb0(%rsp)
    32bb:	00 
    32bc:	48 01 fa             	add    %rdi,%rdx
    32bf:	48 01 bc 24 b8 00 00 	add    %rdi,0xb8(%rsp)
    32c6:	00 
    32c7:	48 01 bc 24 c0 00 00 	add    %rdi,0xc0(%rsp)
    32ce:	00 
    32cf:	49 01 fe             	add    %rdi,%r14
    32d2:	49 01 fa             	add    %rdi,%r10
    32d5:	48 01 bc 24 c8 00 00 	add    %rdi,0xc8(%rsp)
    32dc:	00 
    32dd:	48 01 bc 24 d0 00 00 	add    %rdi,0xd0(%rsp)
    32e4:	00 
    32e5:	48 01 bc 24 d8 00 00 	add    %rdi,0xd8(%rsp)
    32ec:	00 
    32ed:	48 01 f8             	add    %rdi,%rax
    32f0:	48 01 bc 24 88 00 00 	add    %rdi,0x88(%rsp)
    32f7:	00 
    32f8:	48 01 bc 24 90 00 00 	add    %rdi,0x90(%rsp)
    32ff:	00 
    3300:	48 01 7c 24 30       	add    %rdi,0x30(%rsp)
    3305:	49 01 fd             	add    %rdi,%r13
    3308:	48 01 bc 24 98 00 00 	add    %rdi,0x98(%rsp)
    330f:	00 
    3310:	48 01 7c 24 70       	add    %rdi,0x70(%rsp)
    3315:	48 01 7c 24 78       	add    %rdi,0x78(%rsp)
    331a:	48 01 bc 24 00 01 00 	add    %rdi,0x100(%rsp)
    3321:	00 
    3322:	48 01 7c 24 58       	add    %rdi,0x58(%rsp)
    3327:	48 01 7c 24 50       	add    %rdi,0x50(%rsp)
    332c:	48 01 7c 24 60       	add    %rdi,0x60(%rsp)
    3331:	48 01 bc 24 08 01 00 	add    %rdi,0x108(%rsp)
    3338:	00 
    3339:	48 01 7c 24 08       	add    %rdi,0x8(%rsp)
    333e:	48 01 7c 24 40       	add    %rdi,0x40(%rsp)
    3343:	48 01 bc 24 10 01 00 	add    %rdi,0x110(%rsp)
    334a:	00 
    334b:	48 01 bc 24 18 01 00 	add    %rdi,0x118(%rsp)
    3352:	00 
    3353:	49 01 fc             	add    %rdi,%r12
    3356:	4c 89 24 24          	mov    %r12,(%rsp)
    335a:	49 01 ff             	add    %rdi,%r15
    335d:	4c 89 7c 24 10       	mov    %r15,0x10(%rsp)
    3362:	4c 3b 8c 24 e8 00 00 	cmp    0xe8(%rsp),%r9
    3369:	00 
    336a:	0f 84 de 0d 00 00    	je     414e <main_graph_mobilenet+0x1e7e>
    3370:	48 85 db             	test   %rbx,%rbx
    3373:	4c 8b 7c 24 10       	mov    0x10(%rsp),%r15
    3378:	4c 8b 24 24          	mov    (%rsp),%r12
    337c:	0f 8e 0e ff ff ff    	jle    3290 <main_graph_mobilenet+0xfc0>
    3382:	4c 89 74 24 20       	mov    %r14,0x20(%rsp)
    3387:	49 89 f8             	mov    %rdi,%r8
    338a:	4c 89 8c 24 f0 00 00 	mov    %r9,0xf0(%rsp)
    3391:	00 
    3392:	4d 0f af c1          	imul   %r9,%r8
    3396:	48 83 fb 08          	cmp    $0x8,%rbx
    339a:	0f 82 88 00 00 00    	jb     3428 <main_graph_mobilenet+0x1158>
    33a0:	4c 8b 8c 24 00 02 00 	mov    0x200(%rsp),%r9
    33a7:	00 
    33a8:	4d 01 c1             	add    %r8,%r9
    33ab:	4c 8b b4 24 f8 01 00 	mov    0x1f8(%rsp),%r14
    33b2:	00 
    33b3:	4d 01 c6             	add    %r8,%r14
    33b6:	4d 29 ce             	sub    %r9,%r14
    33b9:	49 83 fe 20          	cmp    $0x20,%r14
    33bd:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    33c3:	4c 8b bc 24 08 01 00 	mov    0x108(%rsp),%r15
    33ca:	00 
    33cb:	4c 8b a4 24 00 01 00 	mov    0x100(%rsp),%r12
    33d2:	00 
    33d3:	4c 8b b4 24 f8 00 00 	mov    0xf8(%rsp),%r14
    33da:	00 
    33db:	72 73                	jb     3450 <main_graph_mobilenet+0x1180>
    33dd:	45 31 c9             	xor    %r9d,%r9d
    33e0:	0f 28 1d 19 ac 00 00 	movaps 0xac19(%rip),%xmm3        # e000 <_fini+0x84>
    33e7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    33ee:	00 00 
    33f0:	42 0f 28 44 8d f0    	movaps -0x10(%rbp,%r9,4),%xmm0
    33f6:	0f 58 c3             	addps  %xmm3,%xmm0
    33f9:	42 0f 28 4c 8d 00    	movaps 0x0(%rbp,%r9,4),%xmm1
    33ff:	0f 58 cb             	addps  %xmm3,%xmm1
    3402:	42 0f 29 44 89 f0    	movaps %xmm0,-0x10(%rcx,%r9,4)
    3408:	42 0f 29 0c 89       	movaps %xmm1,(%rcx,%r9,4)
    340d:	49 83 c1 08          	add    $0x8,%r9
    3411:	4d 39 cb             	cmp    %r9,%r11
    3414:	75 da                	jne    33f0 <main_graph_mobilenet+0x1120>
    3416:	4c 39 db             	cmp    %r11,%rbx
    3419:	4d 89 d9             	mov    %r11,%r9
    341c:	f3 0f 10 0d fc ac 00 	movss  0xacfc(%rip),%xmm1        # e120 <_fini+0x1a4>
    3423:	00 
    3424:	75 2a                	jne    3450 <main_graph_mobilenet+0x1180>
    3426:	eb 40                	jmp    3468 <main_graph_mobilenet+0x1198>
    3428:	45 31 c9             	xor    %r9d,%r9d
    342b:	4c 8b bc 24 08 01 00 	mov    0x108(%rsp),%r15
    3432:	00 
    3433:	4c 8b a4 24 00 01 00 	mov    0x100(%rsp),%r12
    343a:	00 
    343b:	4c 8b b4 24 f8 00 00 	mov    0xf8(%rsp),%r14
    3442:	00 
    3443:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    344a:	00 00 00 
    344d:	0f 1f 00             	nopl   (%rax)
    3450:	f3 42 0f 10 04 8e    	movss  (%rsi,%r9,4),%xmm0
    3456:	f3 0f 58 c1          	addss  %xmm1,%xmm0
    345a:	f3 43 0f 11 04 8e    	movss  %xmm0,(%r14,%r9,4)
    3460:	49 ff c1             	inc    %r9
    3463:	4c 39 cb             	cmp    %r9,%rbx
    3466:	75 e8                	jne    3450 <main_graph_mobilenet+0x1180>
    3468:	48 89 b4 24 80 00 00 	mov    %rsi,0x80(%rsp)
    346f:	00 
    3470:	45 0f 28 fe          	movaps %xmm14,%xmm15
    3474:	45 0f 28 f5          	movaps %xmm13,%xmm14
    3478:	45 0f 28 ec          	movaps %xmm12,%xmm13
    347c:	45 0f 28 e3          	movaps %xmm11,%xmm12
    3480:	45 0f 28 da          	movaps %xmm10,%xmm11
    3484:	0f 28 da             	movaps %xmm2,%xmm3
    3487:	48 83 fb 08          	cmp    $0x8,%rbx
    348b:	0f 82 a0 00 00 00    	jb     3531 <main_graph_mobilenet+0x1261>
    3491:	4c 8b 8c 24 48 01 00 	mov    0x148(%rsp),%r9
    3498:	00 
    3499:	4d 01 c1             	add    %r8,%r9
    349c:	4c 8b b4 24 40 01 00 	mov    0x140(%rsp),%r14
    34a3:	00 
    34a4:	4d 01 c6             	add    %r8,%r14
    34a7:	4d 29 f1             	sub    %r14,%r9
    34aa:	49 83 f9 20          	cmp    $0x20,%r9
    34ae:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    34b4:	f3 0f 10 15 68 ac 00 	movss  0xac68(%rip),%xmm2        # e124 <_fini+0x1a8>
    34bb:	00 
    34bc:	4c 8b b4 24 a8 00 00 	mov    0xa8(%rsp),%r14
    34c3:	00 
    34c4:	48 8b b4 24 a0 00 00 	mov    0xa0(%rsp),%rsi
    34cb:	00 
    34cc:	0f 82 7e 00 00 00    	jb     3550 <main_graph_mobilenet+0x1280>
    34d2:	4d 89 d9             	mov    %r11,%r9
    34d5:	49 89 de             	mov    %rbx,%r14
    34d8:	0f 28 15 31 ab 00 00 	movaps 0xab31(%rip),%xmm2        # e010 <_fini+0x94>
    34df:	90                   	nop
    34e0:	42 0f 10 44 b5 f0    	movups -0x10(%rbp,%r14,4),%xmm0
    34e6:	42 0f 10 4c b5 00    	movups 0x0(%rbp,%r14,4),%xmm1
    34ec:	0f 58 c2             	addps  %xmm2,%xmm0
    34ef:	0f 58 ca             	addps  %xmm2,%xmm1
    34f2:	42 0f 11 44 b1 f0    	movups %xmm0,-0x10(%rcx,%r14,4)
    34f8:	42 0f 11 0c b1       	movups %xmm1,(%rcx,%r14,4)
    34fd:	49 83 c6 08          	add    $0x8,%r14
    3501:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    3505:	75 d9                	jne    34e0 <main_graph_mobilenet+0x1210>
    3507:	4c 39 db             	cmp    %r11,%rbx
    350a:	4d 89 d9             	mov    %r11,%r9
    350d:	f3 0f 10 0d 0b ac 00 	movss  0xac0b(%rip),%xmm1        # e120 <_fini+0x1a4>
    3514:	00 
    3515:	f3 0f 10 15 07 ac 00 	movss  0xac07(%rip),%xmm2        # e124 <_fini+0x1a8>
    351c:	00 
    351d:	4c 8b b4 24 a8 00 00 	mov    0xa8(%rsp),%r14
    3524:	00 
    3525:	48 8b b4 24 a0 00 00 	mov    0xa0(%rsp),%rsi
    352c:	00 
    352d:	75 21                	jne    3550 <main_graph_mobilenet+0x1280>
    352f:	eb 37                	jmp    3568 <main_graph_mobilenet+0x1298>
    3531:	45 31 c9             	xor    %r9d,%r9d
    3534:	f3 0f 10 15 e8 ab 00 	movss  0xabe8(%rip),%xmm2        # e124 <_fini+0x1a8>
    353b:	00 
    353c:	4c 8b b4 24 a8 00 00 	mov    0xa8(%rsp),%r14
    3543:	00 
    3544:	48 8b b4 24 a0 00 00 	mov    0xa0(%rsp),%rsi
    354b:	00 
    354c:	0f 1f 40 00          	nopl   0x0(%rax)
    3550:	f3 43 0f 10 04 8e    	movss  (%r14,%r9,4),%xmm0
    3556:	f3 0f 58 c2          	addss  %xmm2,%xmm0
    355a:	f3 42 0f 11 04 8e    	movss  %xmm0,(%rsi,%r9,4)
    3560:	49 ff c1             	inc    %r9
    3563:	4c 39 cb             	cmp    %r9,%rbx
    3566:	75 e8                	jne    3550 <main_graph_mobilenet+0x1280>
    3568:	48 83 fb 08          	cmp    $0x8,%rbx
    356c:	0f 82 cf 00 00 00    	jb     3641 <main_graph_mobilenet+0x1371>
    3572:	4c 8b 8c 24 58 01 00 	mov    0x158(%rsp),%r9
    3579:	00 
    357a:	4d 01 c1             	add    %r8,%r9
    357d:	4c 8b b4 24 50 01 00 	mov    0x150(%rsp),%r14
    3584:	00 
    3585:	4d 01 c6             	add    %r8,%r14
    3588:	4d 29 f1             	sub    %r14,%r9
    358b:	49 83 f9 20          	cmp    $0x20,%r9
    358f:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    3595:	0f 28 d3             	movaps %xmm3,%xmm2
    3598:	45 0f 28 d3          	movaps %xmm11,%xmm10
    359c:	45 0f 28 dc          	movaps %xmm12,%xmm11
    35a0:	45 0f 28 e5          	movaps %xmm13,%xmm12
    35a4:	45 0f 28 ee          	movaps %xmm14,%xmm13
    35a8:	45 0f 28 f7          	movaps %xmm15,%xmm14
    35ac:	f3 44 0f 10 3d a7 ab 	movss  0xaba7(%rip),%xmm15        # e15c <_fini+0x1e0>
    35b3:	00 00 
    35b5:	0f 28 1d 14 ab 00 00 	movaps 0xab14(%rip),%xmm3        # e0d0 <_fini+0x154>
    35bc:	0f 28 2d dd aa 00 00 	movaps 0xaadd(%rip),%xmm5        # e0a0 <_fini+0x124>
    35c3:	0f 28 3d b6 aa 00 00 	movaps 0xaab6(%rip),%xmm7        # e080 <_fini+0x104>
    35ca:	44 0f 28 05 6e aa 00 	movaps 0xaa6e(%rip),%xmm8        # e040 <_fini+0xc4>
    35d1:	00 
    35d2:	4c 8b b4 24 b0 00 00 	mov    0xb0(%rsp),%r14
    35d9:	00 
    35da:	48 8b b4 24 80 00 00 	mov    0x80(%rsp),%rsi
    35e1:	00 
    35e2:	0f 82 b8 00 00 00    	jb     36a0 <main_graph_mobilenet+0x13d0>
    35e8:	4d 89 d9             	mov    %r11,%r9
    35eb:	4c 8b b4 24 e0 00 00 	mov    0xe0(%rsp),%r14
    35f2:	00 
    35f3:	0f 28 35 26 aa 00 00 	movaps 0xaa26(%rip),%xmm6        # e020 <_fini+0xa4>
    35fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    3600:	42 0f 10 44 35 f0    	movups -0x10(%rbp,%r14,1),%xmm0
    3606:	42 0f 10 4c 35 00    	movups 0x0(%rbp,%r14,1),%xmm1
    360c:	0f 58 c6             	addps  %xmm6,%xmm0
    360f:	0f 58 ce             	addps  %xmm6,%xmm1
    3612:	42 0f 11 44 31 f0    	movups %xmm0,-0x10(%rcx,%r14,1)
    3618:	42 0f 11 0c 31       	movups %xmm1,(%rcx,%r14,1)
    361d:	49 83 c6 20          	add    $0x20,%r14
    3621:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    3625:	75 d9                	jne    3600 <main_graph_mobilenet+0x1330>
    3627:	4c 39 db             	cmp    %r11,%rbx
    362a:	4d 89 d9             	mov    %r11,%r9
    362d:	f3 0f 10 0d eb aa 00 	movss  0xaaeb(%rip),%xmm1        # e120 <_fini+0x1a4>
    3634:	00 
    3635:	4c 8b b4 24 b0 00 00 	mov    0xb0(%rsp),%r14
    363c:	00 
    363d:	75 61                	jne    36a0 <main_graph_mobilenet+0x13d0>
    363f:	eb 77                	jmp    36b8 <main_graph_mobilenet+0x13e8>
    3641:	45 31 c9             	xor    %r9d,%r9d
    3644:	0f 28 d3             	movaps %xmm3,%xmm2
    3647:	45 0f 28 d3          	movaps %xmm11,%xmm10
    364b:	45 0f 28 dc          	movaps %xmm12,%xmm11
    364f:	45 0f 28 e5          	movaps %xmm13,%xmm12
    3653:	45 0f 28 ee          	movaps %xmm14,%xmm13
    3657:	45 0f 28 f7          	movaps %xmm15,%xmm14
    365b:	f3 44 0f 10 3d f8 aa 	movss  0xaaf8(%rip),%xmm15        # e15c <_fini+0x1e0>
    3662:	00 00 
    3664:	0f 28 1d 65 aa 00 00 	movaps 0xaa65(%rip),%xmm3        # e0d0 <_fini+0x154>
    366b:	0f 28 2d 2e aa 00 00 	movaps 0xaa2e(%rip),%xmm5        # e0a0 <_fini+0x124>
    3672:	0f 28 3d 07 aa 00 00 	movaps 0xaa07(%rip),%xmm7        # e080 <_fini+0x104>
    3679:	44 0f 28 05 bf a9 00 	movaps 0xa9bf(%rip),%xmm8        # e040 <_fini+0xc4>
    3680:	00 
    3681:	4c 8b b4 24 b0 00 00 	mov    0xb0(%rsp),%r14
    3688:	00 
    3689:	48 8b b4 24 80 00 00 	mov    0x80(%rsp),%rsi
    3690:	00 
    3691:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    3698:	00 00 00 
    369b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    36a0:	f3 42 0f 10 04 8a    	movss  (%rdx,%r9,4),%xmm0
    36a6:	f3 0f 58 c2          	addss  %xmm2,%xmm0
    36aa:	f3 43 0f 11 04 8e    	movss  %xmm0,(%r14,%r9,4)
    36b0:	49 ff c1             	inc    %r9
    36b3:	4c 39 cb             	cmp    %r9,%rbx
    36b6:	75 e8                	jne    36a0 <main_graph_mobilenet+0x13d0>
    36b8:	48 89 94 24 80 00 00 	mov    %rdx,0x80(%rsp)
    36bf:	00 
    36c0:	48 83 fb 08          	cmp    $0x8,%rbx
    36c4:	0f 82 aa 00 00 00    	jb     3774 <main_graph_mobilenet+0x14a4>
    36ca:	4c 8b 8c 24 68 01 00 	mov    0x168(%rsp),%r9
    36d1:	00 
    36d2:	4d 01 c1             	add    %r8,%r9
    36d5:	4c 8b b4 24 60 01 00 	mov    0x160(%rsp),%r14
    36dc:	00 
    36dd:	4d 01 c6             	add    %r8,%r14
    36e0:	4d 29 f1             	sub    %r14,%r9
    36e3:	49 83 f9 20          	cmp    $0x20,%r9
    36e7:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    36ed:	4c 8b b4 24 c0 00 00 	mov    0xc0(%rsp),%r14
    36f4:	00 
    36f5:	48 8b 94 24 b8 00 00 	mov    0xb8(%rsp),%rdx
    36fc:	00 
    36fd:	f3 0f 10 35 2b aa 00 	movss  0xaa2b(%rip),%xmm6        # e130 <_fini+0x1b4>
    3704:	00 
    3705:	0f 82 85 00 00 00    	jb     3790 <main_graph_mobilenet+0x14c0>
    370b:	4d 89 d9             	mov    %r11,%r9
    370e:	4c 8b b4 24 30 01 00 	mov    0x130(%rsp),%r14
    3715:	00 
    3716:	44 0f 28 0d 12 a9 00 	movaps 0xa912(%rip),%xmm9        # e030 <_fini+0xb4>
    371d:	00 
    371e:	66 90                	xchg   %ax,%ax
    3720:	42 0f 10 44 35 f0    	movups -0x10(%rbp,%r14,1),%xmm0
    3726:	42 0f 10 4c 35 00    	movups 0x0(%rbp,%r14,1),%xmm1
    372c:	41 0f 58 c1          	addps  %xmm9,%xmm0
    3730:	41 0f 58 c9          	addps  %xmm9,%xmm1
    3734:	42 0f 11 44 31 f0    	movups %xmm0,-0x10(%rcx,%r14,1)
    373a:	42 0f 11 0c 31       	movups %xmm1,(%rcx,%r14,1)
    373f:	49 83 c6 20          	add    $0x20,%r14
    3743:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    3747:	75 d7                	jne    3720 <main_graph_mobilenet+0x1450>
    3749:	4c 39 db             	cmp    %r11,%rbx
    374c:	4d 89 d9             	mov    %r11,%r9
    374f:	f3 0f 10 0d c9 a9 00 	movss  0xa9c9(%rip),%xmm1        # e120 <_fini+0x1a4>
    3756:	00 
    3757:	4c 8b b4 24 c0 00 00 	mov    0xc0(%rsp),%r14
    375e:	00 
    375f:	48 8b 94 24 b8 00 00 	mov    0xb8(%rsp),%rdx
    3766:	00 
    3767:	f3 44 0f 10 0d d0 a9 	movss  0xa9d0(%rip),%xmm9        # e140 <_fini+0x1c4>
    376e:	00 00 
    3770:	75 1e                	jne    3790 <main_graph_mobilenet+0x14c0>
    3772:	eb 34                	jmp    37a8 <main_graph_mobilenet+0x14d8>
    3774:	45 31 c9             	xor    %r9d,%r9d
    3777:	4c 8b b4 24 c0 00 00 	mov    0xc0(%rsp),%r14
    377e:	00 
    377f:	48 8b 94 24 b8 00 00 	mov    0xb8(%rsp),%rdx
    3786:	00 
    3787:	f3 0f 10 35 a1 a9 00 	movss  0xa9a1(%rip),%xmm6        # e130 <_fini+0x1b4>
    378e:	00 
    378f:	90                   	nop
    3790:	f3 43 0f 10 04 8e    	movss  (%r14,%r9,4),%xmm0
    3796:	f3 0f 58 c4          	addss  %xmm4,%xmm0
    379a:	f3 42 0f 11 04 8a    	movss  %xmm0,(%rdx,%r9,4)
    37a0:	49 ff c1             	inc    %r9
    37a3:	4c 39 cb             	cmp    %r9,%rbx
    37a6:	75 e8                	jne    3790 <main_graph_mobilenet+0x14c0>
    37a8:	48 83 fb 08          	cmp    $0x8,%rbx
    37ac:	0f 82 7e 00 00 00    	jb     3830 <main_graph_mobilenet+0x1560>
    37b2:	4c 8b 8c 24 80 02 00 	mov    0x280(%rsp),%r9
    37b9:	00 
    37ba:	4d 01 c1             	add    %r8,%r9
    37bd:	4c 8b b4 24 70 01 00 	mov    0x170(%rsp),%r14
    37c4:	00 
    37c5:	4d 01 c6             	add    %r8,%r14
    37c8:	4d 29 f1             	sub    %r14,%r9
    37cb:	49 83 f9 20          	cmp    $0x20,%r9
    37cf:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    37d5:	4c 8b 74 24 20       	mov    0x20(%rsp),%r14
    37da:	48 8b 94 24 80 00 00 	mov    0x80(%rsp),%rdx
    37e1:	00 
    37e2:	72 5c                	jb     3840 <main_graph_mobilenet+0x1570>
    37e4:	4d 89 d9             	mov    %r11,%r9
    37e7:	4c 8b b4 24 38 01 00 	mov    0x138(%rsp),%r14
    37ee:	00 
    37ef:	90                   	nop
    37f0:	42 0f 28 44 35 f0    	movaps -0x10(%rbp,%r14,1),%xmm0
    37f6:	41 0f 58 c0          	addps  %xmm8,%xmm0
    37fa:	42 0f 28 4c 35 00    	movaps 0x0(%rbp,%r14,1),%xmm1
    3800:	41 0f 58 c8          	addps  %xmm8,%xmm1
    3804:	42 0f 29 44 31 f0    	movaps %xmm0,-0x10(%rcx,%r14,1)
    380a:	42 0f 29 0c 31       	movaps %xmm1,(%rcx,%r14,1)
    380f:	49 83 c6 20          	add    $0x20,%r14
    3813:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    3817:	75 d7                	jne    37f0 <main_graph_mobilenet+0x1520>
    3819:	4c 39 db             	cmp    %r11,%rbx
    381c:	4d 89 d9             	mov    %r11,%r9
    381f:	f3 0f 10 0d f9 a8 00 	movss  0xa8f9(%rip),%xmm1        # e120 <_fini+0x1a4>
    3826:	00 
    3827:	4c 8b 74 24 20       	mov    0x20(%rsp),%r14
    382c:	75 12                	jne    3840 <main_graph_mobilenet+0x1570>
    382e:	eb 28                	jmp    3858 <main_graph_mobilenet+0x1588>
    3830:	45 31 c9             	xor    %r9d,%r9d
    3833:	4c 8b 74 24 20       	mov    0x20(%rsp),%r14
    3838:	48 8b 94 24 80 00 00 	mov    0x80(%rsp),%rdx
    383f:	00 
    3840:	f3 43 0f 10 04 8a    	movss  (%r10,%r9,4),%xmm0
    3846:	f3 0f 58 c6          	addss  %xmm6,%xmm0
    384a:	f3 43 0f 11 04 8e    	movss  %xmm0,(%r14,%r9,4)
    3850:	49 ff c1             	inc    %r9
    3853:	4c 39 cb             	cmp    %r9,%rbx
    3856:	75 e8                	jne    3840 <main_graph_mobilenet+0x1570>
    3858:	4c 89 94 24 80 00 00 	mov    %r10,0x80(%rsp)
    385f:	00 
    3860:	48 83 fb 08          	cmp    $0x8,%rbx
    3864:	0f 82 ba 00 00 00    	jb     3924 <main_graph_mobilenet+0x1654>
    386a:	4c 8b 8c 24 80 01 00 	mov    0x180(%rsp),%r9
    3871:	00 
    3872:	4d 01 c1             	add    %r8,%r9
    3875:	4c 8b b4 24 78 01 00 	mov    0x178(%rsp),%r14
    387c:	00 
    387d:	4d 01 c6             	add    %r8,%r14
    3880:	4d 29 f1             	sub    %r14,%r9
    3883:	49 83 f9 20          	cmp    $0x20,%r9
    3887:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    388d:	4c 8b b4 24 d0 00 00 	mov    0xd0(%rsp),%r14
    3894:	00 
    3895:	4c 8b 94 24 c8 00 00 	mov    0xc8(%rsp),%r10
    389c:	00 
    389d:	0f 28 35 bc a7 00 00 	movaps 0xa7bc(%rip),%xmm6        # e060 <_fini+0xe4>
    38a4:	f3 44 0f 10 05 87 a8 	movss  0xa887(%rip),%xmm8        # e134 <_fini+0x1b8>
    38ab:	00 00 
    38ad:	0f 82 9d 00 00 00    	jb     3950 <main_graph_mobilenet+0x1680>
    38b3:	4d 89 d9             	mov    %r11,%r9
    38b6:	4c 8b b4 24 28 01 00 	mov    0x128(%rsp),%r14
    38bd:	00 
    38be:	44 0f 28 05 8a a7 00 	movaps 0xa78a(%rip),%xmm8        # e050 <_fini+0xd4>
    38c5:	00 
    38c6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    38cd:	00 00 00 
    38d0:	42 0f 10 44 35 f0    	movups -0x10(%rbp,%r14,1),%xmm0
    38d6:	42 0f 10 4c 35 00    	movups 0x0(%rbp,%r14,1),%xmm1
    38dc:	41 0f 58 c0          	addps  %xmm8,%xmm0
    38e0:	41 0f 58 c8          	addps  %xmm8,%xmm1
    38e4:	42 0f 11 44 31 f0    	movups %xmm0,-0x10(%rcx,%r14,1)
    38ea:	42 0f 11 0c 31       	movups %xmm1,(%rcx,%r14,1)
    38ef:	49 83 c6 20          	add    $0x20,%r14
    38f3:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    38f7:	75 d7                	jne    38d0 <main_graph_mobilenet+0x1600>
    38f9:	4c 39 db             	cmp    %r11,%rbx
    38fc:	4d 89 d9             	mov    %r11,%r9
    38ff:	f3 0f 10 0d 19 a8 00 	movss  0xa819(%rip),%xmm1        # e120 <_fini+0x1a4>
    3906:	00 
    3907:	4c 8b b4 24 d0 00 00 	mov    0xd0(%rsp),%r14
    390e:	00 
    390f:	4c 8b 94 24 c8 00 00 	mov    0xc8(%rsp),%r10
    3916:	00 
    3917:	f3 44 0f 10 05 14 a8 	movss  0xa814(%rip),%xmm8        # e134 <_fini+0x1b8>
    391e:	00 00 
    3920:	75 2e                	jne    3950 <main_graph_mobilenet+0x1680>
    3922:	eb 45                	jmp    3969 <main_graph_mobilenet+0x1699>
    3924:	45 31 c9             	xor    %r9d,%r9d
    3927:	4c 8b b4 24 d0 00 00 	mov    0xd0(%rsp),%r14
    392e:	00 
    392f:	4c 8b 94 24 c8 00 00 	mov    0xc8(%rsp),%r10
    3936:	00 
    3937:	0f 28 35 22 a7 00 00 	movaps 0xa722(%rip),%xmm6        # e060 <_fini+0xe4>
    393e:	f3 44 0f 10 05 ed a7 	movss  0xa7ed(%rip),%xmm8        # e134 <_fini+0x1b8>
    3945:	00 00 
    3947:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    394e:	00 00 
    3950:	f3 43 0f 10 04 8e    	movss  (%r14,%r9,4),%xmm0
    3956:	f3 41 0f 58 c0       	addss  %xmm8,%xmm0
    395b:	f3 43 0f 11 04 8a    	movss  %xmm0,(%r10,%r9,4)
    3961:	49 ff c1             	inc    %r9
    3964:	4c 39 cb             	cmp    %r9,%rbx
    3967:	75 e7                	jne    3950 <main_graph_mobilenet+0x1680>
    3969:	48 83 fb 08          	cmp    $0x8,%rbx
    396d:	0f 82 8e 00 00 00    	jb     3a01 <main_graph_mobilenet+0x1731>
    3973:	4c 8b 8c 24 88 01 00 	mov    0x188(%rsp),%r9
    397a:	00 
    397b:	4d 01 c1             	add    %r8,%r9
    397e:	4c 8b b4 24 88 02 00 	mov    0x288(%rsp),%r14
    3985:	00 
    3986:	4d 01 c6             	add    %r8,%r14
    3989:	4d 29 f1             	sub    %r14,%r9
    398c:	49 83 f9 20          	cmp    $0x20,%r9
    3990:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    3996:	4c 8b b4 24 d8 00 00 	mov    0xd8(%rsp),%r14
    399d:	00 
    399e:	4c 8b 94 24 80 00 00 	mov    0x80(%rsp),%r10
    39a5:	00 
    39a6:	f3 44 0f 10 05 89 a7 	movss  0xa789(%rip),%xmm8        # e138 <_fini+0x1bc>
    39ad:	00 00 
    39af:	72 6f                	jb     3a20 <main_graph_mobilenet+0x1750>
    39b1:	4d 89 d9             	mov    %r11,%r9
    39b4:	4c 8b b4 24 30 02 00 	mov    0x230(%rsp),%r14
    39bb:	00 
    39bc:	0f 1f 40 00          	nopl   0x0(%rax)
    39c0:	42 0f 10 44 35 f0    	movups -0x10(%rbp,%r14,1),%xmm0
    39c6:	42 0f 10 4c 35 00    	movups 0x0(%rbp,%r14,1),%xmm1
    39cc:	0f 58 c6             	addps  %xmm6,%xmm0
    39cf:	0f 58 ce             	addps  %xmm6,%xmm1
    39d2:	42 0f 11 44 31 f0    	movups %xmm0,-0x10(%rcx,%r14,1)
    39d8:	42 0f 11 0c 31       	movups %xmm1,(%rcx,%r14,1)
    39dd:	49 83 c6 20          	add    $0x20,%r14
    39e1:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    39e5:	75 d9                	jne    39c0 <main_graph_mobilenet+0x16f0>
    39e7:	4c 39 db             	cmp    %r11,%rbx
    39ea:	4d 89 d9             	mov    %r11,%r9
    39ed:	f3 0f 10 0d 2b a7 00 	movss  0xa72b(%rip),%xmm1        # e120 <_fini+0x1a4>
    39f4:	00 
    39f5:	4c 8b b4 24 d8 00 00 	mov    0xd8(%rsp),%r14
    39fc:	00 
    39fd:	75 21                	jne    3a20 <main_graph_mobilenet+0x1750>
    39ff:	eb 38                	jmp    3a39 <main_graph_mobilenet+0x1769>
    3a01:	45 31 c9             	xor    %r9d,%r9d
    3a04:	4c 8b b4 24 d8 00 00 	mov    0xd8(%rsp),%r14
    3a0b:	00 
    3a0c:	4c 8b 94 24 80 00 00 	mov    0x80(%rsp),%r10
    3a13:	00 
    3a14:	f3 44 0f 10 05 1b a7 	movss  0xa71b(%rip),%xmm8        # e138 <_fini+0x1bc>
    3a1b:	00 00 
    3a1d:	0f 1f 00             	nopl   (%rax)
    3a20:	f3 42 0f 10 04 88    	movss  (%rax,%r9,4),%xmm0
    3a26:	f3 41 0f 58 c0       	addss  %xmm8,%xmm0
    3a2b:	f3 43 0f 11 04 8e    	movss  %xmm0,(%r14,%r9,4)
    3a31:	49 ff c1             	inc    %r9
    3a34:	4c 39 cb             	cmp    %r9,%rbx
    3a37:	75 e7                	jne    3a20 <main_graph_mobilenet+0x1750>
    3a39:	48 89 84 24 80 00 00 	mov    %rax,0x80(%rsp)
    3a40:	00 
    3a41:	48 83 fb 08          	cmp    $0x8,%rbx
    3a45:	0f 82 9e 00 00 00    	jb     3ae9 <main_graph_mobilenet+0x1819>
    3a4b:	4c 8b 8c 24 90 01 00 	mov    0x190(%rsp),%r9
    3a52:	00 
    3a53:	4d 01 c1             	add    %r8,%r9
    3a56:	4c 8b b4 24 90 02 00 	mov    0x290(%rsp),%r14
    3a5d:	00 
    3a5e:	4d 01 c6             	add    %r8,%r14
    3a61:	4d 29 f1             	sub    %r14,%r9
    3a64:	49 83 f9 20          	cmp    $0x20,%r9
    3a68:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    3a6e:	4c 8b b4 24 90 00 00 	mov    0x90(%rsp),%r14
    3a75:	00 
    3a76:	48 8b 84 24 88 00 00 	mov    0x88(%rsp),%rax
    3a7d:	00 
    3a7e:	f3 44 0f 10 05 b5 a6 	movss  0xa6b5(%rip),%xmm8        # e13c <_fini+0x1c0>
    3a85:	00 00 
    3a87:	0f 82 83 00 00 00    	jb     3b10 <main_graph_mobilenet+0x1840>
    3a8d:	4d 89 d9             	mov    %r11,%r9
    3a90:	4c 8b b4 24 50 02 00 	mov    0x250(%rsp),%r14
    3a97:	00 
    3a98:	0f 28 35 d1 a5 00 00 	movaps 0xa5d1(%rip),%xmm6        # e070 <_fini+0xf4>
    3a9f:	90                   	nop
    3aa0:	42 0f 10 44 35 f0    	movups -0x10(%rbp,%r14,1),%xmm0
    3aa6:	42 0f 10 4c 35 00    	movups 0x0(%rbp,%r14,1),%xmm1
    3aac:	0f 58 c6             	addps  %xmm6,%xmm0
    3aaf:	0f 58 ce             	addps  %xmm6,%xmm1
    3ab2:	42 0f 11 44 31 f0    	movups %xmm0,-0x10(%rcx,%r14,1)
    3ab8:	42 0f 11 0c 31       	movups %xmm1,(%rcx,%r14,1)
    3abd:	49 83 c6 20          	add    $0x20,%r14
    3ac1:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    3ac5:	75 d9                	jne    3aa0 <main_graph_mobilenet+0x17d0>
    3ac7:	4c 39 db             	cmp    %r11,%rbx
    3aca:	4d 89 d9             	mov    %r11,%r9
    3acd:	f3 0f 10 0d 4b a6 00 	movss  0xa64b(%rip),%xmm1        # e120 <_fini+0x1a4>
    3ad4:	00 
    3ad5:	4c 8b b4 24 90 00 00 	mov    0x90(%rsp),%r14
    3adc:	00 
    3add:	48 8b 84 24 88 00 00 	mov    0x88(%rsp),%rax
    3ae4:	00 
    3ae5:	75 29                	jne    3b10 <main_graph_mobilenet+0x1840>
    3ae7:	eb 40                	jmp    3b29 <main_graph_mobilenet+0x1859>
    3ae9:	45 31 c9             	xor    %r9d,%r9d
    3aec:	4c 8b b4 24 90 00 00 	mov    0x90(%rsp),%r14
    3af3:	00 
    3af4:	48 8b 84 24 88 00 00 	mov    0x88(%rsp),%rax
    3afb:	00 
    3afc:	f3 44 0f 10 05 37 a6 	movss  0xa637(%rip),%xmm8        # e13c <_fini+0x1c0>
    3b03:	00 00 
    3b05:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    3b0c:	00 00 00 
    3b0f:	90                   	nop
    3b10:	f3 43 0f 10 04 8e    	movss  (%r14,%r9,4),%xmm0
    3b16:	f3 41 0f 58 c0       	addss  %xmm8,%xmm0
    3b1b:	f3 42 0f 11 04 88    	movss  %xmm0,(%rax,%r9,4)
    3b21:	49 ff c1             	inc    %r9
    3b24:	4c 39 cb             	cmp    %r9,%rbx
    3b27:	75 e7                	jne    3b10 <main_graph_mobilenet+0x1840>
    3b29:	48 83 fb 08          	cmp    $0x8,%rbx
    3b2d:	f3 0f 10 35 0f a6 00 	movss  0xa60f(%rip),%xmm6        # e144 <_fini+0x1c8>
    3b34:	00 
    3b35:	0f 82 83 00 00 00    	jb     3bbe <main_graph_mobilenet+0x18ee>
    3b3b:	4c 8b 8c 24 a0 02 00 	mov    0x2a0(%rsp),%r9
    3b42:	00 
    3b43:	4d 01 c1             	add    %r8,%r9
    3b46:	4c 8b b4 24 98 02 00 	mov    0x298(%rsp),%r14
    3b4d:	00 
    3b4e:	4d 01 c6             	add    %r8,%r14
    3b51:	4d 29 f1             	sub    %r14,%r9
    3b54:	49 83 f9 20          	cmp    $0x20,%r9
    3b58:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    3b5e:	4c 8b 74 24 30       	mov    0x30(%rsp),%r14
    3b63:	48 8b 84 24 80 00 00 	mov    0x80(%rsp),%rax
    3b6a:	00 
    3b6b:	72 63                	jb     3bd0 <main_graph_mobilenet+0x1900>
    3b6d:	4d 89 d9             	mov    %r11,%r9
    3b70:	4c 8b b4 24 58 02 00 	mov    0x258(%rsp),%r14
    3b77:	00 
    3b78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    3b7f:	00 
    3b80:	42 0f 28 44 35 f0    	movaps -0x10(%rbp,%r14,1),%xmm0
    3b86:	0f 58 c7             	addps  %xmm7,%xmm0
    3b89:	42 0f 28 4c 35 00    	movaps 0x0(%rbp,%r14,1),%xmm1
    3b8f:	0f 58 cf             	addps  %xmm7,%xmm1
    3b92:	42 0f 29 44 31 f0    	movaps %xmm0,-0x10(%rcx,%r14,1)
    3b98:	42 0f 29 0c 31       	movaps %xmm1,(%rcx,%r14,1)
    3b9d:	49 83 c6 20          	add    $0x20,%r14
    3ba1:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    3ba5:	75 d9                	jne    3b80 <main_graph_mobilenet+0x18b0>
    3ba7:	4c 39 db             	cmp    %r11,%rbx
    3baa:	4d 89 d9             	mov    %r11,%r9
    3bad:	f3 0f 10 0d 6b a5 00 	movss  0xa56b(%rip),%xmm1        # e120 <_fini+0x1a4>
    3bb4:	00 
    3bb5:	4c 8b 74 24 30       	mov    0x30(%rsp),%r14
    3bba:	75 14                	jne    3bd0 <main_graph_mobilenet+0x1900>
    3bbc:	eb 2c                	jmp    3bea <main_graph_mobilenet+0x191a>
    3bbe:	45 31 c9             	xor    %r9d,%r9d
    3bc1:	4c 8b 74 24 30       	mov    0x30(%rsp),%r14
    3bc6:	48 8b 84 24 80 00 00 	mov    0x80(%rsp),%rax
    3bcd:	00 
    3bce:	66 90                	xchg   %ax,%ax
    3bd0:	f3 43 0f 10 44 8d 00 	movss  0x0(%r13,%r9,4),%xmm0
    3bd7:	f3 41 0f 58 c1       	addss  %xmm9,%xmm0
    3bdc:	f3 43 0f 11 04 8e    	movss  %xmm0,(%r14,%r9,4)
    3be2:	49 ff c1             	inc    %r9
    3be5:	4c 39 cb             	cmp    %r9,%rbx
    3be8:	75 e6                	jne    3bd0 <main_graph_mobilenet+0x1900>
    3bea:	4c 89 ac 24 80 00 00 	mov    %r13,0x80(%rsp)
    3bf1:	00 
    3bf2:	48 83 fb 08          	cmp    $0x8,%rbx
    3bf6:	0f 82 8a 00 00 00    	jb     3c86 <main_graph_mobilenet+0x19b6>
    3bfc:	4c 8b 8c 24 98 01 00 	mov    0x198(%rsp),%r9
    3c03:	00 
    3c04:	4d 01 c1             	add    %r8,%r9
    3c07:	4c 8b b4 24 a8 02 00 	mov    0x2a8(%rsp),%r14
    3c0e:	00 
    3c0f:	4d 01 c6             	add    %r8,%r14
    3c12:	4d 29 f1             	sub    %r14,%r9
    3c15:	49 83 f9 20          	cmp    $0x20,%r9
    3c19:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    3c1f:	4c 8b 74 24 70       	mov    0x70(%rsp),%r14
    3c24:	4c 8b ac 24 98 00 00 	mov    0x98(%rsp),%r13
    3c2b:	00 
    3c2c:	72 72                	jb     3ca0 <main_graph_mobilenet+0x19d0>
    3c2e:	4d 89 d9             	mov    %r11,%r9
    3c31:	4c 8b b4 24 60 02 00 	mov    0x260(%rsp),%r14
    3c38:	00 
    3c39:	0f 28 3d 50 a4 00 00 	movaps 0xa450(%rip),%xmm7        # e090 <_fini+0x114>
    3c40:	42 0f 10 44 35 f0    	movups -0x10(%rbp,%r14,1),%xmm0
    3c46:	42 0f 10 4c 35 00    	movups 0x0(%rbp,%r14,1),%xmm1
    3c4c:	0f 58 c7             	addps  %xmm7,%xmm0
    3c4f:	0f 58 cf             	addps  %xmm7,%xmm1
    3c52:	42 0f 11 44 31 f0    	movups %xmm0,-0x10(%rcx,%r14,1)
    3c58:	42 0f 11 0c 31       	movups %xmm1,(%rcx,%r14,1)
    3c5d:	49 83 c6 20          	add    $0x20,%r14
    3c61:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    3c65:	75 d9                	jne    3c40 <main_graph_mobilenet+0x1970>
    3c67:	4c 39 db             	cmp    %r11,%rbx
    3c6a:	4d 89 d9             	mov    %r11,%r9
    3c6d:	f3 0f 10 0d ab a4 00 	movss  0xa4ab(%rip),%xmm1        # e120 <_fini+0x1a4>
    3c74:	00 
    3c75:	4c 8b 74 24 70       	mov    0x70(%rsp),%r14
    3c7a:	4c 8b ac 24 98 00 00 	mov    0x98(%rsp),%r13
    3c81:	00 
    3c82:	75 1c                	jne    3ca0 <main_graph_mobilenet+0x19d0>
    3c84:	eb 33                	jmp    3cb9 <main_graph_mobilenet+0x19e9>
    3c86:	45 31 c9             	xor    %r9d,%r9d
    3c89:	4c 8b 74 24 70       	mov    0x70(%rsp),%r14
    3c8e:	4c 8b ac 24 98 00 00 	mov    0x98(%rsp),%r13
    3c95:	00 
    3c96:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    3c9d:	00 00 00 
    3ca0:	f3 43 0f 10 04 8e    	movss  (%r14,%r9,4),%xmm0
    3ca6:	f3 0f 58 c6          	addss  %xmm6,%xmm0
    3caa:	f3 43 0f 11 44 8d 00 	movss  %xmm0,0x0(%r13,%r9,4)
    3cb1:	49 ff c1             	inc    %r9
    3cb4:	4c 39 cb             	cmp    %r9,%rbx
    3cb7:	75 e7                	jne    3ca0 <main_graph_mobilenet+0x19d0>
    3cb9:	48 83 fb 08          	cmp    $0x8,%rbx
    3cbd:	72 7f                	jb     3d3e <main_graph_mobilenet+0x1a6e>
    3cbf:	4c 8b 8c 24 a8 01 00 	mov    0x1a8(%rsp),%r9
    3cc6:	00 
    3cc7:	4d 01 c1             	add    %r8,%r9
    3cca:	4c 8b b4 24 a0 01 00 	mov    0x1a0(%rsp),%r14
    3cd1:	00 
    3cd2:	4d 01 c6             	add    %r8,%r14
    3cd5:	4d 29 f1             	sub    %r14,%r9
    3cd8:	49 83 f9 20          	cmp    $0x20,%r9
    3cdc:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    3ce2:	4c 8b 74 24 78       	mov    0x78(%rsp),%r14
    3ce7:	4c 8b ac 24 80 00 00 	mov    0x80(%rsp),%r13
    3cee:	00 
    3cef:	72 5f                	jb     3d50 <main_graph_mobilenet+0x1a80>
    3cf1:	4d 89 d9             	mov    %r11,%r9
    3cf4:	4c 8b b4 24 38 02 00 	mov    0x238(%rsp),%r14
    3cfb:	00 
    3cfc:	0f 1f 40 00          	nopl   0x0(%rax)
    3d00:	42 0f 10 44 35 f0    	movups -0x10(%rbp,%r14,1),%xmm0
    3d06:	42 0f 10 4c 35 00    	movups 0x0(%rbp,%r14,1),%xmm1
    3d0c:	0f 58 c5             	addps  %xmm5,%xmm0
    3d0f:	0f 58 cd             	addps  %xmm5,%xmm1
    3d12:	42 0f 11 44 31 f0    	movups %xmm0,-0x10(%rcx,%r14,1)
    3d18:	42 0f 11 0c 31       	movups %xmm1,(%rcx,%r14,1)
    3d1d:	49 83 c6 20          	add    $0x20,%r14
    3d21:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    3d25:	75 d9                	jne    3d00 <main_graph_mobilenet+0x1a30>
    3d27:	4c 39 db             	cmp    %r11,%rbx
    3d2a:	4d 89 d9             	mov    %r11,%r9
    3d2d:	f3 0f 10 0d eb a3 00 	movss  0xa3eb(%rip),%xmm1        # e120 <_fini+0x1a4>
    3d34:	00 
    3d35:	4c 8b 74 24 78       	mov    0x78(%rsp),%r14
    3d3a:	75 14                	jne    3d50 <main_graph_mobilenet+0x1a80>
    3d3c:	eb 2b                	jmp    3d69 <main_graph_mobilenet+0x1a99>
    3d3e:	45 31 c9             	xor    %r9d,%r9d
    3d41:	4c 8b 74 24 78       	mov    0x78(%rsp),%r14
    3d46:	4c 8b ac 24 80 00 00 	mov    0x80(%rsp),%r13
    3d4d:	00 
    3d4e:	66 90                	xchg   %ax,%ax
    3d50:	f3 43 0f 10 04 8c    	movss  (%r12,%r9,4),%xmm0
    3d56:	f3 41 0f 58 c2       	addss  %xmm10,%xmm0
    3d5b:	f3 43 0f 11 04 8e    	movss  %xmm0,(%r14,%r9,4)
    3d61:	49 ff c1             	inc    %r9
    3d64:	4c 39 cb             	cmp    %r9,%rbx
    3d67:	75 e7                	jne    3d50 <main_graph_mobilenet+0x1a80>
    3d69:	48 83 fb 08          	cmp    $0x8,%rbx
    3d6d:	0f 82 90 00 00 00    	jb     3e03 <main_graph_mobilenet+0x1b33>
    3d73:	4c 8b 8c 24 b8 01 00 	mov    0x1b8(%rsp),%r9
    3d7a:	00 
    3d7b:	4d 01 c1             	add    %r8,%r9
    3d7e:	4c 8b b4 24 b0 01 00 	mov    0x1b0(%rsp),%r14
    3d85:	00 
    3d86:	4d 01 c6             	add    %r8,%r14
    3d89:	4d 29 f1             	sub    %r14,%r9
    3d8c:	49 83 f9 20          	cmp    $0x20,%r9
    3d90:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    3d96:	4c 8b 74 24 50       	mov    0x50(%rsp),%r14
    3d9b:	4c 8b 64 24 58       	mov    0x58(%rsp),%r12
    3da0:	72 6e                	jb     3e10 <main_graph_mobilenet+0x1b40>
    3da2:	4d 89 d9             	mov    %r11,%r9
    3da5:	4c 8b b4 24 40 02 00 	mov    0x240(%rsp),%r14
    3dac:	00 
    3dad:	0f 28 2d fc a2 00 00 	movaps 0xa2fc(%rip),%xmm5        # e0b0 <_fini+0x134>
    3db4:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    3dbb:	00 00 00 
    3dbe:	66 90                	xchg   %ax,%ax
    3dc0:	42 0f 10 44 35 f0    	movups -0x10(%rbp,%r14,1),%xmm0
    3dc6:	42 0f 10 4c 35 00    	movups 0x0(%rbp,%r14,1),%xmm1
    3dcc:	0f 58 c5             	addps  %xmm5,%xmm0
    3dcf:	0f 58 cd             	addps  %xmm5,%xmm1
    3dd2:	42 0f 11 44 31 f0    	movups %xmm0,-0x10(%rcx,%r14,1)
    3dd8:	42 0f 11 0c 31       	movups %xmm1,(%rcx,%r14,1)
    3ddd:	49 83 c6 20          	add    $0x20,%r14
    3de1:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    3de5:	75 d9                	jne    3dc0 <main_graph_mobilenet+0x1af0>
    3de7:	4c 39 db             	cmp    %r11,%rbx
    3dea:	4d 89 d9             	mov    %r11,%r9
    3ded:	f3 0f 10 0d 2b a3 00 	movss  0xa32b(%rip),%xmm1        # e120 <_fini+0x1a4>
    3df4:	00 
    3df5:	4c 8b 74 24 50       	mov    0x50(%rsp),%r14
    3dfa:	4c 8b 64 24 58       	mov    0x58(%rsp),%r12
    3dff:	75 0f                	jne    3e10 <main_graph_mobilenet+0x1b40>
    3e01:	eb 26                	jmp    3e29 <main_graph_mobilenet+0x1b59>
    3e03:	45 31 c9             	xor    %r9d,%r9d
    3e06:	4c 8b 74 24 50       	mov    0x50(%rsp),%r14
    3e0b:	4c 8b 64 24 58       	mov    0x58(%rsp),%r12
    3e10:	f3 43 0f 10 04 8e    	movss  (%r14,%r9,4),%xmm0
    3e16:	f3 41 0f 58 c3       	addss  %xmm11,%xmm0
    3e1b:	f3 43 0f 11 04 8c    	movss  %xmm0,(%r12,%r9,4)
    3e21:	49 ff c1             	inc    %r9
    3e24:	4c 39 cb             	cmp    %r9,%rbx
    3e27:	75 e7                	jne    3e10 <main_graph_mobilenet+0x1b40>
    3e29:	48 83 fb 08          	cmp    $0x8,%rbx
    3e2d:	0f 82 93 00 00 00    	jb     3ec6 <main_graph_mobilenet+0x1bf6>
    3e33:	4c 8b 8c 24 b0 02 00 	mov    0x2b0(%rsp),%r9
    3e3a:	00 
    3e3b:	4d 01 c1             	add    %r8,%r9
    3e3e:	4c 8b b4 24 c0 01 00 	mov    0x1c0(%rsp),%r14
    3e45:	00 
    3e46:	4d 01 c6             	add    %r8,%r14
    3e49:	4d 29 f1             	sub    %r14,%r9
    3e4c:	49 83 f9 20          	cmp    $0x20,%r9
    3e50:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    3e56:	4c 8b 74 24 60       	mov    0x60(%rsp),%r14
    3e5b:	4c 8b a4 24 10 01 00 	mov    0x110(%rsp),%r12
    3e62:	00 
    3e63:	72 7b                	jb     3ee0 <main_graph_mobilenet+0x1c10>
    3e65:	4d 89 d9             	mov    %r11,%r9
    3e68:	4c 8b b4 24 48 02 00 	mov    0x248(%rsp),%r14
    3e6f:	00 
    3e70:	0f 28 25 49 a2 00 00 	movaps 0xa249(%rip),%xmm4        # e0c0 <_fini+0x144>
    3e77:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    3e7e:	00 00 
    3e80:	42 0f 28 44 35 f0    	movaps -0x10(%rbp,%r14,1),%xmm0
    3e86:	0f 58 c4             	addps  %xmm4,%xmm0
    3e89:	42 0f 28 4c 35 00    	movaps 0x0(%rbp,%r14,1),%xmm1
    3e8f:	0f 58 cc             	addps  %xmm4,%xmm1
    3e92:	42 0f 29 44 31 f0    	movaps %xmm0,-0x10(%rcx,%r14,1)
    3e98:	42 0f 29 0c 31       	movaps %xmm1,(%rcx,%r14,1)
    3e9d:	49 83 c6 20          	add    $0x20,%r14
    3ea1:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    3ea5:	75 d9                	jne    3e80 <main_graph_mobilenet+0x1bb0>
    3ea7:	4c 39 db             	cmp    %r11,%rbx
    3eaa:	4d 89 d9             	mov    %r11,%r9
    3ead:	f3 0f 10 0d 6b a2 00 	movss  0xa26b(%rip),%xmm1        # e120 <_fini+0x1a4>
    3eb4:	00 
    3eb5:	4c 8b 74 24 60       	mov    0x60(%rsp),%r14
    3eba:	f3 0f 10 25 6a a2 00 	movss  0xa26a(%rip),%xmm4        # e12c <_fini+0x1b0>
    3ec1:	00 
    3ec2:	75 1c                	jne    3ee0 <main_graph_mobilenet+0x1c10>
    3ec4:	eb 33                	jmp    3ef9 <main_graph_mobilenet+0x1c29>
    3ec6:	45 31 c9             	xor    %r9d,%r9d
    3ec9:	4c 8b 74 24 60       	mov    0x60(%rsp),%r14
    3ece:	4c 8b a4 24 10 01 00 	mov    0x110(%rsp),%r12
    3ed5:	00 
    3ed6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    3edd:	00 00 00 
    3ee0:	f3 43 0f 10 04 8f    	movss  (%r15,%r9,4),%xmm0
    3ee6:	f3 41 0f 58 c4       	addss  %xmm12,%xmm0
    3eeb:	f3 43 0f 11 04 8e    	movss  %xmm0,(%r14,%r9,4)
    3ef1:	49 ff c1             	inc    %r9
    3ef4:	4c 39 cb             	cmp    %r9,%rbx
    3ef7:	75 e7                	jne    3ee0 <main_graph_mobilenet+0x1c10>
    3ef9:	48 83 fb 08          	cmp    $0x8,%rbx
    3efd:	0f 82 80 00 00 00    	jb     3f83 <main_graph_mobilenet+0x1cb3>
    3f03:	4c 8b 8c 24 d0 01 00 	mov    0x1d0(%rsp),%r9
    3f0a:	00 
    3f0b:	4d 01 c1             	add    %r8,%r9
    3f0e:	4c 8b b4 24 c8 01 00 	mov    0x1c8(%rsp),%r14
    3f15:	00 
    3f16:	4d 01 c6             	add    %r8,%r14
    3f19:	4d 29 f1             	sub    %r14,%r9
    3f1c:	49 83 f9 20          	cmp    $0x20,%r9
    3f20:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    3f26:	4c 8b 74 24 40       	mov    0x40(%rsp),%r14
    3f2b:	4c 8b 7c 24 08       	mov    0x8(%rsp),%r15
    3f30:	72 5e                	jb     3f90 <main_graph_mobilenet+0x1cc0>
    3f32:	4d 89 d9             	mov    %r11,%r9
    3f35:	4c 8b b4 24 68 02 00 	mov    0x268(%rsp),%r14
    3f3c:	00 
    3f3d:	0f 1f 00             	nopl   (%rax)
    3f40:	42 0f 10 44 35 f0    	movups -0x10(%rbp,%r14,1),%xmm0
    3f46:	42 0f 10 4c 35 00    	movups 0x0(%rbp,%r14,1),%xmm1
    3f4c:	0f 58 c3             	addps  %xmm3,%xmm0
    3f4f:	0f 58 cb             	addps  %xmm3,%xmm1
    3f52:	42 0f 11 44 31 f0    	movups %xmm0,-0x10(%rcx,%r14,1)
    3f58:	42 0f 11 0c 31       	movups %xmm1,(%rcx,%r14,1)
    3f5d:	49 83 c6 20          	add    $0x20,%r14
    3f61:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    3f65:	75 d9                	jne    3f40 <main_graph_mobilenet+0x1c70>
    3f67:	4c 39 db             	cmp    %r11,%rbx
    3f6a:	4d 89 d9             	mov    %r11,%r9
    3f6d:	f3 0f 10 0d ab a1 00 	movss  0xa1ab(%rip),%xmm1        # e120 <_fini+0x1a4>
    3f74:	00 
    3f75:	4c 8b 74 24 40       	mov    0x40(%rsp),%r14
    3f7a:	4c 8b 7c 24 08       	mov    0x8(%rsp),%r15
    3f7f:	75 0f                	jne    3f90 <main_graph_mobilenet+0x1cc0>
    3f81:	eb 26                	jmp    3fa9 <main_graph_mobilenet+0x1cd9>
    3f83:	45 31 c9             	xor    %r9d,%r9d
    3f86:	4c 8b 74 24 40       	mov    0x40(%rsp),%r14
    3f8b:	4c 8b 7c 24 08       	mov    0x8(%rsp),%r15
    3f90:	f3 43 0f 10 04 8e    	movss  (%r14,%r9,4),%xmm0
    3f96:	f3 41 0f 58 c5       	addss  %xmm13,%xmm0
    3f9b:	f3 43 0f 11 04 8f    	movss  %xmm0,(%r15,%r9,4)
    3fa1:	49 ff c1             	inc    %r9
    3fa4:	4c 39 cb             	cmp    %r9,%rbx
    3fa7:	75 e7                	jne    3f90 <main_graph_mobilenet+0x1cc0>
    3fa9:	48 83 fb 08          	cmp    $0x8,%rbx
    3fad:	72 7a                	jb     4029 <main_graph_mobilenet+0x1d59>
    3faf:	4c 8b 8c 24 e0 01 00 	mov    0x1e0(%rsp),%r9
    3fb6:	00 
    3fb7:	4d 01 c1             	add    %r8,%r9
    3fba:	4c 8b b4 24 d8 01 00 	mov    0x1d8(%rsp),%r14
    3fc1:	00 
    3fc2:	4d 01 c6             	add    %r8,%r14
    3fc5:	4d 29 f1             	sub    %r14,%r9
    3fc8:	49 83 f9 20          	cmp    $0x20,%r9
    3fcc:	41 b9 00 00 00 00    	mov    $0x0,%r9d
    3fd2:	4c 8b bc 24 18 01 00 	mov    0x118(%rsp),%r15
    3fd9:	00 
    3fda:	72 64                	jb     4040 <main_graph_mobilenet+0x1d70>
    3fdc:	4d 89 d9             	mov    %r11,%r9
    3fdf:	4c 8b b4 24 70 02 00 	mov    0x270(%rsp),%r14
    3fe6:	00 
    3fe7:	0f 28 1d f2 a0 00 00 	movaps 0xa0f2(%rip),%xmm3        # e0e0 <_fini+0x164>
    3fee:	66 90                	xchg   %ax,%ax
    3ff0:	42 0f 10 44 35 f0    	movups -0x10(%rbp,%r14,1),%xmm0
    3ff6:	42 0f 10 4c 35 00    	movups 0x0(%rbp,%r14,1),%xmm1
    3ffc:	0f 58 c3             	addps  %xmm3,%xmm0
    3fff:	0f 58 cb             	addps  %xmm3,%xmm1
    4002:	42 0f 11 44 31 f0    	movups %xmm0,-0x10(%rcx,%r14,1)
    4008:	42 0f 11 0c 31       	movups %xmm1,(%rcx,%r14,1)
    400d:	49 83 c6 20          	add    $0x20,%r14
    4011:	49 83 c1 f8          	add    $0xfffffffffffffff8,%r9
    4015:	75 d9                	jne    3ff0 <main_graph_mobilenet+0x1d20>
    4017:	4c 39 db             	cmp    %r11,%rbx
    401a:	4d 89 d9             	mov    %r11,%r9
    401d:	f3 0f 10 0d fb a0 00 	movss  0xa0fb(%rip),%xmm1        # e120 <_fini+0x1a4>
    4024:	00 
    4025:	75 19                	jne    4040 <main_graph_mobilenet+0x1d70>
    4027:	eb 30                	jmp    4059 <main_graph_mobilenet+0x1d89>
    4029:	45 31 c9             	xor    %r9d,%r9d
    402c:	4c 8b bc 24 18 01 00 	mov    0x118(%rsp),%r15
    4033:	00 
    4034:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    403b:	00 00 00 
    403e:	66 90                	xchg   %ax,%ax
    4040:	f3 43 0f 10 04 8f    	movss  (%r15,%r9,4),%xmm0
    4046:	f3 41 0f 58 c6       	addss  %xmm14,%xmm0
    404b:	f3 43 0f 11 04 8c    	movss  %xmm0,(%r12,%r9,4)
    4051:	49 ff c1             	inc    %r9
    4054:	4c 39 cb             	cmp    %r9,%rbx
    4057:	75 e7                	jne    4040 <main_graph_mobilenet+0x1d70>
    4059:	48 83 fb 08          	cmp    $0x8,%rbx
    405d:	0f 82 b2 00 00 00    	jb     4115 <main_graph_mobilenet+0x1e45>
    4063:	4c 8b 8c 24 f0 01 00 	mov    0x1f0(%rsp),%r9
    406a:	00 
    406b:	4d 01 c1             	add    %r8,%r9
    406e:	4c 03 84 24 e8 01 00 	add    0x1e8(%rsp),%r8
    4075:	00 
    4076:	4d 29 c1             	sub    %r8,%r9
    4079:	49 83 f9 20          	cmp    $0x20,%r9
    407d:	41 b8 00 00 00 00    	mov    $0x0,%r8d
    4083:	4c 8b 74 24 20       	mov    0x20(%rsp),%r14
    4088:	4c 8b 8c 24 f0 00 00 	mov    0xf0(%rsp),%r9
    408f:	00 
    4090:	4c 8b 7c 24 10       	mov    0x10(%rsp),%r15
    4095:	4c 8b 24 24          	mov    (%rsp),%r12
    4099:	0f 82 91 00 00 00    	jb     4130 <main_graph_mobilenet+0x1e60>
    409f:	4d 89 d8             	mov    %r11,%r8
    40a2:	4c 8b 8c 24 78 02 00 	mov    0x278(%rsp),%r9
    40a9:	00 
    40aa:	0f 28 15 3f a0 00 00 	movaps 0xa03f(%rip),%xmm2        # e0f0 <_fini+0x174>
    40b1:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    40b8:	00 00 00 
    40bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    40c0:	42 0f 10 44 0d f0    	movups -0x10(%rbp,%r9,1),%xmm0
    40c6:	42 0f 10 4c 0d 00    	movups 0x0(%rbp,%r9,1),%xmm1
    40cc:	0f 58 c2             	addps  %xmm2,%xmm0
    40cf:	0f 58 ca             	addps  %xmm2,%xmm1
    40d2:	42 0f 11 44 09 f0    	movups %xmm0,-0x10(%rcx,%r9,1)
    40d8:	42 0f 11 0c 09       	movups %xmm1,(%rcx,%r9,1)
    40dd:	49 83 c1 20          	add    $0x20,%r9
    40e1:	49 83 c0 f8          	add    $0xfffffffffffffff8,%r8
    40e5:	75 d9                	jne    40c0 <main_graph_mobilenet+0x1df0>
    40e7:	4c 39 db             	cmp    %r11,%rbx
    40ea:	4d 89 d8             	mov    %r11,%r8
    40ed:	f3 0f 10 0d 2b a0 00 	movss  0xa02b(%rip),%xmm1        # e120 <_fini+0x1a4>
    40f4:	00 
    40f5:	4c 8b 8c 24 f0 00 00 	mov    0xf0(%rsp),%r9
    40fc:	00 
    40fd:	4c 8b 7c 24 10       	mov    0x10(%rsp),%r15
    4102:	4c 8b 24 24          	mov    (%rsp),%r12
    4106:	f3 0f 10 15 1a a0 00 	movss  0xa01a(%rip),%xmm2        # e128 <_fini+0x1ac>
    410d:	00 
    410e:	75 20                	jne    4130 <main_graph_mobilenet+0x1e60>
    4110:	e9 7b f1 ff ff       	jmp    3290 <main_graph_mobilenet+0xfc0>
    4115:	45 31 c0             	xor    %r8d,%r8d
    4118:	4c 8b 74 24 20       	mov    0x20(%rsp),%r14
    411d:	4c 8b 8c 24 f0 00 00 	mov    0xf0(%rsp),%r9
    4124:	00 
    4125:	4c 8b 7c 24 10       	mov    0x10(%rsp),%r15
    412a:	4c 8b 24 24          	mov    (%rsp),%r12
    412e:	66 90                	xchg   %ax,%ax
    4130:	f3 43 0f 10 04 87    	movss  (%r15,%r8,4),%xmm0
    4136:	f3 41 0f 58 c7       	addss  %xmm15,%xmm0
    413b:	f3 43 0f 11 04 84    	movss  %xmm0,(%r12,%r8,4)
    4141:	49 ff c0             	inc    %r8
    4144:	4c 39 c3             	cmp    %r8,%rbx
    4147:	75 e7                	jne    4130 <main_graph_mobilenet+0x1e60>
    4149:	e9 42 f1 ff ff       	jmp    3290 <main_graph_mobilenet+0xfc0>
    414e:	48 8b bc 24 08 02 00 	mov    0x208(%rsp),%rdi
    4155:	00 
    4156:	e8 b5 df ff ff       	call   2110 <free@plt>
    415b:	bf 30 00 00 00       	mov    $0x30,%edi
    4160:	e8 6b df ff ff       	call   20d0 <malloc@plt>
    4165:	48 89 c5             	mov    %rax,%rbp
    4168:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    416c:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    4170:	48 8b 84 24 10 02 00 	mov    0x210(%rsp),%rax
    4177:	00 
    4178:	48 8b 00             	mov    (%rax),%rax
    417b:	49 89 07             	mov    %rax,(%r15)
    417e:	48 8b bc 24 18 02 00 	mov    0x218(%rsp),%rdi
    4185:	00 
    4186:	e8 85 df ff ff       	call   2110 <free@plt>
    418b:	48 8b 84 24 e8 00 00 	mov    0xe8(%rsp),%rax
    4192:	00 
    4193:	48 0f af d8          	imul   %rax,%rbx
    4197:	48 c1 e3 04          	shl    $0x4,%rbx
    419b:	49 8b 0f             	mov    (%r15),%rcx
    419e:	48 85 c9             	test   %rcx,%rcx
    41a1:	48 0f 44 c8          	cmove  %rax,%rcx
    41a5:	48 89 c8             	mov    %rcx,%rax
    41a8:	48 c1 e0 0c          	shl    $0xc,%rax
    41ac:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    41b0:	48 8d 04 c0          	lea    (%rax,%rax,8),%rax
    41b4:	be 00 90 00 00       	mov    $0x9000,%esi
    41b9:	48 0f 45 f0          	cmovne %rax,%rsi
    41bd:	48 89 f0             	mov    %rsi,%rax
    41c0:	48 f7 d0             	not    %rax
    41c3:	48 c1 f8 3f          	sar    $0x3f,%rax
    41c7:	48 83 c8 01          	or     $0x1,%rax
    41cb:	48 29 d8             	sub    %rbx,%rax
    41ce:	48 99                	cqto   
    41d0:	48 f7 fe             	idiv   %rsi
    41d3:	49 89 c6             	mov    %rax,%r14
    41d6:	49 f7 d6             	not    %r14
    41d9:	48 89 d8             	mov    %rbx,%rax
    41dc:	48 99                	cqto   
    41de:	48 f7 fe             	idiv   %rsi
    41e1:	48 85 f6             	test   %rsi,%rsi
    41e4:	48 89 c2             	mov    %rax,%rdx
    41e7:	49 0f 48 d6          	cmovs  %r14,%rdx
    41eb:	48 85 db             	test   %rbx,%rbx
    41ee:	48 0f 4e d0          	cmovle %rax,%rdx
    41f2:	48 85 f6             	test   %rsi,%rsi
    41f5:	4c 0f 4e f2          	cmovle %rdx,%r14
    41f9:	48 85 db             	test   %rbx,%rbx
    41fc:	4c 0f 49 f2          	cmovns %rdx,%r14
    4200:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    4204:	4c 0f 45 f1          	cmovne %rcx,%r14
    4208:	4c 89 f0             	mov    %r14,%rax
    420b:	48 c1 e0 0a          	shl    $0xa,%rax
    420f:	48 8d 0c c0          	lea    (%rax,%rax,8),%rcx
    4213:	4c 89 f0             	mov    %r14,%rax
    4216:	48 c1 e0 04          	shl    $0x4,%rax
    421a:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    421e:	be 10 00 00 00       	mov    $0x10,%esi
    4223:	48 0f 45 f0          	cmovne %rax,%rsi
    4227:	48 89 f0             	mov    %rsi,%rax
    422a:	48 f7 d0             	not    %rax
    422d:	48 c1 f8 3f          	sar    $0x3f,%rax
    4231:	48 83 c8 01          	or     $0x1,%rax
    4235:	48 29 c8             	sub    %rcx,%rax
    4238:	48 99                	cqto   
    423a:	48 f7 fe             	idiv   %rsi
    423d:	49 89 c4             	mov    %rax,%r12
    4240:	49 f7 d4             	not    %r12
    4243:	48 89 c8             	mov    %rcx,%rax
    4246:	48 99                	cqto   
    4248:	48 f7 fe             	idiv   %rsi
    424b:	48 85 f6             	test   %rsi,%rsi
    424e:	48 89 c2             	mov    %rax,%rdx
    4251:	49 0f 48 d4          	cmovs  %r12,%rdx
    4255:	48 85 c9             	test   %rcx,%rcx
    4258:	48 0f 4e d0          	cmovle %rax,%rdx
    425c:	48 85 f6             	test   %rsi,%rsi
    425f:	4c 0f 4e e2          	cmovle %rdx,%r12
    4263:	48 85 c9             	test   %rcx,%rcx
    4266:	4c 0f 49 e2          	cmovns %rdx,%r12
    426a:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    426e:	49 c7 47 08 10 00 00 	movq   $0x10,0x8(%r15)
    4275:	00 
    4276:	49 c7 47 10 30 00 00 	movq   $0x30,0x10(%r15)
    427d:	00 
    427e:	49 c7 47 18 30 00 00 	movq   $0x30,0x18(%r15)
    4285:	00 
    4286:	4c 89 f0             	mov    %r14,%rax
    4289:	49 0f 44 c4          	cmove  %r12,%rax
    428d:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    4292:	48 89 ef             	mov    %rbp,%rdi
    4295:	e8 76 de ff ff       	call   2110 <free@plt>
    429a:	4c 89 f0             	mov    %r14,%rax
    429d:	48 c1 e0 0e          	shl    $0xe,%rax
    42a1:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    42a6:	e8 25 de ff ff       	call   20d0 <malloc@plt>
    42ab:	48 89 c3             	mov    %rax,%rbx
    42ae:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    42b2:	48 89 44 24 08       	mov    %rax,0x8(%rsp)
    42b7:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    42bb:	4c 89 7c 24 30       	mov    %r15,0x30(%rsp)
    42c0:	bf 04 00 00 00       	mov    $0x4,%edi
    42c5:	e8 d6 de ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    42ca:	48 89 c5             	mov    %rax,%rbp
    42cd:	48 89 c7             	mov    %rax,%rdi
    42d0:	31 f6                	xor    %esi,%esi
    42d2:	48 89 da             	mov    %rbx,%rdx
    42d5:	4c 89 f9             	mov    %r15,%rcx
    42d8:	e8 13 de ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    42dd:	be 01 00 00 00       	mov    $0x1,%esi
    42e2:	48 89 ef             	mov    %rbp,%rdi
    42e5:	e8 b6 dd ff ff       	call   20a0 <omTensorSetDataType@plt>
    42ea:	48 89 ef             	mov    %rbp,%rdi
    42ed:	e8 de de ff ff       	call   21d0 <omTensorGetShape@plt>
    42f2:	49 89 c7             	mov    %rax,%r15
    42f5:	48 89 ef             	mov    %rbp,%rdi
    42f8:	e8 73 dd ff ff       	call   2070 <omTensorGetStrides@plt>
    42fd:	4d 89 37             	mov    %r14,(%r15)
    4300:	48 c7 00 00 90 00 00 	movq   $0x9000,(%rax)
    4307:	49 c7 47 08 10 00 00 	movq   $0x10,0x8(%r15)
    430e:	00 
    430f:	48 c7 40 08 00 09 00 	movq   $0x900,0x8(%rax)
    4316:	00 
    4317:	49 c7 47 10 30 00 00 	movq   $0x30,0x10(%r15)
    431e:	00 
    431f:	48 c7 40 10 30 00 00 	movq   $0x30,0x10(%rax)
    4326:	00 
    4327:	49 c7 47 18 30 00 00 	movq   $0x30,0x18(%r15)
    432e:	00 
    432f:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    4336:	00 
    4337:	bf 04 00 00 00       	mov    $0x4,%edi
    433c:	e8 5f de ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    4341:	49 89 c7             	mov    %rax,%r15
    4344:	48 89 c7             	mov    %rax,%rdi
    4347:	31 f6                	xor    %esi,%esi
    4349:	48 8b 9c 24 20 02 00 	mov    0x220(%rsp),%rbx
    4350:	00 
    4351:	48 89 da             	mov    %rbx,%rdx
    4354:	48 8b 8c 24 20 01 00 	mov    0x120(%rsp),%rcx
    435b:	00 
    435c:	e8 8f dd ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    4361:	be 01 00 00 00       	mov    $0x1,%esi
    4366:	4c 89 ff             	mov    %r15,%rdi
    4369:	e8 32 dd ff ff       	call   20a0 <omTensorSetDataType@plt>
    436e:	4c 89 ff             	mov    %r15,%rdi
    4371:	e8 5a de ff ff       	call   21d0 <omTensorGetShape@plt>
    4376:	49 89 c5             	mov    %rax,%r13
    4379:	4c 89 ff             	mov    %r15,%rdi
    437c:	e8 ef dc ff ff       	call   2070 <omTensorGetStrides@plt>
    4381:	4d 89 75 00          	mov    %r14,0x0(%r13)
    4385:	48 c7 00 00 90 00 00 	movq   $0x9000,(%rax)
    438c:	49 c7 45 08 10 00 00 	movq   $0x10,0x8(%r13)
    4393:	00 
    4394:	48 c7 40 08 00 09 00 	movq   $0x900,0x8(%rax)
    439b:	00 
    439c:	49 c7 45 10 30 00 00 	movq   $0x30,0x10(%r13)
    43a3:	00 
    43a4:	48 c7 40 10 30 00 00 	movq   $0x30,0x10(%rax)
    43ab:	00 
    43ac:	49 c7 45 18 30 00 00 	movq   $0x30,0x18(%r13)
    43b3:	00 
    43b4:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    43bb:	00 
    43bc:	48 89 ef             	mov    %rbp,%rdi
    43bf:	4c 89 fe             	mov    %r15,%rsi
    43c2:	e8 29 de ff ff       	call   21f0 <Relu@plt>
    43c7:	48 89 ef             	mov    %rbp,%rdi
    43ca:	e8 e1 dd ff ff       	call   21b0 <omTensorDestroy@plt>
    43cf:	4c 89 ff             	mov    %r15,%rdi
    43d2:	e8 d9 dd ff ff       	call   21b0 <omTensorDestroy@plt>
    43d7:	48 89 df             	mov    %rbx,%rdi
    43da:	e8 31 dd ff ff       	call   2110 <free@plt>
    43df:	4c 89 f0             	mov    %r14,%rax
    43e2:	48 c1 e0 0c          	shl    $0xc,%rax
    43e6:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    43eb:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    43f0:	e8 db dc ff ff       	call   20d0 <malloc@plt>
    43f5:	48 89 c3             	mov    %rax,%rbx
    43f8:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    43fc:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    4401:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    4405:	4c 89 3c 24          	mov    %r15,(%rsp)
    4409:	bf 04 00 00 00       	mov    $0x4,%edi
    440e:	e8 8d dd ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    4413:	49 89 c5             	mov    %rax,%r13
    4416:	48 89 c7             	mov    %rax,%rdi
    4419:	31 f6                	xor    %esi,%esi
    441b:	48 89 da             	mov    %rbx,%rdx
    441e:	4c 89 f9             	mov    %r15,%rcx
    4421:	e8 ca dc ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    4426:	be 01 00 00 00       	mov    $0x1,%esi
    442b:	4c 89 eb             	mov    %r13,%rbx
    442e:	4c 89 6c 24 50       	mov    %r13,0x50(%rsp)
    4433:	4c 89 ef             	mov    %r13,%rdi
    4436:	e8 65 dc ff ff       	call   20a0 <omTensorSetDataType@plt>
    443b:	4c 89 ef             	mov    %r13,%rdi
    443e:	e8 8d dd ff ff       	call   21d0 <omTensorGetShape@plt>
    4443:	49 89 c5             	mov    %rax,%r13
    4446:	48 89 df             	mov    %rbx,%rdi
    4449:	e8 22 dc ff ff       	call   2070 <omTensorGetStrides@plt>
    444e:	4d 89 75 00          	mov    %r14,0x0(%r13)
    4452:	4d 89 f7             	mov    %r14,%r15
    4455:	4c 89 74 24 40       	mov    %r14,0x40(%rsp)
    445a:	48 c7 00 00 24 00 00 	movq   $0x2400,(%rax)
    4461:	49 c7 45 08 10 00 00 	movq   $0x10,0x8(%r13)
    4468:	00 
    4469:	48 c7 40 08 40 02 00 	movq   $0x240,0x8(%rax)
    4470:	00 
    4471:	49 c7 45 10 18 00 00 	movq   $0x18,0x10(%r13)
    4478:	00 
    4479:	48 c7 40 10 18 00 00 	movq   $0x18,0x10(%rax)
    4480:	00 
    4481:	49 c7 45 18 18 00 00 	movq   $0x18,0x18(%r13)
    4488:	00 
    4489:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    4490:	00 
    4491:	bf 04 00 00 00       	mov    $0x4,%edi
    4496:	e8 05 dd ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    449b:	48 89 c5             	mov    %rax,%rbp
    449e:	48 89 c7             	mov    %rax,%rdi
    44a1:	31 f6                	xor    %esi,%esi
    44a3:	48 8b 5c 24 08       	mov    0x8(%rsp),%rbx
    44a8:	48 89 da             	mov    %rbx,%rdx
    44ab:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    44b0:	e8 3b dc ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    44b5:	be 01 00 00 00       	mov    $0x1,%esi
    44ba:	48 89 ef             	mov    %rbp,%rdi
    44bd:	e8 de db ff ff       	call   20a0 <omTensorSetDataType@plt>
    44c2:	48 89 ef             	mov    %rbp,%rdi
    44c5:	e8 06 dd ff ff       	call   21d0 <omTensorGetShape@plt>
    44ca:	49 89 c6             	mov    %rax,%r14
    44cd:	48 89 ef             	mov    %rbp,%rdi
    44d0:	e8 9b db ff ff       	call   2070 <omTensorGetStrides@plt>
    44d5:	4d 89 3e             	mov    %r15,(%r14)
    44d8:	48 c7 00 00 90 00 00 	movq   $0x9000,(%rax)
    44df:	49 c7 46 08 10 00 00 	movq   $0x10,0x8(%r14)
    44e6:	00 
    44e7:	48 c7 40 08 00 09 00 	movq   $0x900,0x8(%rax)
    44ee:	00 
    44ef:	49 c7 46 10 30 00 00 	movq   $0x30,0x10(%r14)
    44f6:	00 
    44f7:	48 c7 40 10 30 00 00 	movq   $0x30,0x10(%rax)
    44fe:	00 
    44ff:	49 c7 46 18 30 00 00 	movq   $0x30,0x18(%r14)
    4506:	00 
    4507:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    450e:	00 
    450f:	4c 8d 2d 1a 54 01 00 	lea    0x1541a(%rip),%r13        # 19930 <constant_54_mobilenet>
    4516:	bf 04 00 00 00       	mov    $0x4,%edi
    451b:	e8 80 dc ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    4520:	49 89 c6             	mov    %rax,%r14
    4523:	48 89 c7             	mov    %rax,%rdi
    4526:	31 f6                	xor    %esi,%esi
    4528:	4c 89 ea             	mov    %r13,%rdx
    452b:	4c 89 e9             	mov    %r13,%rcx
    452e:	e8 bd db ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    4533:	be 01 00 00 00       	mov    $0x1,%esi
    4538:	4c 89 f7             	mov    %r14,%rdi
    453b:	e8 60 db ff ff       	call   20a0 <omTensorSetDataType@plt>
    4540:	4c 89 f7             	mov    %r14,%rdi
    4543:	e8 88 dc ff ff       	call   21d0 <omTensorGetShape@plt>
    4548:	49 89 c5             	mov    %rax,%r13
    454b:	4c 89 f7             	mov    %r14,%rdi
    454e:	e8 1d db ff ff       	call   2070 <omTensorGetStrides@plt>
    4553:	49 c7 45 00 10 00 00 	movq   $0x10,0x0(%r13)
    455a:	00 
    455b:	48 c7 00 09 00 00 00 	movq   $0x9,(%rax)
    4562:	49 c7 45 08 01 00 00 	movq   $0x1,0x8(%r13)
    4569:	00 
    456a:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    4571:	00 
    4572:	49 c7 45 10 03 00 00 	movq   $0x3,0x10(%r13)
    4579:	00 
    457a:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    4581:	00 
    4582:	49 c7 45 18 03 00 00 	movq   $0x3,0x18(%r13)
    4589:	00 
    458a:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    4591:	00 
    4592:	4c 8d 3d 37 68 01 00 	lea    0x16837(%rip),%r15        # 1add0 <constant_42_mobilenet>
    4599:	bf 01 00 00 00       	mov    $0x1,%edi
    459e:	e8 fd db ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    45a3:	49 89 c5             	mov    %rax,%r13
    45a6:	48 89 c7             	mov    %rax,%rdi
    45a9:	31 f6                	xor    %esi,%esi
    45ab:	4c 89 fa             	mov    %r15,%rdx
    45ae:	4c 89 f9             	mov    %r15,%rcx
    45b1:	e8 3a db ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    45b6:	be 01 00 00 00       	mov    $0x1,%esi
    45bb:	4c 89 ef             	mov    %r13,%rdi
    45be:	e8 dd da ff ff       	call   20a0 <omTensorSetDataType@plt>
    45c3:	4c 89 ef             	mov    %r13,%rdi
    45c6:	e8 05 dc ff ff       	call   21d0 <omTensorGetShape@plt>
    45cb:	49 89 c7             	mov    %rax,%r15
    45ce:	4c 89 ef             	mov    %r13,%rdi
    45d1:	e8 9a da ff ff       	call   2070 <omTensorGetStrides@plt>
    45d6:	49 c7 07 10 00 00 00 	movq   $0x10,(%r15)
    45dd:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    45e4:	4c 8b 7c 24 50       	mov    0x50(%rsp),%r15
    45e9:	4c 89 ff             	mov    %r15,%rdi
    45ec:	48 89 ee             	mov    %rbp,%rsi
    45ef:	4c 89 f2             	mov    %r14,%rdx
    45f2:	4c 89 e9             	mov    %r13,%rcx
    45f5:	45 31 c0             	xor    %r8d,%r8d
    45f8:	45 31 c9             	xor    %r9d,%r9d
    45fb:	6a 02                	push   $0x2
    45fd:	6a 02                	push   $0x2
    45ff:	6a 01                	push   $0x1
    4601:	6a 01                	push   $0x1
    4603:	e8 58 db ff ff       	call   2160 <Conv2D_bias@plt>
    4608:	48 83 c4 20          	add    $0x20,%rsp
    460c:	4c 89 ff             	mov    %r15,%rdi
    460f:	e8 9c db ff ff       	call   21b0 <omTensorDestroy@plt>
    4614:	48 89 ef             	mov    %rbp,%rdi
    4617:	e8 94 db ff ff       	call   21b0 <omTensorDestroy@plt>
    461c:	4c 89 f7             	mov    %r14,%rdi
    461f:	e8 8c db ff ff       	call   21b0 <omTensorDestroy@plt>
    4624:	4c 89 ef             	mov    %r13,%rdi
    4627:	e8 84 db ff ff       	call   21b0 <omTensorDestroy@plt>
    462c:	48 89 df             	mov    %rbx,%rdi
    462f:	e8 dc da ff ff       	call   2110 <free@plt>
    4634:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    4639:	e8 92 da ff ff       	call   20d0 <malloc@plt>
    463e:	48 89 c3             	mov    %rax,%rbx
    4641:	4c 8d 70 0f          	lea    0xf(%rax),%r14
    4645:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    464a:	49 83 e6 f0          	and    $0xfffffffffffffff0,%r14
    464e:	4c 89 74 24 30       	mov    %r14,0x30(%rsp)
    4653:	bf 04 00 00 00       	mov    $0x4,%edi
    4658:	e8 43 db ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    465d:	48 89 c5             	mov    %rax,%rbp
    4660:	48 89 c7             	mov    %rax,%rdi
    4663:	31 f6                	xor    %esi,%esi
    4665:	48 89 da             	mov    %rbx,%rdx
    4668:	4c 89 f1             	mov    %r14,%rcx
    466b:	e8 80 da ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    4670:	be 01 00 00 00       	mov    $0x1,%esi
    4675:	48 89 ef             	mov    %rbp,%rdi
    4678:	e8 23 da ff ff       	call   20a0 <omTensorSetDataType@plt>
    467d:	48 89 ef             	mov    %rbp,%rdi
    4680:	e8 4b db ff ff       	call   21d0 <omTensorGetShape@plt>
    4685:	49 89 c6             	mov    %rax,%r14
    4688:	48 89 ef             	mov    %rbp,%rdi
    468b:	e8 e0 d9 ff ff       	call   2070 <omTensorGetStrides@plt>
    4690:	4c 8b 7c 24 40       	mov    0x40(%rsp),%r15
    4695:	4d 89 3e             	mov    %r15,(%r14)
    4698:	48 c7 00 00 24 00 00 	movq   $0x2400,(%rax)
    469f:	49 c7 46 08 10 00 00 	movq   $0x10,0x8(%r14)
    46a6:	00 
    46a7:	48 c7 40 08 40 02 00 	movq   $0x240,0x8(%rax)
    46ae:	00 
    46af:	49 c7 46 10 18 00 00 	movq   $0x18,0x10(%r14)
    46b6:	00 
    46b7:	48 c7 40 10 18 00 00 	movq   $0x18,0x10(%rax)
    46be:	00 
    46bf:	49 c7 46 18 18 00 00 	movq   $0x18,0x18(%r14)
    46c6:	00 
    46c7:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    46ce:	00 
    46cf:	bf 04 00 00 00       	mov    $0x4,%edi
    46d4:	e8 c7 da ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    46d9:	49 89 c6             	mov    %rax,%r14
    46dc:	48 89 c7             	mov    %rax,%rdi
    46df:	31 f6                	xor    %esi,%esi
    46e1:	48 8b 5c 24 20       	mov    0x20(%rsp),%rbx
    46e6:	48 89 da             	mov    %rbx,%rdx
    46e9:	48 8b 0c 24          	mov    (%rsp),%rcx
    46ed:	e8 fe d9 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    46f2:	be 01 00 00 00       	mov    $0x1,%esi
    46f7:	4c 89 f7             	mov    %r14,%rdi
    46fa:	e8 a1 d9 ff ff       	call   20a0 <omTensorSetDataType@plt>
    46ff:	4c 89 f7             	mov    %r14,%rdi
    4702:	e8 c9 da ff ff       	call   21d0 <omTensorGetShape@plt>
    4707:	49 89 c5             	mov    %rax,%r13
    470a:	4c 89 f7             	mov    %r14,%rdi
    470d:	e8 5e d9 ff ff       	call   2070 <omTensorGetStrides@plt>
    4712:	4d 89 7d 00          	mov    %r15,0x0(%r13)
    4716:	48 c7 00 00 24 00 00 	movq   $0x2400,(%rax)
    471d:	49 c7 45 08 10 00 00 	movq   $0x10,0x8(%r13)
    4724:	00 
    4725:	48 c7 40 08 40 02 00 	movq   $0x240,0x8(%rax)
    472c:	00 
    472d:	49 c7 45 10 18 00 00 	movq   $0x18,0x10(%r13)
    4734:	00 
    4735:	48 c7 40 10 18 00 00 	movq   $0x18,0x10(%rax)
    473c:	00 
    473d:	49 c7 45 18 18 00 00 	movq   $0x18,0x18(%r13)
    4744:	00 
    4745:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    474c:	00 
    474d:	48 89 ef             	mov    %rbp,%rdi
    4750:	4c 89 f6             	mov    %r14,%rsi
    4753:	e8 98 da ff ff       	call   21f0 <Relu@plt>
    4758:	48 89 ef             	mov    %rbp,%rdi
    475b:	e8 50 da ff ff       	call   21b0 <omTensorDestroy@plt>
    4760:	4c 89 f7             	mov    %r14,%rdi
    4763:	e8 48 da ff ff       	call   21b0 <omTensorDestroy@plt>
    4768:	48 89 df             	mov    %rbx,%rdi
    476b:	e8 a0 d9 ff ff       	call   2110 <free@plt>
    4770:	bf 18 00 00 00       	mov    $0x18,%edi
    4775:	e8 56 d9 ff ff       	call   20d0 <malloc@plt>
    477a:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    477f:	48 83 c0 0f          	add    $0xf,%rax
    4783:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    4787:	4c 89 fb             	mov    %r15,%rbx
    478a:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    478f:	4c 89 38             	mov    %r15,(%rax)
    4792:	bf 28 00 00 00       	mov    $0x28,%edi
    4797:	e8 34 d9 ff ff       	call   20d0 <malloc@plt>
    479c:	48 8d 48 0f          	lea    0xf(%rax),%rcx
    47a0:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
    47a4:	4c 89 39             	mov    %r15,(%rcx)
    47a7:	48 c7 41 08 10 00 00 	movq   $0x10,0x8(%rcx)
    47ae:	00 
    47af:	48 c7 41 10 ff ff ff 	movq   $0xffffffffffffffff,0x10(%rcx)
    47b6:	ff 
    47b7:	48 89 c7             	mov    %rax,%rdi
    47ba:	e8 51 d9 ff ff       	call   2110 <free@plt>
    47bf:	4c 89 e0             	mov    %r12,%rax
    47c2:	48 c1 e0 04          	shl    $0x4,%rax
    47c6:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    47cb:	4c 89 e3             	mov    %r12,%rbx
    47ce:	48 c1 e3 05          	shl    $0x5,%rbx
    47d2:	48 8b 6c 24 60       	mov    0x60(%rsp),%rbp
    47d7:	48 89 e8             	mov    %rbp,%rax
    47da:	48 0f af c3          	imul   %rbx,%rax
    47de:	48 8d 3c 85 10 00 00 	lea    0x10(,%rax,4),%rdi
    47e5:	00 
    47e6:	48 89 3c 24          	mov    %rdi,(%rsp)
    47ea:	e8 e1 d8 ff ff       	call   20d0 <malloc@plt>
    47ef:	49 89 c6             	mov    %rax,%r14
    47f2:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    47f6:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    47fa:	bf 03 00 00 00       	mov    $0x3,%edi
    47ff:	e8 9c d9 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    4804:	49 89 c5             	mov    %rax,%r13
    4807:	48 89 c7             	mov    %rax,%rdi
    480a:	31 f6                	xor    %esi,%esi
    480c:	4c 89 74 24 70       	mov    %r14,0x70(%rsp)
    4811:	4c 89 f2             	mov    %r14,%rdx
    4814:	4c 89 7c 24 08       	mov    %r15,0x8(%rsp)
    4819:	4c 89 f9             	mov    %r15,%rcx
    481c:	e8 cf d8 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    4821:	be 01 00 00 00       	mov    $0x1,%esi
    4826:	4c 89 ef             	mov    %r13,%rdi
    4829:	e8 72 d8 ff ff       	call   20a0 <omTensorSetDataType@plt>
    482e:	4c 89 ef             	mov    %r13,%rdi
    4831:	e8 9a d9 ff ff       	call   21d0 <omTensorGetShape@plt>
    4836:	49 89 c6             	mov    %rax,%r14
    4839:	4c 89 ef             	mov    %r13,%rdi
    483c:	e8 2f d8 ff ff       	call   2070 <omTensorGetStrides@plt>
    4841:	49 89 2e             	mov    %rbp,(%r14)
    4844:	48 89 18             	mov    %rbx,(%rax)
    4847:	49 c7 46 08 20 00 00 	movq   $0x20,0x8(%r14)
    484e:	00 
    484f:	4c 89 60 08          	mov    %r12,0x8(%rax)
    4853:	4d 89 66 10          	mov    %r12,0x10(%r14)
    4857:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    485e:	00 
    485f:	bf 02 00 00 00       	mov    $0x2,%edi
    4864:	e8 37 d9 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    4869:	49 89 c6             	mov    %rax,%r14
    486c:	48 8d 0d 9d 6e 0d 00 	lea    0xd6e9d(%rip),%rcx        # db710 <constant_12_mobilenet>
    4873:	48 89 c7             	mov    %rax,%rdi
    4876:	31 f6                	xor    %esi,%esi
    4878:	48 89 ca             	mov    %rcx,%rdx
    487b:	e8 70 d8 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    4880:	be 01 00 00 00       	mov    $0x1,%esi
    4885:	4c 89 f7             	mov    %r14,%rdi
    4888:	e8 13 d8 ff ff       	call   20a0 <omTensorSetDataType@plt>
    488d:	4c 89 f7             	mov    %r14,%rdi
    4890:	e8 3b d9 ff ff       	call   21d0 <omTensorGetShape@plt>
    4895:	48 89 c3             	mov    %rax,%rbx
    4898:	4c 89 f7             	mov    %r14,%rdi
    489b:	e8 d0 d7 ff ff       	call   2070 <omTensorGetStrides@plt>
    48a0:	48 c7 03 20 00 00 00 	movq   $0x20,(%rbx)
    48a7:	48 c7 00 10 00 00 00 	movq   $0x10,(%rax)
    48ae:	48 c7 43 08 10 00 00 	movq   $0x10,0x8(%rbx)
    48b5:	00 
    48b6:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    48bd:	00 
    48be:	bf 03 00 00 00       	mov    $0x3,%edi
    48c3:	e8 d8 d8 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    48c8:	48 89 c3             	mov    %rax,%rbx
    48cb:	48 89 c7             	mov    %rax,%rdi
    48ce:	31 f6                	xor    %esi,%esi
    48d0:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    48d5:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    48da:	e8 11 d8 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    48df:	be 01 00 00 00       	mov    $0x1,%esi
    48e4:	48 89 df             	mov    %rbx,%rdi
    48e7:	e8 b4 d7 ff ff       	call   20a0 <omTensorSetDataType@plt>
    48ec:	48 89 df             	mov    %rbx,%rdi
    48ef:	e8 dc d8 ff ff       	call   21d0 <omTensorGetShape@plt>
    48f4:	49 89 c7             	mov    %rax,%r15
    48f7:	48 89 df             	mov    %rbx,%rdi
    48fa:	e8 71 d7 ff ff       	call   2070 <omTensorGetStrides@plt>
    48ff:	49 89 2f             	mov    %rbp,(%r15)
    4902:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    4907:	48 89 08             	mov    %rcx,(%rax)
    490a:	49 c7 47 08 10 00 00 	movq   $0x10,0x8(%r15)
    4911:	00 
    4912:	4c 89 60 08          	mov    %r12,0x8(%rax)
    4916:	4d 89 67 10          	mov    %r12,0x10(%r15)
    491a:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    4921:	00 
    4922:	4c 89 ef             	mov    %r13,%rdi
    4925:	4c 89 f6             	mov    %r14,%rsi
    4928:	48 89 da             	mov    %rbx,%rdx
    492b:	e8 20 d7 ff ff       	call   2050 <MatMul@plt>
    4930:	4c 89 ef             	mov    %r13,%rdi
    4933:	e8 78 d8 ff ff       	call   21b0 <omTensorDestroy@plt>
    4938:	4c 89 f7             	mov    %r14,%rdi
    493b:	e8 70 d8 ff ff       	call   21b0 <omTensorDestroy@plt>
    4940:	48 89 df             	mov    %rbx,%rdi
    4943:	e8 68 d8 ff ff       	call   21b0 <omTensorDestroy@plt>
    4948:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    494d:	e8 be d7 ff ff       	call   2110 <free@plt>
    4952:	48 8b 3c 24          	mov    (%rsp),%rdi
    4956:	e8 75 d7 ff ff       	call   20d0 <malloc@plt>
    495b:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    4960:	48 83 c0 0f          	add    $0xf,%rax
    4964:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    4968:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    496d:	48 85 ed             	test   %rbp,%rbp
    4970:	0f 8e 58 01 00 00    	jle    4ace <main_graph_mobilenet+0x27fe>
    4976:	4c 89 e0             	mov    %r12,%rax
    4979:	48 c1 e0 07          	shl    $0x7,%rax
    497d:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    4982:	4a 8d 0c a5 00 00 00 	lea    0x0(,%r12,4),%rcx
    4989:	00 
    498a:	4c 89 e3             	mov    %r12,%rbx
    498d:	48 83 e3 f8          	and    $0xfffffffffffffff8,%rbx
    4991:	48 8b 6c 24 40       	mov    0x40(%rsp),%rbp
    4996:	48 8d 45 10          	lea    0x10(%rbp),%rax
    499a:	48 89 04 24          	mov    %rax,(%rsp)
    499e:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    49a3:	48 8d 50 10          	lea    0x10(%rax),%rdx
    49a7:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
    49ac:	45 31 c0             	xor    %r8d,%r8d
    49af:	4c 8d 0d da 7c 01 00 	lea    0x17cda(%rip),%r9        # 1c690 <constant_25_mobilenet>
    49b6:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    49bb:	eb 27                	jmp    49e4 <main_graph_mobilenet+0x2714>
    49bd:	0f 1f 00             	nopl   (%rax)
    49c0:	49 ff c0             	inc    %r8
    49c3:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    49c8:	48 01 04 24          	add    %rax,(%rsp)
    49cc:	48 01 44 24 20       	add    %rax,0x20(%rsp)
    49d1:	48 01 c5             	add    %rax,%rbp
    49d4:	48 01 44 24 30       	add    %rax,0x30(%rsp)
    49d9:	4c 3b 44 24 60       	cmp    0x60(%rsp),%r8
    49de:	0f 84 ea 00 00 00    	je     4ace <main_graph_mobilenet+0x27fe>
    49e4:	4d 85 e4             	test   %r12,%r12
    49e7:	7e d7                	jle    49c0 <main_graph_mobilenet+0x26f0>
    49e9:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
    49ee:	4d 0f af f0          	imul   %r8,%r14
    49f2:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    49f7:	4e 8d 3c 30          	lea    (%rax,%r14,1),%r15
    49fb:	4c 03 74 24 08       	add    0x8(%rsp),%r14
    4a00:	4c 8b 6c 24 30       	mov    0x30(%rsp),%r13
    4a05:	49 89 eb             	mov    %rbp,%r11
    4a08:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    4a0d:	48 8b 34 24          	mov    (%rsp),%rsi
    4a11:	31 c0                	xor    %eax,%eax
    4a13:	eb 20                	jmp    4a35 <main_graph_mobilenet+0x2765>
    4a15:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    4a1c:	00 00 00 
    4a1f:	90                   	nop
    4a20:	48 ff c0             	inc    %rax
    4a23:	48 01 ce             	add    %rcx,%rsi
    4a26:	48 01 cf             	add    %rcx,%rdi
    4a29:	49 01 cb             	add    %rcx,%r11
    4a2c:	49 01 cd             	add    %rcx,%r13
    4a2f:	48 83 f8 20          	cmp    $0x20,%rax
    4a33:	74 8b                	je     49c0 <main_graph_mobilenet+0x26f0>
    4a35:	49 83 fc 08          	cmp    $0x8,%r12
    4a39:	f3 41 0f 10 04 81    	movss  (%r9,%rax,4),%xmm0
    4a3f:	73 0f                	jae    4a50 <main_graph_mobilenet+0x2780>
    4a41:	31 d2                	xor    %edx,%edx
    4a43:	eb 6b                	jmp    4ab0 <main_graph_mobilenet+0x27e0>
    4a45:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    4a4c:	00 00 00 
    4a4f:	90                   	nop
    4a50:	48 89 ca             	mov    %rcx,%rdx
    4a53:	48 0f af d0          	imul   %rax,%rdx
    4a57:	4d 8d 14 16          	lea    (%r14,%rdx,1),%r10
    4a5b:	4c 01 fa             	add    %r15,%rdx
    4a5e:	4c 29 d2             	sub    %r10,%rdx
    4a61:	48 83 fa 20          	cmp    $0x20,%rdx
    4a65:	73 09                	jae    4a70 <main_graph_mobilenet+0x27a0>
    4a67:	31 d2                	xor    %edx,%edx
    4a69:	eb 45                	jmp    4ab0 <main_graph_mobilenet+0x27e0>
    4a6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    4a70:	0f 28 c8             	movaps %xmm0,%xmm1
    4a73:	0f c6 c8 00          	shufps $0x0,%xmm0,%xmm1
    4a77:	31 d2                	xor    %edx,%edx
    4a79:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    4a80:	0f 10 54 97 f0       	movups -0x10(%rdi,%rdx,4),%xmm2
    4a85:	0f 10 1c 97          	movups (%rdi,%rdx,4),%xmm3
    4a89:	0f 58 d1             	addps  %xmm1,%xmm2
    4a8c:	0f 58 d9             	addps  %xmm1,%xmm3
    4a8f:	0f 11 54 96 f0       	movups %xmm2,-0x10(%rsi,%rdx,4)
    4a94:	0f 11 1c 96          	movups %xmm3,(%rsi,%rdx,4)
    4a98:	48 83 c2 08          	add    $0x8,%rdx
    4a9c:	48 39 d3             	cmp    %rdx,%rbx
    4a9f:	75 df                	jne    4a80 <main_graph_mobilenet+0x27b0>
    4aa1:	49 39 dc             	cmp    %rbx,%r12
    4aa4:	48 89 da             	mov    %rbx,%rdx
    4aa7:	0f 84 73 ff ff ff    	je     4a20 <main_graph_mobilenet+0x2750>
    4aad:	0f 1f 00             	nopl   (%rax)
    4ab0:	f3 41 0f 10 4c 95 00 	movss  0x0(%r13,%rdx,4),%xmm1
    4ab7:	f3 0f 58 c8          	addss  %xmm0,%xmm1
    4abb:	f3 41 0f 11 0c 93    	movss  %xmm1,(%r11,%rdx,4)
    4ac1:	48 ff c2             	inc    %rdx
    4ac4:	49 39 d4             	cmp    %rdx,%r12
    4ac7:	75 e7                	jne    4ab0 <main_graph_mobilenet+0x27e0>
    4ac9:	e9 52 ff ff ff       	jmp    4a20 <main_graph_mobilenet+0x2750>
    4ace:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    4ad3:	e8 38 d6 ff ff       	call   2110 <free@plt>
    4ad8:	bf 30 00 00 00       	mov    $0x30,%edi
    4add:	e8 ee d5 ff ff       	call   20d0 <malloc@plt>
    4ae2:	49 89 c7             	mov    %rax,%r15
    4ae5:	4c 8d 68 0f          	lea    0xf(%rax),%r13
    4ae9:	49 83 e5 f0          	and    $0xfffffffffffffff0,%r13
    4aed:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    4af2:	48 8b 00             	mov    (%rax),%rax
    4af5:	49 89 45 00          	mov    %rax,0x0(%r13)
    4af9:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    4afe:	e8 0d d6 ff ff       	call   2110 <free@plt>
    4b03:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    4b08:	4c 0f af e0          	imul   %rax,%r12
    4b0c:	49 c1 e4 05          	shl    $0x5,%r12
    4b10:	49 8b 4d 00          	mov    0x0(%r13),%rcx
    4b14:	48 85 c9             	test   %rcx,%rcx
    4b17:	48 0f 44 c8          	cmove  %rax,%rcx
    4b1b:	48 89 c8             	mov    %rcx,%rax
    4b1e:	48 c1 e0 0b          	shl    $0xb,%rax
    4b22:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    4b26:	48 8d 34 c0          	lea    (%rax,%rax,8),%rsi
    4b2a:	b8 00 48 00 00       	mov    $0x4800,%eax
    4b2f:	48 0f 44 f0          	cmove  %rax,%rsi
    4b33:	48 89 f0             	mov    %rsi,%rax
    4b36:	48 f7 d0             	not    %rax
    4b39:	48 c1 f8 3f          	sar    $0x3f,%rax
    4b3d:	48 83 c8 01          	or     $0x1,%rax
    4b41:	4c 29 e0             	sub    %r12,%rax
    4b44:	48 99                	cqto   
    4b46:	48 f7 fe             	idiv   %rsi
    4b49:	49 89 c6             	mov    %rax,%r14
    4b4c:	49 f7 d6             	not    %r14
    4b4f:	4c 89 e0             	mov    %r12,%rax
    4b52:	48 99                	cqto   
    4b54:	48 f7 fe             	idiv   %rsi
    4b57:	48 85 f6             	test   %rsi,%rsi
    4b5a:	48 89 c2             	mov    %rax,%rdx
    4b5d:	49 0f 48 d6          	cmovs  %r14,%rdx
    4b61:	4d 85 e4             	test   %r12,%r12
    4b64:	48 0f 4e d0          	cmovle %rax,%rdx
    4b68:	48 85 f6             	test   %rsi,%rsi
    4b6b:	4c 0f 4e f2          	cmovle %rdx,%r14
    4b6f:	4d 85 e4             	test   %r12,%r12
    4b72:	4c 0f 49 f2          	cmovns %rdx,%r14
    4b76:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    4b7a:	4c 0f 45 f1          	cmovne %rcx,%r14
    4b7e:	4c 89 f0             	mov    %r14,%rax
    4b81:	48 c1 e0 0b          	shl    $0xb,%rax
    4b85:	48 8d 0c c0          	lea    (%rax,%rax,8),%rcx
    4b89:	4c 89 f6             	mov    %r14,%rsi
    4b8c:	48 c1 e6 05          	shl    $0x5,%rsi
    4b90:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    4b94:	b8 20 00 00 00       	mov    $0x20,%eax
    4b99:	48 0f 44 f0          	cmove  %rax,%rsi
    4b9d:	48 89 f0             	mov    %rsi,%rax
    4ba0:	48 f7 d0             	not    %rax
    4ba3:	48 c1 f8 3f          	sar    $0x3f,%rax
    4ba7:	48 83 c8 01          	or     $0x1,%rax
    4bab:	48 29 c8             	sub    %rcx,%rax
    4bae:	48 99                	cqto   
    4bb0:	48 f7 fe             	idiv   %rsi
    4bb3:	48 89 c5             	mov    %rax,%rbp
    4bb6:	48 f7 d5             	not    %rbp
    4bb9:	48 89 c8             	mov    %rcx,%rax
    4bbc:	48 99                	cqto   
    4bbe:	48 f7 fe             	idiv   %rsi
    4bc1:	48 85 f6             	test   %rsi,%rsi
    4bc4:	48 89 c2             	mov    %rax,%rdx
    4bc7:	48 0f 48 d5          	cmovs  %rbp,%rdx
    4bcb:	48 85 c9             	test   %rcx,%rcx
    4bce:	48 0f 4e d0          	cmovle %rax,%rdx
    4bd2:	48 85 f6             	test   %rsi,%rsi
    4bd5:	48 0f 4e ea          	cmovle %rdx,%rbp
    4bd9:	48 85 c9             	test   %rcx,%rcx
    4bdc:	48 0f 49 ea          	cmovns %rdx,%rbp
    4be0:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    4be4:	49 c7 45 08 20 00 00 	movq   $0x20,0x8(%r13)
    4beb:	00 
    4bec:	49 c7 45 10 18 00 00 	movq   $0x18,0x10(%r13)
    4bf3:	00 
    4bf4:	49 c7 45 18 18 00 00 	movq   $0x18,0x18(%r13)
    4bfb:	00 
    4bfc:	4c 89 f0             	mov    %r14,%rax
    4bff:	48 0f 44 c5          	cmove  %rbp,%rax
    4c03:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    4c08:	4c 89 ff             	mov    %r15,%rdi
    4c0b:	e8 00 d5 ff ff       	call   2110 <free@plt>
    4c10:	4c 89 f0             	mov    %r14,%rax
    4c13:	48 c1 e0 0d          	shl    $0xd,%rax
    4c17:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    4c1c:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    4c21:	e8 aa d4 ff ff       	call   20d0 <malloc@plt>
    4c26:	48 89 c3             	mov    %rax,%rbx
    4c29:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    4c2d:	48 89 04 24          	mov    %rax,(%rsp)
    4c31:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    4c35:	4c 89 7c 24 08       	mov    %r15,0x8(%rsp)
    4c3a:	bf 04 00 00 00       	mov    $0x4,%edi
    4c3f:	e8 5c d5 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    4c44:	49 89 c5             	mov    %rax,%r13
    4c47:	48 89 c7             	mov    %rax,%rdi
    4c4a:	31 f6                	xor    %esi,%esi
    4c4c:	48 89 da             	mov    %rbx,%rdx
    4c4f:	4c 89 f9             	mov    %r15,%rcx
    4c52:	e8 99 d4 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    4c57:	be 01 00 00 00       	mov    $0x1,%esi
    4c5c:	4c 89 ef             	mov    %r13,%rdi
    4c5f:	e8 3c d4 ff ff       	call   20a0 <omTensorSetDataType@plt>
    4c64:	4c 89 ef             	mov    %r13,%rdi
    4c67:	e8 64 d5 ff ff       	call   21d0 <omTensorGetShape@plt>
    4c6c:	48 89 c3             	mov    %rax,%rbx
    4c6f:	4c 89 ef             	mov    %r13,%rdi
    4c72:	e8 f9 d3 ff ff       	call   2070 <omTensorGetStrides@plt>
    4c77:	4c 89 33             	mov    %r14,(%rbx)
    4c7a:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    4c81:	48 c7 43 08 20 00 00 	movq   $0x20,0x8(%rbx)
    4c88:	00 
    4c89:	48 c7 40 08 40 02 00 	movq   $0x240,0x8(%rax)
    4c90:	00 
    4c91:	48 c7 43 10 18 00 00 	movq   $0x18,0x10(%rbx)
    4c98:	00 
    4c99:	48 c7 40 10 18 00 00 	movq   $0x18,0x10(%rax)
    4ca0:	00 
    4ca1:	48 c7 43 18 18 00 00 	movq   $0x18,0x18(%rbx)
    4ca8:	00 
    4ca9:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    4cb0:	00 
    4cb1:	bf 04 00 00 00       	mov    $0x4,%edi
    4cb6:	e8 e5 d4 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    4cbb:	48 89 c3             	mov    %rax,%rbx
    4cbe:	48 89 c7             	mov    %rax,%rdi
    4cc1:	31 f6                	xor    %esi,%esi
    4cc3:	4c 8b 7c 24 50       	mov    0x50(%rsp),%r15
    4cc8:	4c 89 fa             	mov    %r15,%rdx
    4ccb:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    4cd0:	e8 1b d4 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    4cd5:	be 01 00 00 00       	mov    $0x1,%esi
    4cda:	48 89 df             	mov    %rbx,%rdi
    4cdd:	e8 be d3 ff ff       	call   20a0 <omTensorSetDataType@plt>
    4ce2:	48 89 df             	mov    %rbx,%rdi
    4ce5:	e8 e6 d4 ff ff       	call   21d0 <omTensorGetShape@plt>
    4cea:	49 89 c4             	mov    %rax,%r12
    4ced:	48 89 df             	mov    %rbx,%rdi
    4cf0:	e8 7b d3 ff ff       	call   2070 <omTensorGetStrides@plt>
    4cf5:	4d 89 34 24          	mov    %r14,(%r12)
    4cf9:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    4d00:	49 c7 44 24 08 20 00 	movq   $0x20,0x8(%r12)
    4d07:	00 00 
    4d09:	48 c7 40 08 40 02 00 	movq   $0x240,0x8(%rax)
    4d10:	00 
    4d11:	49 c7 44 24 10 18 00 	movq   $0x18,0x10(%r12)
    4d18:	00 00 
    4d1a:	48 c7 40 10 18 00 00 	movq   $0x18,0x10(%rax)
    4d21:	00 
    4d22:	49 c7 44 24 18 18 00 	movq   $0x18,0x18(%r12)
    4d29:	00 00 
    4d2b:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    4d32:	00 
    4d33:	4c 89 ef             	mov    %r13,%rdi
    4d36:	48 89 de             	mov    %rbx,%rsi
    4d39:	e8 b2 d4 ff ff       	call   21f0 <Relu@plt>
    4d3e:	4c 89 ef             	mov    %r13,%rdi
    4d41:	e8 6a d4 ff ff       	call   21b0 <omTensorDestroy@plt>
    4d46:	48 89 df             	mov    %rbx,%rdi
    4d49:	e8 62 d4 ff ff       	call   21b0 <omTensorDestroy@plt>
    4d4e:	4c 89 ff             	mov    %r15,%rdi
    4d51:	e8 ba d3 ff ff       	call   2110 <free@plt>
    4d56:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    4d5b:	e8 70 d3 ff ff       	call   20d0 <malloc@plt>
    4d60:	48 89 c3             	mov    %rax,%rbx
    4d63:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    4d67:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    4d6c:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    4d70:	4c 89 7c 24 20       	mov    %r15,0x20(%rsp)
    4d75:	bf 04 00 00 00       	mov    $0x4,%edi
    4d7a:	e8 21 d4 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    4d7f:	49 89 c4             	mov    %rax,%r12
    4d82:	48 89 c7             	mov    %rax,%rdi
    4d85:	31 f6                	xor    %esi,%esi
    4d87:	48 89 da             	mov    %rbx,%rdx
    4d8a:	4c 89 f9             	mov    %r15,%rcx
    4d8d:	e8 5e d3 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    4d92:	be 01 00 00 00       	mov    $0x1,%esi
    4d97:	4c 89 e3             	mov    %r12,%rbx
    4d9a:	4c 89 64 24 40       	mov    %r12,0x40(%rsp)
    4d9f:	4c 89 e7             	mov    %r12,%rdi
    4da2:	e8 f9 d2 ff ff       	call   20a0 <omTensorSetDataType@plt>
    4da7:	4c 89 e7             	mov    %r12,%rdi
    4daa:	e8 21 d4 ff ff       	call   21d0 <omTensorGetShape@plt>
    4daf:	49 89 c4             	mov    %rax,%r12
    4db2:	48 89 df             	mov    %rbx,%rdi
    4db5:	e8 b6 d2 ff ff       	call   2070 <omTensorGetStrides@plt>
    4dba:	4d 89 34 24          	mov    %r14,(%r12)
    4dbe:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    4dc5:	49 c7 44 24 08 20 00 	movq   $0x20,0x8(%r12)
    4dcc:	00 00 
    4dce:	48 c7 40 08 40 02 00 	movq   $0x240,0x8(%rax)
    4dd5:	00 
    4dd6:	49 c7 44 24 10 18 00 	movq   $0x18,0x10(%r12)
    4ddd:	00 00 
    4ddf:	48 c7 40 10 18 00 00 	movq   $0x18,0x10(%rax)
    4de6:	00 
    4de7:	49 c7 44 24 18 18 00 	movq   $0x18,0x18(%r12)
    4dee:	00 00 
    4df0:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    4df7:	00 
    4df8:	bf 04 00 00 00       	mov    $0x4,%edi
    4dfd:	e8 9e d3 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    4e02:	48 89 c3             	mov    %rax,%rbx
    4e05:	48 89 c7             	mov    %rax,%rdi
    4e08:	31 f6                	xor    %esi,%esi
    4e0a:	48 8b 14 24          	mov    (%rsp),%rdx
    4e0e:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    4e13:	e8 d8 d2 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    4e18:	be 01 00 00 00       	mov    $0x1,%esi
    4e1d:	48 89 df             	mov    %rbx,%rdi
    4e20:	e8 7b d2 ff ff       	call   20a0 <omTensorSetDataType@plt>
    4e25:	48 89 df             	mov    %rbx,%rdi
    4e28:	e8 a3 d3 ff ff       	call   21d0 <omTensorGetShape@plt>
    4e2d:	49 89 c7             	mov    %rax,%r15
    4e30:	48 89 df             	mov    %rbx,%rdi
    4e33:	e8 38 d2 ff ff       	call   2070 <omTensorGetStrides@plt>
    4e38:	4d 89 37             	mov    %r14,(%r15)
    4e3b:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    4e42:	49 c7 47 08 20 00 00 	movq   $0x20,0x8(%r15)
    4e49:	00 
    4e4a:	48 c7 40 08 40 02 00 	movq   $0x240,0x8(%rax)
    4e51:	00 
    4e52:	49 c7 47 10 18 00 00 	movq   $0x18,0x10(%r15)
    4e59:	00 
    4e5a:	48 c7 40 10 18 00 00 	movq   $0x18,0x10(%rax)
    4e61:	00 
    4e62:	49 c7 47 18 18 00 00 	movq   $0x18,0x18(%r15)
    4e69:	00 
    4e6a:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    4e71:	00 
    4e72:	4c 8d 25 57 ae 00 00 	lea    0xae57(%rip),%r12        # fcd0 <constant_64_mobilenet>
    4e79:	bf 04 00 00 00       	mov    $0x4,%edi
    4e7e:	e8 1d d3 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    4e83:	49 89 c7             	mov    %rax,%r15
    4e86:	48 89 c7             	mov    %rax,%rdi
    4e89:	31 f6                	xor    %esi,%esi
    4e8b:	4c 89 e2             	mov    %r12,%rdx
    4e8e:	4c 89 e1             	mov    %r12,%rcx
    4e91:	e8 5a d2 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    4e96:	be 01 00 00 00       	mov    $0x1,%esi
    4e9b:	4c 89 ff             	mov    %r15,%rdi
    4e9e:	e8 fd d1 ff ff       	call   20a0 <omTensorSetDataType@plt>
    4ea3:	4c 89 ff             	mov    %r15,%rdi
    4ea6:	e8 25 d3 ff ff       	call   21d0 <omTensorGetShape@plt>
    4eab:	49 89 c4             	mov    %rax,%r12
    4eae:	4c 89 ff             	mov    %r15,%rdi
    4eb1:	e8 ba d1 ff ff       	call   2070 <omTensorGetStrides@plt>
    4eb6:	49 c7 04 24 20 00 00 	movq   $0x20,(%r12)
    4ebd:	00 
    4ebe:	48 c7 00 09 00 00 00 	movq   $0x9,(%rax)
    4ec5:	49 c7 44 24 08 01 00 	movq   $0x1,0x8(%r12)
    4ecc:	00 00 
    4ece:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    4ed5:	00 
    4ed6:	49 c7 44 24 10 03 00 	movq   $0x3,0x10(%r12)
    4edd:	00 00 
    4edf:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    4ee6:	00 
    4ee7:	49 c7 44 24 18 03 00 	movq   $0x3,0x18(%r12)
    4eee:	00 00 
    4ef0:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    4ef7:	00 
    4ef8:	4c 8d 2d 11 5f 01 00 	lea    0x15f11(%rip),%r13        # 1ae10 <constant_41_mobilenet>
    4eff:	bf 01 00 00 00       	mov    $0x1,%edi
    4f04:	e8 97 d2 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    4f09:	49 89 c4             	mov    %rax,%r12
    4f0c:	48 89 c7             	mov    %rax,%rdi
    4f0f:	31 f6                	xor    %esi,%esi
    4f11:	4c 89 ea             	mov    %r13,%rdx
    4f14:	4c 89 e9             	mov    %r13,%rcx
    4f17:	e8 d4 d1 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    4f1c:	be 01 00 00 00       	mov    $0x1,%esi
    4f21:	4c 89 e7             	mov    %r12,%rdi
    4f24:	e8 77 d1 ff ff       	call   20a0 <omTensorSetDataType@plt>
    4f29:	4c 89 e7             	mov    %r12,%rdi
    4f2c:	e8 9f d2 ff ff       	call   21d0 <omTensorGetShape@plt>
    4f31:	49 89 c5             	mov    %rax,%r13
    4f34:	4c 89 e7             	mov    %r12,%rdi
    4f37:	e8 34 d1 ff ff       	call   2070 <omTensorGetStrides@plt>
    4f3c:	49 c7 45 00 20 00 00 	movq   $0x20,0x0(%r13)
    4f43:	00 
    4f44:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    4f4b:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    4f50:	4c 89 ef             	mov    %r13,%rdi
    4f53:	48 89 de             	mov    %rbx,%rsi
    4f56:	4c 89 fa             	mov    %r15,%rdx
    4f59:	4c 89 e1             	mov    %r12,%rcx
    4f5c:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    4f62:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    4f68:	6a 01                	push   $0x1
    4f6a:	6a 01                	push   $0x1
    4f6c:	6a 01                	push   $0x1
    4f6e:	6a 01                	push   $0x1
    4f70:	e8 eb d1 ff ff       	call   2160 <Conv2D_bias@plt>
    4f75:	48 83 c4 20          	add    $0x20,%rsp
    4f79:	4c 89 ef             	mov    %r13,%rdi
    4f7c:	e8 2f d2 ff ff       	call   21b0 <omTensorDestroy@plt>
    4f81:	48 89 df             	mov    %rbx,%rdi
    4f84:	e8 27 d2 ff ff       	call   21b0 <omTensorDestroy@plt>
    4f89:	4c 89 ff             	mov    %r15,%rdi
    4f8c:	e8 1f d2 ff ff       	call   21b0 <omTensorDestroy@plt>
    4f91:	4c 89 e7             	mov    %r12,%rdi
    4f94:	e8 17 d2 ff ff       	call   21b0 <omTensorDestroy@plt>
    4f99:	48 8b 3c 24          	mov    (%rsp),%rdi
    4f9d:	e8 6e d1 ff ff       	call   2110 <free@plt>
    4fa2:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    4fa7:	e8 24 d1 ff ff       	call   20d0 <malloc@plt>
    4fac:	49 89 c7             	mov    %rax,%r15
    4faf:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    4fb3:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    4fb8:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    4fbc:	4c 89 24 24          	mov    %r12,(%rsp)
    4fc0:	bf 04 00 00 00       	mov    $0x4,%edi
    4fc5:	e8 d6 d1 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    4fca:	48 89 c3             	mov    %rax,%rbx
    4fcd:	48 89 c7             	mov    %rax,%rdi
    4fd0:	31 f6                	xor    %esi,%esi
    4fd2:	4c 89 fa             	mov    %r15,%rdx
    4fd5:	4c 89 e1             	mov    %r12,%rcx
    4fd8:	e8 13 d1 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    4fdd:	be 01 00 00 00       	mov    $0x1,%esi
    4fe2:	48 89 df             	mov    %rbx,%rdi
    4fe5:	e8 b6 d0 ff ff       	call   20a0 <omTensorSetDataType@plt>
    4fea:	48 89 df             	mov    %rbx,%rdi
    4fed:	e8 de d1 ff ff       	call   21d0 <omTensorGetShape@plt>
    4ff2:	49 89 c4             	mov    %rax,%r12
    4ff5:	48 89 df             	mov    %rbx,%rdi
    4ff8:	e8 73 d0 ff ff       	call   2070 <omTensorGetStrides@plt>
    4ffd:	4d 89 34 24          	mov    %r14,(%r12)
    5001:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    5008:	49 c7 44 24 08 20 00 	movq   $0x20,0x8(%r12)
    500f:	00 00 
    5011:	48 c7 40 08 40 02 00 	movq   $0x240,0x8(%rax)
    5018:	00 
    5019:	49 c7 44 24 10 18 00 	movq   $0x18,0x10(%r12)
    5020:	00 00 
    5022:	48 c7 40 10 18 00 00 	movq   $0x18,0x10(%rax)
    5029:	00 
    502a:	49 c7 44 24 18 18 00 	movq   $0x18,0x18(%r12)
    5031:	00 00 
    5033:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    503a:	00 
    503b:	bf 04 00 00 00       	mov    $0x4,%edi
    5040:	e8 5b d1 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    5045:	49 89 c4             	mov    %rax,%r12
    5048:	48 89 c7             	mov    %rax,%rdi
    504b:	31 f6                	xor    %esi,%esi
    504d:	4c 8b 7c 24 30       	mov    0x30(%rsp),%r15
    5052:	4c 89 fa             	mov    %r15,%rdx
    5055:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    505a:	e8 91 d0 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    505f:	be 01 00 00 00       	mov    $0x1,%esi
    5064:	4c 89 e7             	mov    %r12,%rdi
    5067:	e8 34 d0 ff ff       	call   20a0 <omTensorSetDataType@plt>
    506c:	4c 89 e7             	mov    %r12,%rdi
    506f:	e8 5c d1 ff ff       	call   21d0 <omTensorGetShape@plt>
    5074:	49 89 c5             	mov    %rax,%r13
    5077:	4c 89 e7             	mov    %r12,%rdi
    507a:	e8 f1 cf ff ff       	call   2070 <omTensorGetStrides@plt>
    507f:	4d 89 75 00          	mov    %r14,0x0(%r13)
    5083:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    508a:	49 c7 45 08 20 00 00 	movq   $0x20,0x8(%r13)
    5091:	00 
    5092:	48 c7 40 08 40 02 00 	movq   $0x240,0x8(%rax)
    5099:	00 
    509a:	49 c7 45 10 18 00 00 	movq   $0x18,0x10(%r13)
    50a1:	00 
    50a2:	48 c7 40 10 18 00 00 	movq   $0x18,0x10(%rax)
    50a9:	00 
    50aa:	49 c7 45 18 18 00 00 	movq   $0x18,0x18(%r13)
    50b1:	00 
    50b2:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    50b9:	00 
    50ba:	48 89 df             	mov    %rbx,%rdi
    50bd:	4c 89 e6             	mov    %r12,%rsi
    50c0:	e8 2b d1 ff ff       	call   21f0 <Relu@plt>
    50c5:	48 89 df             	mov    %rbx,%rdi
    50c8:	e8 e3 d0 ff ff       	call   21b0 <omTensorDestroy@plt>
    50cd:	4c 89 e7             	mov    %r12,%rdi
    50d0:	e8 db d0 ff ff       	call   21b0 <omTensorDestroy@plt>
    50d5:	4c 89 ff             	mov    %r15,%rdi
    50d8:	e8 33 d0 ff ff       	call   2110 <free@plt>
    50dd:	bf 18 00 00 00       	mov    $0x18,%edi
    50e2:	e8 e9 cf ff ff       	call   20d0 <malloc@plt>
    50e7:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    50ec:	48 83 c0 0f          	add    $0xf,%rax
    50f0:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    50f4:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    50f9:	4c 89 30             	mov    %r14,(%rax)
    50fc:	bf 28 00 00 00       	mov    $0x28,%edi
    5101:	e8 ca cf ff ff       	call   20d0 <malloc@plt>
    5106:	48 8d 48 0f          	lea    0xf(%rax),%rcx
    510a:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
    510e:	4c 89 31             	mov    %r14,(%rcx)
    5111:	48 c7 41 08 20 00 00 	movq   $0x20,0x8(%rcx)
    5118:	00 
    5119:	48 c7 41 10 ff ff ff 	movq   $0xffffffffffffffff,0x10(%rcx)
    5120:	ff 
    5121:	48 89 c7             	mov    %rax,%rdi
    5124:	e8 e7 cf ff ff       	call   2110 <free@plt>
    5129:	48 89 e9             	mov    %rbp,%rcx
    512c:	48 c1 e1 05          	shl    $0x5,%rcx
    5130:	48 8b 5c 24 60       	mov    0x60(%rsp),%rbx
    5135:	48 89 d8             	mov    %rbx,%rax
    5138:	48 0f af c1          	imul   %rcx,%rax
    513c:	49 89 cf             	mov    %rcx,%r15
    513f:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
    5144:	48 8d 3c 85 10 00 00 	lea    0x10(,%rax,4),%rdi
    514b:	00 
    514c:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
    5151:	e8 7a cf ff ff       	call   20d0 <malloc@plt>
    5156:	49 89 c4             	mov    %rax,%r12
    5159:	4c 8d 68 0f          	lea    0xf(%rax),%r13
    515d:	49 83 e5 f0          	and    $0xfffffffffffffff0,%r13
    5161:	bf 03 00 00 00       	mov    $0x3,%edi
    5166:	e8 35 d0 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    516b:	49 89 c6             	mov    %rax,%r14
    516e:	48 89 c7             	mov    %rax,%rdi
    5171:	31 f6                	xor    %esi,%esi
    5173:	4c 89 64 24 70       	mov    %r12,0x70(%rsp)
    5178:	4c 89 e2             	mov    %r12,%rdx
    517b:	4c 89 6c 24 08       	mov    %r13,0x8(%rsp)
    5180:	4c 89 e9             	mov    %r13,%rcx
    5183:	e8 68 cf ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    5188:	be 01 00 00 00       	mov    $0x1,%esi
    518d:	4c 89 f7             	mov    %r14,%rdi
    5190:	e8 0b cf ff ff       	call   20a0 <omTensorSetDataType@plt>
    5195:	4c 89 f7             	mov    %r14,%rdi
    5198:	e8 33 d0 ff ff       	call   21d0 <omTensorGetShape@plt>
    519d:	49 89 c4             	mov    %rax,%r12
    51a0:	4c 89 f7             	mov    %r14,%rdi
    51a3:	e8 c8 ce ff ff       	call   2070 <omTensorGetStrides@plt>
    51a8:	49 89 1c 24          	mov    %rbx,(%r12)
    51ac:	4c 89 38             	mov    %r15,(%rax)
    51af:	49 c7 44 24 08 20 00 	movq   $0x20,0x8(%r12)
    51b6:	00 00 
    51b8:	48 89 68 08          	mov    %rbp,0x8(%rax)
    51bc:	49 89 6c 24 10       	mov    %rbp,0x10(%r12)
    51c1:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    51c8:	00 
    51c9:	bf 02 00 00 00       	mov    $0x2,%edi
    51ce:	e8 cd cf ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    51d3:	49 89 c5             	mov    %rax,%r13
    51d6:	48 8d 0d 33 55 0d 00 	lea    0xd5533(%rip),%rcx        # da710 <constant_13_mobilenet>
    51dd:	48 89 c7             	mov    %rax,%rdi
    51e0:	31 f6                	xor    %esi,%esi
    51e2:	48 89 ca             	mov    %rcx,%rdx
    51e5:	e8 06 cf ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    51ea:	be 01 00 00 00       	mov    $0x1,%esi
    51ef:	4c 89 ef             	mov    %r13,%rdi
    51f2:	e8 a9 ce ff ff       	call   20a0 <omTensorSetDataType@plt>
    51f7:	4c 89 ef             	mov    %r13,%rdi
    51fa:	e8 d1 cf ff ff       	call   21d0 <omTensorGetShape@plt>
    51ff:	49 89 c4             	mov    %rax,%r12
    5202:	4c 89 ef             	mov    %r13,%rdi
    5205:	e8 66 ce ff ff       	call   2070 <omTensorGetStrides@plt>
    520a:	49 c7 04 24 20 00 00 	movq   $0x20,(%r12)
    5211:	00 
    5212:	48 c7 00 20 00 00 00 	movq   $0x20,(%rax)
    5219:	49 c7 44 24 08 20 00 	movq   $0x20,0x8(%r12)
    5220:	00 00 
    5222:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    5229:	00 
    522a:	bf 03 00 00 00       	mov    $0x3,%edi
    522f:	e8 6c cf ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    5234:	49 89 c4             	mov    %rax,%r12
    5237:	48 89 c7             	mov    %rax,%rdi
    523a:	31 f6                	xor    %esi,%esi
    523c:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    5241:	48 8b 0c 24          	mov    (%rsp),%rcx
    5245:	e8 a6 ce ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    524a:	be 01 00 00 00       	mov    $0x1,%esi
    524f:	4c 89 e7             	mov    %r12,%rdi
    5252:	e8 49 ce ff ff       	call   20a0 <omTensorSetDataType@plt>
    5257:	4c 89 e7             	mov    %r12,%rdi
    525a:	e8 71 cf ff ff       	call   21d0 <omTensorGetShape@plt>
    525f:	49 89 c7             	mov    %rax,%r15
    5262:	4c 89 e7             	mov    %r12,%rdi
    5265:	e8 06 ce ff ff       	call   2070 <omTensorGetStrides@plt>
    526a:	49 89 1f             	mov    %rbx,(%r15)
    526d:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    5272:	48 89 08             	mov    %rcx,(%rax)
    5275:	49 c7 47 08 20 00 00 	movq   $0x20,0x8(%r15)
    527c:	00 
    527d:	48 89 68 08          	mov    %rbp,0x8(%rax)
    5281:	49 89 6f 10          	mov    %rbp,0x10(%r15)
    5285:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    528c:	00 
    528d:	4c 89 f7             	mov    %r14,%rdi
    5290:	4c 89 ee             	mov    %r13,%rsi
    5293:	4c 89 e2             	mov    %r12,%rdx
    5296:	e8 b5 cd ff ff       	call   2050 <MatMul@plt>
    529b:	4c 89 f7             	mov    %r14,%rdi
    529e:	e8 0d cf ff ff       	call   21b0 <omTensorDestroy@plt>
    52a3:	4c 89 ef             	mov    %r13,%rdi
    52a6:	e8 05 cf ff ff       	call   21b0 <omTensorDestroy@plt>
    52ab:	4c 89 e7             	mov    %r12,%rdi
    52ae:	e8 fd ce ff ff       	call   21b0 <omTensorDestroy@plt>
    52b3:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    52b8:	e8 53 ce ff ff       	call   2110 <free@plt>
    52bd:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    52c2:	e8 09 ce ff ff       	call   20d0 <malloc@plt>
    52c7:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    52cc:	48 83 c0 0f          	add    $0xf,%rax
    52d0:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    52d4:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    52d9:	48 85 db             	test   %rbx,%rbx
    52dc:	0f 8e 5b 01 00 00    	jle    543d <main_graph_mobilenet+0x316d>
    52e2:	48 89 e8             	mov    %rbp,%rax
    52e5:	48 c1 e0 07          	shl    $0x7,%rax
    52e9:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    52ee:	48 8d 0c ad 00 00 00 	lea    0x0(,%rbp,4),%rcx
    52f5:	00 
    52f6:	48 89 eb             	mov    %rbp,%rbx
    52f9:	48 83 e3 f8          	and    $0xfffffffffffffff8,%rbx
    52fd:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    5302:	49 8d 45 10          	lea    0x10(%r13),%rax
    5306:	48 89 04 24          	mov    %rax,(%rsp)
    530a:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    530f:	48 8d 50 10          	lea    0x10(%rax),%rdx
    5313:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
    5318:	45 31 c0             	xor    %r8d,%r8d
    531b:	4c 8d 0d ee 72 01 00 	lea    0x172ee(%rip),%r9        # 1c610 <constant_26_mobilenet>
    5322:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    5327:	eb 2b                	jmp    5354 <main_graph_mobilenet+0x3084>
    5329:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    5330:	49 ff c0             	inc    %r8
    5333:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    5338:	48 01 04 24          	add    %rax,(%rsp)
    533c:	48 01 44 24 20       	add    %rax,0x20(%rsp)
    5341:	49 01 c5             	add    %rax,%r13
    5344:	48 01 44 24 30       	add    %rax,0x30(%rsp)
    5349:	4c 3b 44 24 60       	cmp    0x60(%rsp),%r8
    534e:	0f 84 e9 00 00 00    	je     543d <main_graph_mobilenet+0x316d>
    5354:	48 85 ed             	test   %rbp,%rbp
    5357:	7e d7                	jle    5330 <main_graph_mobilenet+0x3060>
    5359:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
    535e:	4d 0f af f0          	imul   %r8,%r14
    5362:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    5367:	4e 8d 3c 30          	lea    (%rax,%r14,1),%r15
    536b:	4c 03 74 24 08       	add    0x8(%rsp),%r14
    5370:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
    5375:	4d 89 eb             	mov    %r13,%r11
    5378:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    537d:	48 8b 34 24          	mov    (%rsp),%rsi
    5381:	31 c0                	xor    %eax,%eax
    5383:	eb 20                	jmp    53a5 <main_graph_mobilenet+0x30d5>
    5385:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    538c:	00 00 00 
    538f:	90                   	nop
    5390:	48 ff c0             	inc    %rax
    5393:	48 01 ce             	add    %rcx,%rsi
    5396:	48 01 cf             	add    %rcx,%rdi
    5399:	49 01 cb             	add    %rcx,%r11
    539c:	49 01 cc             	add    %rcx,%r12
    539f:	48 83 f8 20          	cmp    $0x20,%rax
    53a3:	74 8b                	je     5330 <main_graph_mobilenet+0x3060>
    53a5:	48 83 fd 08          	cmp    $0x8,%rbp
    53a9:	f3 41 0f 10 04 81    	movss  (%r9,%rax,4),%xmm0
    53af:	73 0f                	jae    53c0 <main_graph_mobilenet+0x30f0>
    53b1:	31 d2                	xor    %edx,%edx
    53b3:	eb 6b                	jmp    5420 <main_graph_mobilenet+0x3150>
    53b5:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    53bc:	00 00 00 
    53bf:	90                   	nop
    53c0:	48 89 ca             	mov    %rcx,%rdx
    53c3:	48 0f af d0          	imul   %rax,%rdx
    53c7:	4d 8d 14 16          	lea    (%r14,%rdx,1),%r10
    53cb:	4c 01 fa             	add    %r15,%rdx
    53ce:	4c 29 d2             	sub    %r10,%rdx
    53d1:	48 83 fa 20          	cmp    $0x20,%rdx
    53d5:	73 09                	jae    53e0 <main_graph_mobilenet+0x3110>
    53d7:	31 d2                	xor    %edx,%edx
    53d9:	eb 45                	jmp    5420 <main_graph_mobilenet+0x3150>
    53db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    53e0:	0f 28 c8             	movaps %xmm0,%xmm1
    53e3:	0f c6 c8 00          	shufps $0x0,%xmm0,%xmm1
    53e7:	31 d2                	xor    %edx,%edx
    53e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    53f0:	0f 10 54 97 f0       	movups -0x10(%rdi,%rdx,4),%xmm2
    53f5:	0f 10 1c 97          	movups (%rdi,%rdx,4),%xmm3
    53f9:	0f 58 d1             	addps  %xmm1,%xmm2
    53fc:	0f 58 d9             	addps  %xmm1,%xmm3
    53ff:	0f 11 54 96 f0       	movups %xmm2,-0x10(%rsi,%rdx,4)
    5404:	0f 11 1c 96          	movups %xmm3,(%rsi,%rdx,4)
    5408:	48 83 c2 08          	add    $0x8,%rdx
    540c:	48 39 d3             	cmp    %rdx,%rbx
    540f:	75 df                	jne    53f0 <main_graph_mobilenet+0x3120>
    5411:	48 39 dd             	cmp    %rbx,%rbp
    5414:	48 89 da             	mov    %rbx,%rdx
    5417:	0f 84 73 ff ff ff    	je     5390 <main_graph_mobilenet+0x30c0>
    541d:	0f 1f 00             	nopl   (%rax)
    5420:	f3 41 0f 10 0c 94    	movss  (%r12,%rdx,4),%xmm1
    5426:	f3 0f 58 c8          	addss  %xmm0,%xmm1
    542a:	f3 41 0f 11 0c 93    	movss  %xmm1,(%r11,%rdx,4)
    5430:	48 ff c2             	inc    %rdx
    5433:	48 39 d5             	cmp    %rdx,%rbp
    5436:	75 e8                	jne    5420 <main_graph_mobilenet+0x3150>
    5438:	e9 53 ff ff ff       	jmp    5390 <main_graph_mobilenet+0x30c0>
    543d:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    5442:	e8 c9 cc ff ff       	call   2110 <free@plt>
    5447:	bf 30 00 00 00       	mov    $0x30,%edi
    544c:	e8 7f cc ff ff       	call   20d0 <malloc@plt>
    5451:	49 89 c4             	mov    %rax,%r12
    5454:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    5458:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    545c:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    5461:	48 8b 00             	mov    (%rax),%rax
    5464:	49 89 07             	mov    %rax,(%r15)
    5467:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    546c:	e8 9f cc ff ff       	call   2110 <free@plt>
    5471:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    5476:	48 0f af e8          	imul   %rax,%rbp
    547a:	48 c1 e5 05          	shl    $0x5,%rbp
    547e:	49 8b 0f             	mov    (%r15),%rcx
    5481:	48 85 c9             	test   %rcx,%rcx
    5484:	48 0f 44 c8          	cmove  %rax,%rcx
    5488:	48 89 c8             	mov    %rcx,%rax
    548b:	48 c1 e0 0b          	shl    $0xb,%rax
    548f:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    5493:	48 8d 34 c0          	lea    (%rax,%rax,8),%rsi
    5497:	b8 00 48 00 00       	mov    $0x4800,%eax
    549c:	48 0f 44 f0          	cmove  %rax,%rsi
    54a0:	48 89 f0             	mov    %rsi,%rax
    54a3:	48 f7 d0             	not    %rax
    54a6:	48 c1 f8 3f          	sar    $0x3f,%rax
    54aa:	48 83 c8 01          	or     $0x1,%rax
    54ae:	48 29 e8             	sub    %rbp,%rax
    54b1:	48 99                	cqto   
    54b3:	48 f7 fe             	idiv   %rsi
    54b6:	49 89 c6             	mov    %rax,%r14
    54b9:	49 f7 d6             	not    %r14
    54bc:	48 89 e8             	mov    %rbp,%rax
    54bf:	48 99                	cqto   
    54c1:	48 f7 fe             	idiv   %rsi
    54c4:	48 85 f6             	test   %rsi,%rsi
    54c7:	48 89 c2             	mov    %rax,%rdx
    54ca:	49 0f 48 d6          	cmovs  %r14,%rdx
    54ce:	48 85 ed             	test   %rbp,%rbp
    54d1:	48 0f 4e d0          	cmovle %rax,%rdx
    54d5:	48 85 f6             	test   %rsi,%rsi
    54d8:	4c 0f 4e f2          	cmovle %rdx,%r14
    54dc:	48 85 ed             	test   %rbp,%rbp
    54df:	4c 0f 49 f2          	cmovns %rdx,%r14
    54e3:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    54e7:	4c 0f 45 f1          	cmovne %rcx,%r14
    54eb:	4c 89 f0             	mov    %r14,%rax
    54ee:	48 c1 e0 09          	shl    $0x9,%rax
    54f2:	48 8d 0c c0          	lea    (%rax,%rax,8),%rcx
    54f6:	4c 89 f6             	mov    %r14,%rsi
    54f9:	48 c1 e6 05          	shl    $0x5,%rsi
    54fd:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    5501:	b8 20 00 00 00       	mov    $0x20,%eax
    5506:	48 0f 44 f0          	cmove  %rax,%rsi
    550a:	48 89 f0             	mov    %rsi,%rax
    550d:	48 f7 d0             	not    %rax
    5510:	48 c1 f8 3f          	sar    $0x3f,%rax
    5514:	48 83 c8 01          	or     $0x1,%rax
    5518:	48 29 c8             	sub    %rcx,%rax
    551b:	48 99                	cqto   
    551d:	48 f7 fe             	idiv   %rsi
    5520:	49 89 c5             	mov    %rax,%r13
    5523:	49 f7 d5             	not    %r13
    5526:	48 89 c8             	mov    %rcx,%rax
    5529:	48 99                	cqto   
    552b:	48 f7 fe             	idiv   %rsi
    552e:	48 85 f6             	test   %rsi,%rsi
    5531:	48 89 c2             	mov    %rax,%rdx
    5534:	49 0f 48 d5          	cmovs  %r13,%rdx
    5538:	48 85 c9             	test   %rcx,%rcx
    553b:	48 0f 4e d0          	cmovle %rax,%rdx
    553f:	48 85 f6             	test   %rsi,%rsi
    5542:	4c 0f 4e ea          	cmovle %rdx,%r13
    5546:	48 85 c9             	test   %rcx,%rcx
    5549:	4c 0f 49 ea          	cmovns %rdx,%r13
    554d:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    5551:	49 c7 47 08 20 00 00 	movq   $0x20,0x8(%r15)
    5558:	00 
    5559:	49 c7 47 10 18 00 00 	movq   $0x18,0x10(%r15)
    5560:	00 
    5561:	49 c7 47 18 18 00 00 	movq   $0x18,0x18(%r15)
    5568:	00 
    5569:	4c 89 f0             	mov    %r14,%rax
    556c:	49 0f 44 c5          	cmove  %r13,%rax
    5570:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    5575:	4c 89 e7             	mov    %r12,%rdi
    5578:	e8 93 cb ff ff       	call   2110 <free@plt>
    557d:	4c 89 f0             	mov    %r14,%rax
    5580:	48 c1 e0 0d          	shl    $0xd,%rax
    5584:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    5589:	e8 42 cb ff ff       	call   20d0 <malloc@plt>
    558e:	48 89 c3             	mov    %rax,%rbx
    5591:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    5595:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    559a:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    559e:	4c 89 7c 24 08       	mov    %r15,0x8(%rsp)
    55a3:	bf 04 00 00 00       	mov    $0x4,%edi
    55a8:	e8 f3 cb ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    55ad:	48 89 c5             	mov    %rax,%rbp
    55b0:	48 89 c7             	mov    %rax,%rdi
    55b3:	31 f6                	xor    %esi,%esi
    55b5:	48 89 da             	mov    %rbx,%rdx
    55b8:	4c 89 f9             	mov    %r15,%rcx
    55bb:	e8 30 cb ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    55c0:	be 01 00 00 00       	mov    $0x1,%esi
    55c5:	48 89 ef             	mov    %rbp,%rdi
    55c8:	e8 d3 ca ff ff       	call   20a0 <omTensorSetDataType@plt>
    55cd:	48 89 ef             	mov    %rbp,%rdi
    55d0:	e8 fb cb ff ff       	call   21d0 <omTensorGetShape@plt>
    55d5:	49 89 c7             	mov    %rax,%r15
    55d8:	48 89 ef             	mov    %rbp,%rdi
    55db:	e8 90 ca ff ff       	call   2070 <omTensorGetStrides@plt>
    55e0:	4d 89 37             	mov    %r14,(%r15)
    55e3:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    55ea:	49 c7 47 08 20 00 00 	movq   $0x20,0x8(%r15)
    55f1:	00 
    55f2:	48 c7 40 08 40 02 00 	movq   $0x240,0x8(%rax)
    55f9:	00 
    55fa:	49 c7 47 10 18 00 00 	movq   $0x18,0x10(%r15)
    5601:	00 
    5602:	48 c7 40 10 18 00 00 	movq   $0x18,0x10(%rax)
    5609:	00 
    560a:	49 c7 47 18 18 00 00 	movq   $0x18,0x18(%r15)
    5611:	00 
    5612:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    5619:	00 
    561a:	bf 04 00 00 00       	mov    $0x4,%edi
    561f:	e8 7c cb ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    5624:	49 89 c7             	mov    %rax,%r15
    5627:	48 89 c7             	mov    %rax,%rdi
    562a:	31 f6                	xor    %esi,%esi
    562c:	4c 8b 64 24 50       	mov    0x50(%rsp),%r12
    5631:	4c 89 e2             	mov    %r12,%rdx
    5634:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    5639:	e8 b2 ca ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    563e:	be 01 00 00 00       	mov    $0x1,%esi
    5643:	4c 89 ff             	mov    %r15,%rdi
    5646:	e8 55 ca ff ff       	call   20a0 <omTensorSetDataType@plt>
    564b:	4c 89 ff             	mov    %r15,%rdi
    564e:	e8 7d cb ff ff       	call   21d0 <omTensorGetShape@plt>
    5653:	48 89 c3             	mov    %rax,%rbx
    5656:	4c 89 ff             	mov    %r15,%rdi
    5659:	e8 12 ca ff ff       	call   2070 <omTensorGetStrides@plt>
    565e:	4c 89 33             	mov    %r14,(%rbx)
    5661:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    5668:	48 c7 43 08 20 00 00 	movq   $0x20,0x8(%rbx)
    566f:	00 
    5670:	48 c7 40 08 40 02 00 	movq   $0x240,0x8(%rax)
    5677:	00 
    5678:	48 c7 43 10 18 00 00 	movq   $0x18,0x10(%rbx)
    567f:	00 
    5680:	48 c7 40 10 18 00 00 	movq   $0x18,0x10(%rax)
    5687:	00 
    5688:	48 c7 43 18 18 00 00 	movq   $0x18,0x18(%rbx)
    568f:	00 
    5690:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    5697:	00 
    5698:	48 89 ef             	mov    %rbp,%rdi
    569b:	4c 89 fe             	mov    %r15,%rsi
    569e:	e8 4d cb ff ff       	call   21f0 <Relu@plt>
    56a3:	48 89 ef             	mov    %rbp,%rdi
    56a6:	e8 05 cb ff ff       	call   21b0 <omTensorDestroy@plt>
    56ab:	4c 89 ff             	mov    %r15,%rdi
    56ae:	e8 fd ca ff ff       	call   21b0 <omTensorDestroy@plt>
    56b3:	4c 89 e7             	mov    %r12,%rdi
    56b6:	e8 55 ca ff ff       	call   2110 <free@plt>
    56bb:	4c 89 f0             	mov    %r14,%rax
    56be:	48 c1 e0 0b          	shl    $0xb,%rax
    56c2:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    56c7:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
    56cc:	e8 ff c9 ff ff       	call   20d0 <malloc@plt>
    56d1:	48 89 c3             	mov    %rax,%rbx
    56d4:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    56d8:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    56dd:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    56e1:	4c 89 3c 24          	mov    %r15,(%rsp)
    56e5:	bf 04 00 00 00       	mov    $0x4,%edi
    56ea:	e8 b1 ca ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    56ef:	49 89 c4             	mov    %rax,%r12
    56f2:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    56f7:	48 89 c7             	mov    %rax,%rdi
    56fa:	31 f6                	xor    %esi,%esi
    56fc:	48 89 da             	mov    %rbx,%rdx
    56ff:	4c 89 f9             	mov    %r15,%rcx
    5702:	e8 e9 c9 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    5707:	be 01 00 00 00       	mov    $0x1,%esi
    570c:	4c 89 e7             	mov    %r12,%rdi
    570f:	e8 8c c9 ff ff       	call   20a0 <omTensorSetDataType@plt>
    5714:	4c 89 e7             	mov    %r12,%rdi
    5717:	e8 b4 ca ff ff       	call   21d0 <omTensorGetShape@plt>
    571c:	49 89 c7             	mov    %rax,%r15
    571f:	4c 89 e7             	mov    %r12,%rdi
    5722:	e8 49 c9 ff ff       	call   2070 <omTensorGetStrides@plt>
    5727:	4d 89 37             	mov    %r14,(%r15)
    572a:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    5731:	49 c7 47 08 20 00 00 	movq   $0x20,0x8(%r15)
    5738:	00 
    5739:	48 c7 40 08 90 00 00 	movq   $0x90,0x8(%rax)
    5740:	00 
    5741:	49 c7 47 10 0c 00 00 	movq   $0xc,0x10(%r15)
    5748:	00 
    5749:	48 c7 40 10 0c 00 00 	movq   $0xc,0x10(%rax)
    5750:	00 
    5751:	49 c7 47 18 0c 00 00 	movq   $0xc,0x18(%r15)
    5758:	00 
    5759:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    5760:	00 
    5761:	bf 04 00 00 00       	mov    $0x4,%edi
    5766:	e8 35 ca ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    576b:	49 89 c7             	mov    %rax,%r15
    576e:	48 89 c7             	mov    %rax,%rdi
    5771:	31 f6                	xor    %esi,%esi
    5773:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    5778:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    577d:	e8 6e c9 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    5782:	be 01 00 00 00       	mov    $0x1,%esi
    5787:	4c 89 ff             	mov    %r15,%rdi
    578a:	e8 11 c9 ff ff       	call   20a0 <omTensorSetDataType@plt>
    578f:	4c 89 ff             	mov    %r15,%rdi
    5792:	e8 39 ca ff ff       	call   21d0 <omTensorGetShape@plt>
    5797:	49 89 c4             	mov    %rax,%r12
    579a:	4c 89 ff             	mov    %r15,%rdi
    579d:	e8 ce c8 ff ff       	call   2070 <omTensorGetStrides@plt>
    57a2:	4d 89 34 24          	mov    %r14,(%r12)
    57a6:	48 c7 00 00 48 00 00 	movq   $0x4800,(%rax)
    57ad:	49 c7 44 24 08 20 00 	movq   $0x20,0x8(%r12)
    57b4:	00 00 
    57b6:	48 c7 40 08 40 02 00 	movq   $0x240,0x8(%rax)
    57bd:	00 
    57be:	49 c7 44 24 10 18 00 	movq   $0x18,0x10(%r12)
    57c5:	00 00 
    57c7:	48 c7 40 10 18 00 00 	movq   $0x18,0x10(%rax)
    57ce:	00 
    57cf:	49 c7 44 24 18 18 00 	movq   $0x18,0x18(%r12)
    57d6:	00 00 
    57d8:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    57df:	00 
    57e0:	48 8d 2d c9 3c 01 00 	lea    0x13cc9(%rip),%rbp        # 194b0 <constant_55_mobilenet>
    57e7:	bf 04 00 00 00       	mov    $0x4,%edi
    57ec:	e8 af c9 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    57f1:	49 89 c4             	mov    %rax,%r12
    57f4:	48 89 c7             	mov    %rax,%rdi
    57f7:	31 f6                	xor    %esi,%esi
    57f9:	48 89 ea             	mov    %rbp,%rdx
    57fc:	48 89 e9             	mov    %rbp,%rcx
    57ff:	e8 ec c8 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    5804:	be 01 00 00 00       	mov    $0x1,%esi
    5809:	4c 89 e7             	mov    %r12,%rdi
    580c:	e8 8f c8 ff ff       	call   20a0 <omTensorSetDataType@plt>
    5811:	4c 89 e7             	mov    %r12,%rdi
    5814:	e8 b7 c9 ff ff       	call   21d0 <omTensorGetShape@plt>
    5819:	48 89 c5             	mov    %rax,%rbp
    581c:	4c 89 e7             	mov    %r12,%rdi
    581f:	e8 4c c8 ff ff       	call   2070 <omTensorGetStrides@plt>
    5824:	48 c7 45 00 20 00 00 	movq   $0x20,0x0(%rbp)
    582b:	00 
    582c:	48 c7 00 09 00 00 00 	movq   $0x9,(%rax)
    5833:	48 c7 45 08 01 00 00 	movq   $0x1,0x8(%rbp)
    583a:	00 
    583b:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    5842:	00 
    5843:	48 c7 45 10 03 00 00 	movq   $0x3,0x10(%rbp)
    584a:	00 
    584b:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    5852:	00 
    5853:	48 c7 45 18 03 00 00 	movq   $0x3,0x18(%rbp)
    585a:	00 
    585b:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    5862:	00 
    5863:	48 8d 1d 26 56 01 00 	lea    0x15626(%rip),%rbx        # 1ae90 <constant_40_mobilenet>
    586a:	bf 01 00 00 00       	mov    $0x1,%edi
    586f:	e8 2c c9 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    5874:	48 89 c5             	mov    %rax,%rbp
    5877:	48 89 c7             	mov    %rax,%rdi
    587a:	31 f6                	xor    %esi,%esi
    587c:	48 89 da             	mov    %rbx,%rdx
    587f:	48 89 d9             	mov    %rbx,%rcx
    5882:	e8 69 c8 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    5887:	be 01 00 00 00       	mov    $0x1,%esi
    588c:	48 89 ef             	mov    %rbp,%rdi
    588f:	e8 0c c8 ff ff       	call   20a0 <omTensorSetDataType@plt>
    5894:	48 89 ef             	mov    %rbp,%rdi
    5897:	e8 34 c9 ff ff       	call   21d0 <omTensorGetShape@plt>
    589c:	48 89 c3             	mov    %rax,%rbx
    589f:	48 89 ef             	mov    %rbp,%rdi
    58a2:	e8 c9 c7 ff ff       	call   2070 <omTensorGetStrides@plt>
    58a7:	48 c7 03 20 00 00 00 	movq   $0x20,(%rbx)
    58ae:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    58b5:	48 8b 5c 24 40       	mov    0x40(%rsp),%rbx
    58ba:	48 89 df             	mov    %rbx,%rdi
    58bd:	4c 89 fe             	mov    %r15,%rsi
    58c0:	4c 89 e2             	mov    %r12,%rdx
    58c3:	48 89 e9             	mov    %rbp,%rcx
    58c6:	45 31 c0             	xor    %r8d,%r8d
    58c9:	45 31 c9             	xor    %r9d,%r9d
    58cc:	6a 02                	push   $0x2
    58ce:	6a 02                	push   $0x2
    58d0:	6a 01                	push   $0x1
    58d2:	6a 01                	push   $0x1
    58d4:	e8 87 c8 ff ff       	call   2160 <Conv2D_bias@plt>
    58d9:	48 83 c4 20          	add    $0x20,%rsp
    58dd:	48 89 df             	mov    %rbx,%rdi
    58e0:	e8 cb c8 ff ff       	call   21b0 <omTensorDestroy@plt>
    58e5:	4c 89 ff             	mov    %r15,%rdi
    58e8:	e8 c3 c8 ff ff       	call   21b0 <omTensorDestroy@plt>
    58ed:	4c 89 e7             	mov    %r12,%rdi
    58f0:	e8 bb c8 ff ff       	call   21b0 <omTensorDestroy@plt>
    58f5:	48 89 ef             	mov    %rbp,%rdi
    58f8:	e8 b3 c8 ff ff       	call   21b0 <omTensorDestroy@plt>
    58fd:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    5902:	e8 09 c8 ff ff       	call   2110 <free@plt>
    5907:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    590c:	e8 bf c7 ff ff       	call   20d0 <malloc@plt>
    5911:	48 89 c3             	mov    %rax,%rbx
    5914:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    5918:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    591d:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    5921:	4c 89 64 24 30       	mov    %r12,0x30(%rsp)
    5926:	bf 04 00 00 00       	mov    $0x4,%edi
    592b:	e8 70 c8 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    5930:	49 89 c7             	mov    %rax,%r15
    5933:	48 89 c7             	mov    %rax,%rdi
    5936:	31 f6                	xor    %esi,%esi
    5938:	48 89 da             	mov    %rbx,%rdx
    593b:	4c 89 e1             	mov    %r12,%rcx
    593e:	e8 ad c7 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    5943:	be 01 00 00 00       	mov    $0x1,%esi
    5948:	4c 89 ff             	mov    %r15,%rdi
    594b:	e8 50 c7 ff ff       	call   20a0 <omTensorSetDataType@plt>
    5950:	4c 89 ff             	mov    %r15,%rdi
    5953:	e8 78 c8 ff ff       	call   21d0 <omTensorGetShape@plt>
    5958:	48 89 c3             	mov    %rax,%rbx
    595b:	4c 89 ff             	mov    %r15,%rdi
    595e:	e8 0d c7 ff ff       	call   2070 <omTensorGetStrides@plt>
    5963:	4c 89 33             	mov    %r14,(%rbx)
    5966:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    596d:	48 c7 43 08 20 00 00 	movq   $0x20,0x8(%rbx)
    5974:	00 
    5975:	48 c7 40 08 90 00 00 	movq   $0x90,0x8(%rax)
    597c:	00 
    597d:	48 c7 43 10 0c 00 00 	movq   $0xc,0x10(%rbx)
    5984:	00 
    5985:	48 c7 40 10 0c 00 00 	movq   $0xc,0x10(%rax)
    598c:	00 
    598d:	48 c7 43 18 0c 00 00 	movq   $0xc,0x18(%rbx)
    5994:	00 
    5995:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    599c:	00 
    599d:	bf 04 00 00 00       	mov    $0x4,%edi
    59a2:	e8 f9 c7 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    59a7:	48 89 c3             	mov    %rax,%rbx
    59aa:	48 89 c7             	mov    %rax,%rdi
    59ad:	31 f6                	xor    %esi,%esi
    59af:	4c 8b 64 24 20       	mov    0x20(%rsp),%r12
    59b4:	4c 89 e2             	mov    %r12,%rdx
    59b7:	48 8b 0c 24          	mov    (%rsp),%rcx
    59bb:	e8 30 c7 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    59c0:	be 01 00 00 00       	mov    $0x1,%esi
    59c5:	48 89 df             	mov    %rbx,%rdi
    59c8:	e8 d3 c6 ff ff       	call   20a0 <omTensorSetDataType@plt>
    59cd:	48 89 df             	mov    %rbx,%rdi
    59d0:	e8 fb c7 ff ff       	call   21d0 <omTensorGetShape@plt>
    59d5:	48 89 c5             	mov    %rax,%rbp
    59d8:	48 89 df             	mov    %rbx,%rdi
    59db:	e8 90 c6 ff ff       	call   2070 <omTensorGetStrides@plt>
    59e0:	4c 89 75 00          	mov    %r14,0x0(%rbp)
    59e4:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    59eb:	48 c7 45 08 20 00 00 	movq   $0x20,0x8(%rbp)
    59f2:	00 
    59f3:	48 c7 40 08 90 00 00 	movq   $0x90,0x8(%rax)
    59fa:	00 
    59fb:	48 c7 45 10 0c 00 00 	movq   $0xc,0x10(%rbp)
    5a02:	00 
    5a03:	48 c7 40 10 0c 00 00 	movq   $0xc,0x10(%rax)
    5a0a:	00 
    5a0b:	48 c7 45 18 0c 00 00 	movq   $0xc,0x18(%rbp)
    5a12:	00 
    5a13:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    5a1a:	00 
    5a1b:	4c 89 ff             	mov    %r15,%rdi
    5a1e:	48 89 de             	mov    %rbx,%rsi
    5a21:	e8 ca c7 ff ff       	call   21f0 <Relu@plt>
    5a26:	4c 89 ff             	mov    %r15,%rdi
    5a29:	e8 82 c7 ff ff       	call   21b0 <omTensorDestroy@plt>
    5a2e:	48 89 df             	mov    %rbx,%rdi
    5a31:	e8 7a c7 ff ff       	call   21b0 <omTensorDestroy@plt>
    5a36:	4c 89 e7             	mov    %r12,%rdi
    5a39:	e8 d2 c6 ff ff       	call   2110 <free@plt>
    5a3e:	bf 18 00 00 00       	mov    $0x18,%edi
    5a43:	e8 88 c6 ff ff       	call   20d0 <malloc@plt>
    5a48:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    5a4d:	48 83 c0 0f          	add    $0xf,%rax
    5a51:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    5a55:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    5a5a:	4c 89 30             	mov    %r14,(%rax)
    5a5d:	bf 28 00 00 00       	mov    $0x28,%edi
    5a62:	e8 69 c6 ff ff       	call   20d0 <malloc@plt>
    5a67:	48 8d 48 0f          	lea    0xf(%rax),%rcx
    5a6b:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
    5a6f:	4c 89 31             	mov    %r14,(%rcx)
    5a72:	48 c7 41 08 20 00 00 	movq   $0x20,0x8(%rcx)
    5a79:	00 
    5a7a:	48 c7 41 10 ff ff ff 	movq   $0xffffffffffffffff,0x10(%rcx)
    5a81:	ff 
    5a82:	48 89 c7             	mov    %rax,%rdi
    5a85:	e8 86 c6 ff ff       	call   2110 <free@plt>
    5a8a:	4c 89 e8             	mov    %r13,%rax
    5a8d:	48 c1 e0 05          	shl    $0x5,%rax
    5a91:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    5a96:	4c 89 eb             	mov    %r13,%rbx
    5a99:	48 c1 e3 06          	shl    $0x6,%rbx
    5a9d:	48 8b 6c 24 60       	mov    0x60(%rsp),%rbp
    5aa2:	48 89 e8             	mov    %rbp,%rax
    5aa5:	48 0f af c3          	imul   %rbx,%rax
    5aa9:	48 8d 3c 85 10 00 00 	lea    0x10(,%rax,4),%rdi
    5ab0:	00 
    5ab1:	48 89 3c 24          	mov    %rdi,(%rsp)
    5ab5:	e8 16 c6 ff ff       	call   20d0 <malloc@plt>
    5aba:	49 89 c7             	mov    %rax,%r15
    5abd:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    5ac1:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    5ac5:	bf 03 00 00 00       	mov    $0x3,%edi
    5aca:	e8 d1 c6 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    5acf:	49 89 c6             	mov    %rax,%r14
    5ad2:	48 89 c7             	mov    %rax,%rdi
    5ad5:	31 f6                	xor    %esi,%esi
    5ad7:	4c 89 7c 24 70       	mov    %r15,0x70(%rsp)
    5adc:	4c 89 fa             	mov    %r15,%rdx
    5adf:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    5ae4:	4c 89 e1             	mov    %r12,%rcx
    5ae7:	e8 04 c6 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    5aec:	be 01 00 00 00       	mov    $0x1,%esi
    5af1:	4c 89 f7             	mov    %r14,%rdi
    5af4:	e8 a7 c5 ff ff       	call   20a0 <omTensorSetDataType@plt>
    5af9:	4c 89 f7             	mov    %r14,%rdi
    5afc:	e8 cf c6 ff ff       	call   21d0 <omTensorGetShape@plt>
    5b01:	49 89 c7             	mov    %rax,%r15
    5b04:	4c 89 f7             	mov    %r14,%rdi
    5b07:	e8 64 c5 ff ff       	call   2070 <omTensorGetStrides@plt>
    5b0c:	49 89 2f             	mov    %rbp,(%r15)
    5b0f:	48 89 18             	mov    %rbx,(%rax)
    5b12:	49 c7 47 08 40 00 00 	movq   $0x40,0x8(%r15)
    5b19:	00 
    5b1a:	4c 89 68 08          	mov    %r13,0x8(%rax)
    5b1e:	4d 89 6f 10          	mov    %r13,0x10(%r15)
    5b22:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    5b29:	00 
    5b2a:	bf 02 00 00 00       	mov    $0x2,%edi
    5b2f:	e8 6c c6 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    5b34:	48 89 c3             	mov    %rax,%rbx
    5b37:	48 8d 0d d2 2b 0d 00 	lea    0xd2bd2(%rip),%rcx        # d8710 <constant_14_mobilenet>
    5b3e:	48 89 c7             	mov    %rax,%rdi
    5b41:	31 f6                	xor    %esi,%esi
    5b43:	48 89 ca             	mov    %rcx,%rdx
    5b46:	e8 a5 c5 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    5b4b:	be 01 00 00 00       	mov    $0x1,%esi
    5b50:	48 89 df             	mov    %rbx,%rdi
    5b53:	e8 48 c5 ff ff       	call   20a0 <omTensorSetDataType@plt>
    5b58:	48 89 df             	mov    %rbx,%rdi
    5b5b:	e8 70 c6 ff ff       	call   21d0 <omTensorGetShape@plt>
    5b60:	49 89 c7             	mov    %rax,%r15
    5b63:	48 89 df             	mov    %rbx,%rdi
    5b66:	e8 05 c5 ff ff       	call   2070 <omTensorGetStrides@plt>
    5b6b:	49 c7 07 40 00 00 00 	movq   $0x40,(%r15)
    5b72:	48 c7 00 20 00 00 00 	movq   $0x20,(%rax)
    5b79:	49 c7 47 08 20 00 00 	movq   $0x20,0x8(%r15)
    5b80:	00 
    5b81:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    5b88:	00 
    5b89:	bf 03 00 00 00       	mov    $0x3,%edi
    5b8e:	e8 0d c6 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    5b93:	49 89 c7             	mov    %rax,%r15
    5b96:	48 89 c7             	mov    %rax,%rdi
    5b99:	31 f6                	xor    %esi,%esi
    5b9b:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    5ba0:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    5ba5:	e8 46 c5 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    5baa:	be 01 00 00 00       	mov    $0x1,%esi
    5baf:	4c 89 ff             	mov    %r15,%rdi
    5bb2:	e8 e9 c4 ff ff       	call   20a0 <omTensorSetDataType@plt>
    5bb7:	4c 89 ff             	mov    %r15,%rdi
    5bba:	e8 11 c6 ff ff       	call   21d0 <omTensorGetShape@plt>
    5bbf:	49 89 c4             	mov    %rax,%r12
    5bc2:	4c 89 ff             	mov    %r15,%rdi
    5bc5:	e8 a6 c4 ff ff       	call   2070 <omTensorGetStrides@plt>
    5bca:	49 89 2c 24          	mov    %rbp,(%r12)
    5bce:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    5bd3:	48 89 08             	mov    %rcx,(%rax)
    5bd6:	49 c7 44 24 08 20 00 	movq   $0x20,0x8(%r12)
    5bdd:	00 00 
    5bdf:	4c 89 68 08          	mov    %r13,0x8(%rax)
    5be3:	4d 89 6c 24 10       	mov    %r13,0x10(%r12)
    5be8:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    5bef:	00 
    5bf0:	4c 89 f7             	mov    %r14,%rdi
    5bf3:	48 89 de             	mov    %rbx,%rsi
    5bf6:	4c 89 fa             	mov    %r15,%rdx
    5bf9:	e8 52 c4 ff ff       	call   2050 <MatMul@plt>
    5bfe:	4c 89 f7             	mov    %r14,%rdi
    5c01:	e8 aa c5 ff ff       	call   21b0 <omTensorDestroy@plt>
    5c06:	48 89 df             	mov    %rbx,%rdi
    5c09:	e8 a2 c5 ff ff       	call   21b0 <omTensorDestroy@plt>
    5c0e:	4c 89 ff             	mov    %r15,%rdi
    5c11:	e8 9a c5 ff ff       	call   21b0 <omTensorDestroy@plt>
    5c16:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    5c1b:	e8 f0 c4 ff ff       	call   2110 <free@plt>
    5c20:	48 8b 3c 24          	mov    (%rsp),%rdi
    5c24:	e8 a7 c4 ff ff       	call   20d0 <malloc@plt>
    5c29:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    5c2e:	48 83 c0 0f          	add    $0xf,%rax
    5c32:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    5c36:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    5c3b:	48 85 ed             	test   %rbp,%rbp
    5c3e:	0f 8e 59 01 00 00    	jle    5d9d <main_graph_mobilenet+0x3acd>
    5c44:	4c 89 e8             	mov    %r13,%rax
    5c47:	48 c1 e0 08          	shl    $0x8,%rax
    5c4b:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    5c50:	4a 8d 0c ad 00 00 00 	lea    0x0(,%r13,4),%rcx
    5c57:	00 
    5c58:	4c 89 eb             	mov    %r13,%rbx
    5c5b:	48 83 e3 f8          	and    $0xfffffffffffffff8,%rbx
    5c5f:	48 8b 6c 24 40       	mov    0x40(%rsp),%rbp
    5c64:	48 8d 45 10          	lea    0x10(%rbp),%rax
    5c68:	48 89 04 24          	mov    %rax,(%rsp)
    5c6c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    5c71:	48 8d 50 10          	lea    0x10(%rax),%rdx
    5c75:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
    5c7a:	45 31 c0             	xor    %r8d,%r8d
    5c7d:	4c 8d 0d 8c 68 01 00 	lea    0x1688c(%rip),%r9        # 1c510 <constant_27_mobilenet>
    5c84:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    5c89:	eb 29                	jmp    5cb4 <main_graph_mobilenet+0x39e4>
    5c8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5c90:	49 ff c0             	inc    %r8
    5c93:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    5c98:	48 01 04 24          	add    %rax,(%rsp)
    5c9c:	48 01 44 24 20       	add    %rax,0x20(%rsp)
    5ca1:	48 01 c5             	add    %rax,%rbp
    5ca4:	48 01 44 24 30       	add    %rax,0x30(%rsp)
    5ca9:	4c 3b 44 24 60       	cmp    0x60(%rsp),%r8
    5cae:	0f 84 e9 00 00 00    	je     5d9d <main_graph_mobilenet+0x3acd>
    5cb4:	4d 85 ed             	test   %r13,%r13
    5cb7:	7e d7                	jle    5c90 <main_graph_mobilenet+0x39c0>
    5cb9:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
    5cbe:	4d 0f af f0          	imul   %r8,%r14
    5cc2:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    5cc7:	4e 8d 3c 30          	lea    (%rax,%r14,1),%r15
    5ccb:	4c 03 74 24 08       	add    0x8(%rsp),%r14
    5cd0:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
    5cd5:	49 89 eb             	mov    %rbp,%r11
    5cd8:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    5cdd:	48 8b 34 24          	mov    (%rsp),%rsi
    5ce1:	31 c0                	xor    %eax,%eax
    5ce3:	eb 20                	jmp    5d05 <main_graph_mobilenet+0x3a35>
    5ce5:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    5cec:	00 00 00 
    5cef:	90                   	nop
    5cf0:	48 ff c0             	inc    %rax
    5cf3:	48 01 ce             	add    %rcx,%rsi
    5cf6:	48 01 cf             	add    %rcx,%rdi
    5cf9:	49 01 cb             	add    %rcx,%r11
    5cfc:	49 01 cc             	add    %rcx,%r12
    5cff:	48 83 f8 40          	cmp    $0x40,%rax
    5d03:	74 8b                	je     5c90 <main_graph_mobilenet+0x39c0>
    5d05:	49 83 fd 08          	cmp    $0x8,%r13
    5d09:	f3 41 0f 10 04 81    	movss  (%r9,%rax,4),%xmm0
    5d0f:	73 0f                	jae    5d20 <main_graph_mobilenet+0x3a50>
    5d11:	31 d2                	xor    %edx,%edx
    5d13:	eb 6b                	jmp    5d80 <main_graph_mobilenet+0x3ab0>
    5d15:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    5d1c:	00 00 00 
    5d1f:	90                   	nop
    5d20:	48 89 ca             	mov    %rcx,%rdx
    5d23:	48 0f af d0          	imul   %rax,%rdx
    5d27:	4d 8d 14 16          	lea    (%r14,%rdx,1),%r10
    5d2b:	4c 01 fa             	add    %r15,%rdx
    5d2e:	4c 29 d2             	sub    %r10,%rdx
    5d31:	48 83 fa 20          	cmp    $0x20,%rdx
    5d35:	73 09                	jae    5d40 <main_graph_mobilenet+0x3a70>
    5d37:	31 d2                	xor    %edx,%edx
    5d39:	eb 45                	jmp    5d80 <main_graph_mobilenet+0x3ab0>
    5d3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    5d40:	0f 28 c8             	movaps %xmm0,%xmm1
    5d43:	0f c6 c8 00          	shufps $0x0,%xmm0,%xmm1
    5d47:	31 d2                	xor    %edx,%edx
    5d49:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    5d50:	0f 10 54 97 f0       	movups -0x10(%rdi,%rdx,4),%xmm2
    5d55:	0f 10 1c 97          	movups (%rdi,%rdx,4),%xmm3
    5d59:	0f 58 d1             	addps  %xmm1,%xmm2
    5d5c:	0f 58 d9             	addps  %xmm1,%xmm3
    5d5f:	0f 11 54 96 f0       	movups %xmm2,-0x10(%rsi,%rdx,4)
    5d64:	0f 11 1c 96          	movups %xmm3,(%rsi,%rdx,4)
    5d68:	48 83 c2 08          	add    $0x8,%rdx
    5d6c:	48 39 d3             	cmp    %rdx,%rbx
    5d6f:	75 df                	jne    5d50 <main_graph_mobilenet+0x3a80>
    5d71:	49 39 dd             	cmp    %rbx,%r13
    5d74:	48 89 da             	mov    %rbx,%rdx
    5d77:	0f 84 73 ff ff ff    	je     5cf0 <main_graph_mobilenet+0x3a20>
    5d7d:	0f 1f 00             	nopl   (%rax)
    5d80:	f3 41 0f 10 0c 94    	movss  (%r12,%rdx,4),%xmm1
    5d86:	f3 0f 58 c8          	addss  %xmm0,%xmm1
    5d8a:	f3 41 0f 11 0c 93    	movss  %xmm1,(%r11,%rdx,4)
    5d90:	48 ff c2             	inc    %rdx
    5d93:	49 39 d5             	cmp    %rdx,%r13
    5d96:	75 e8                	jne    5d80 <main_graph_mobilenet+0x3ab0>
    5d98:	e9 53 ff ff ff       	jmp    5cf0 <main_graph_mobilenet+0x3a20>
    5d9d:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    5da2:	e8 69 c3 ff ff       	call   2110 <free@plt>
    5da7:	bf 30 00 00 00       	mov    $0x30,%edi
    5dac:	e8 1f c3 ff ff       	call   20d0 <malloc@plt>
    5db1:	49 89 c7             	mov    %rax,%r15
    5db4:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    5db8:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    5dbc:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    5dc1:	48 8b 00             	mov    (%rax),%rax
    5dc4:	49 89 04 24          	mov    %rax,(%r12)
    5dc8:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    5dcd:	e8 3e c3 ff ff       	call   2110 <free@plt>
    5dd2:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    5dd7:	4c 0f af e8          	imul   %rax,%r13
    5ddb:	49 c1 e5 06          	shl    $0x6,%r13
    5ddf:	49 8b 0c 24          	mov    (%r12),%rcx
    5de3:	48 85 c9             	test   %rcx,%rcx
    5de6:	48 0f 44 c8          	cmove  %rax,%rcx
    5dea:	48 89 c8             	mov    %rcx,%rax
    5ded:	48 c1 e0 0a          	shl    $0xa,%rax
    5df1:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    5df5:	48 8d 34 c0          	lea    (%rax,%rax,8),%rsi
    5df9:	b8 00 24 00 00       	mov    $0x2400,%eax
    5dfe:	48 0f 44 f0          	cmove  %rax,%rsi
    5e02:	48 89 f0             	mov    %rsi,%rax
    5e05:	48 f7 d0             	not    %rax
    5e08:	48 c1 f8 3f          	sar    $0x3f,%rax
    5e0c:	48 83 c8 01          	or     $0x1,%rax
    5e10:	4c 29 e8             	sub    %r13,%rax
    5e13:	48 99                	cqto   
    5e15:	48 f7 fe             	idiv   %rsi
    5e18:	49 89 c6             	mov    %rax,%r14
    5e1b:	49 f7 d6             	not    %r14
    5e1e:	4c 89 e8             	mov    %r13,%rax
    5e21:	48 99                	cqto   
    5e23:	48 f7 fe             	idiv   %rsi
    5e26:	48 85 f6             	test   %rsi,%rsi
    5e29:	48 89 c2             	mov    %rax,%rdx
    5e2c:	49 0f 48 d6          	cmovs  %r14,%rdx
    5e30:	4d 85 ed             	test   %r13,%r13
    5e33:	48 0f 4e d0          	cmovle %rax,%rdx
    5e37:	48 85 f6             	test   %rsi,%rsi
    5e3a:	4c 0f 4e f2          	cmovle %rdx,%r14
    5e3e:	4d 85 ed             	test   %r13,%r13
    5e41:	4c 0f 49 f2          	cmovns %rdx,%r14
    5e45:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    5e49:	4c 0f 45 f1          	cmovne %rcx,%r14
    5e4d:	4c 89 f0             	mov    %r14,%rax
    5e50:	48 c1 e0 0a          	shl    $0xa,%rax
    5e54:	48 8d 0c c0          	lea    (%rax,%rax,8),%rcx
    5e58:	4c 89 f6             	mov    %r14,%rsi
    5e5b:	48 c1 e6 06          	shl    $0x6,%rsi
    5e5f:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    5e63:	b8 40 00 00 00       	mov    $0x40,%eax
    5e68:	48 0f 44 f0          	cmove  %rax,%rsi
    5e6c:	48 89 f0             	mov    %rsi,%rax
    5e6f:	48 f7 d0             	not    %rax
    5e72:	48 c1 f8 3f          	sar    $0x3f,%rax
    5e76:	48 83 c8 01          	or     $0x1,%rax
    5e7a:	48 29 c8             	sub    %rcx,%rax
    5e7d:	48 99                	cqto   
    5e7f:	48 f7 fe             	idiv   %rsi
    5e82:	48 89 c5             	mov    %rax,%rbp
    5e85:	48 f7 d5             	not    %rbp
    5e88:	48 89 c8             	mov    %rcx,%rax
    5e8b:	48 99                	cqto   
    5e8d:	48 f7 fe             	idiv   %rsi
    5e90:	48 85 f6             	test   %rsi,%rsi
    5e93:	48 89 c2             	mov    %rax,%rdx
    5e96:	48 0f 48 d5          	cmovs  %rbp,%rdx
    5e9a:	48 85 c9             	test   %rcx,%rcx
    5e9d:	48 0f 4e d0          	cmovle %rax,%rdx
    5ea1:	48 85 f6             	test   %rsi,%rsi
    5ea4:	48 0f 4e ea          	cmovle %rdx,%rbp
    5ea8:	48 85 c9             	test   %rcx,%rcx
    5eab:	48 0f 49 ea          	cmovns %rdx,%rbp
    5eaf:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    5eb3:	49 c7 44 24 08 40 00 	movq   $0x40,0x8(%r12)
    5eba:	00 00 
    5ebc:	49 c7 44 24 10 0c 00 	movq   $0xc,0x10(%r12)
    5ec3:	00 00 
    5ec5:	49 c7 44 24 18 0c 00 	movq   $0xc,0x18(%r12)
    5ecc:	00 00 
    5ece:	4c 89 f0             	mov    %r14,%rax
    5ed1:	48 0f 44 c5          	cmove  %rbp,%rax
    5ed5:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    5eda:	4c 89 ff             	mov    %r15,%rdi
    5edd:	e8 2e c2 ff ff       	call   2110 <free@plt>
    5ee2:	4c 89 f0             	mov    %r14,%rax
    5ee5:	48 c1 e0 0c          	shl    $0xc,%rax
    5ee9:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    5eee:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    5ef3:	e8 d8 c1 ff ff       	call   20d0 <malloc@plt>
    5ef8:	49 89 c7             	mov    %rax,%r15
    5efb:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    5eff:	48 89 04 24          	mov    %rax,(%rsp)
    5f03:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    5f07:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    5f0c:	bf 04 00 00 00       	mov    $0x4,%edi
    5f11:	e8 8a c2 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    5f16:	48 89 c3             	mov    %rax,%rbx
    5f19:	48 89 c7             	mov    %rax,%rdi
    5f1c:	31 f6                	xor    %esi,%esi
    5f1e:	4c 89 fa             	mov    %r15,%rdx
    5f21:	4c 89 e1             	mov    %r12,%rcx
    5f24:	e8 c7 c1 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    5f29:	be 01 00 00 00       	mov    $0x1,%esi
    5f2e:	48 89 df             	mov    %rbx,%rdi
    5f31:	e8 6a c1 ff ff       	call   20a0 <omTensorSetDataType@plt>
    5f36:	48 89 df             	mov    %rbx,%rdi
    5f39:	e8 92 c2 ff ff       	call   21d0 <omTensorGetShape@plt>
    5f3e:	49 89 c4             	mov    %rax,%r12
    5f41:	48 89 df             	mov    %rbx,%rdi
    5f44:	e8 27 c1 ff ff       	call   2070 <omTensorGetStrides@plt>
    5f49:	4d 89 34 24          	mov    %r14,(%r12)
    5f4d:	48 c7 00 00 24 00 00 	movq   $0x2400,(%rax)
    5f54:	49 c7 44 24 08 40 00 	movq   $0x40,0x8(%r12)
    5f5b:	00 00 
    5f5d:	48 c7 40 08 90 00 00 	movq   $0x90,0x8(%rax)
    5f64:	00 
    5f65:	49 c7 44 24 10 0c 00 	movq   $0xc,0x10(%r12)
    5f6c:	00 00 
    5f6e:	48 c7 40 10 0c 00 00 	movq   $0xc,0x10(%rax)
    5f75:	00 
    5f76:	49 c7 44 24 18 0c 00 	movq   $0xc,0x18(%r12)
    5f7d:	00 00 
    5f7f:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    5f86:	00 
    5f87:	bf 04 00 00 00       	mov    $0x4,%edi
    5f8c:	e8 0f c2 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    5f91:	49 89 c4             	mov    %rax,%r12
    5f94:	48 89 c7             	mov    %rax,%rdi
    5f97:	31 f6                	xor    %esi,%esi
    5f99:	4c 8b 7c 24 50       	mov    0x50(%rsp),%r15
    5f9e:	4c 89 fa             	mov    %r15,%rdx
    5fa1:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    5fa6:	e8 45 c1 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    5fab:	be 01 00 00 00       	mov    $0x1,%esi
    5fb0:	4c 89 e7             	mov    %r12,%rdi
    5fb3:	e8 e8 c0 ff ff       	call   20a0 <omTensorSetDataType@plt>
    5fb8:	4c 89 e7             	mov    %r12,%rdi
    5fbb:	e8 10 c2 ff ff       	call   21d0 <omTensorGetShape@plt>
    5fc0:	49 89 c5             	mov    %rax,%r13
    5fc3:	4c 89 e7             	mov    %r12,%rdi
    5fc6:	e8 a5 c0 ff ff       	call   2070 <omTensorGetStrides@plt>
    5fcb:	4d 89 75 00          	mov    %r14,0x0(%r13)
    5fcf:	48 c7 00 00 24 00 00 	movq   $0x2400,(%rax)
    5fd6:	49 c7 45 08 40 00 00 	movq   $0x40,0x8(%r13)
    5fdd:	00 
    5fde:	48 c7 40 08 90 00 00 	movq   $0x90,0x8(%rax)
    5fe5:	00 
    5fe6:	49 c7 45 10 0c 00 00 	movq   $0xc,0x10(%r13)
    5fed:	00 
    5fee:	48 c7 40 10 0c 00 00 	movq   $0xc,0x10(%rax)
    5ff5:	00 
    5ff6:	49 c7 45 18 0c 00 00 	movq   $0xc,0x18(%r13)
    5ffd:	00 
    5ffe:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    6005:	00 
    6006:	48 89 df             	mov    %rbx,%rdi
    6009:	4c 89 e6             	mov    %r12,%rsi
    600c:	e8 df c1 ff ff       	call   21f0 <Relu@plt>
    6011:	48 89 df             	mov    %rbx,%rdi
    6014:	e8 97 c1 ff ff       	call   21b0 <omTensorDestroy@plt>
    6019:	4c 89 e7             	mov    %r12,%rdi
    601c:	e8 8f c1 ff ff       	call   21b0 <omTensorDestroy@plt>
    6021:	4c 89 ff             	mov    %r15,%rdi
    6024:	e8 e7 c0 ff ff       	call   2110 <free@plt>
    6029:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    602e:	e8 9d c0 ff ff       	call   20d0 <malloc@plt>
    6033:	48 89 c3             	mov    %rax,%rbx
    6036:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    603a:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    603f:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    6043:	4c 89 7c 24 20       	mov    %r15,0x20(%rsp)
    6048:	bf 04 00 00 00       	mov    $0x4,%edi
    604d:	e8 4e c1 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    6052:	49 89 c4             	mov    %rax,%r12
    6055:	48 89 c7             	mov    %rax,%rdi
    6058:	31 f6                	xor    %esi,%esi
    605a:	48 89 da             	mov    %rbx,%rdx
    605d:	4c 89 f9             	mov    %r15,%rcx
    6060:	e8 8b c0 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    6065:	be 01 00 00 00       	mov    $0x1,%esi
    606a:	4c 89 e3             	mov    %r12,%rbx
    606d:	4c 89 64 24 40       	mov    %r12,0x40(%rsp)
    6072:	4c 89 e7             	mov    %r12,%rdi
    6075:	e8 26 c0 ff ff       	call   20a0 <omTensorSetDataType@plt>
    607a:	4c 89 e7             	mov    %r12,%rdi
    607d:	e8 4e c1 ff ff       	call   21d0 <omTensorGetShape@plt>
    6082:	49 89 c4             	mov    %rax,%r12
    6085:	48 89 df             	mov    %rbx,%rdi
    6088:	e8 e3 bf ff ff       	call   2070 <omTensorGetStrides@plt>
    608d:	4d 89 34 24          	mov    %r14,(%r12)
    6091:	48 c7 00 00 24 00 00 	movq   $0x2400,(%rax)
    6098:	49 c7 44 24 08 40 00 	movq   $0x40,0x8(%r12)
    609f:	00 00 
    60a1:	48 c7 40 08 90 00 00 	movq   $0x90,0x8(%rax)
    60a8:	00 
    60a9:	49 c7 44 24 10 0c 00 	movq   $0xc,0x10(%r12)
    60b0:	00 00 
    60b2:	48 c7 40 10 0c 00 00 	movq   $0xc,0x10(%rax)
    60b9:	00 
    60ba:	49 c7 44 24 18 0c 00 	movq   $0xc,0x18(%r12)
    60c1:	00 00 
    60c3:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    60ca:	00 
    60cb:	bf 04 00 00 00       	mov    $0x4,%edi
    60d0:	e8 cb c0 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    60d5:	48 89 c3             	mov    %rax,%rbx
    60d8:	48 89 c7             	mov    %rax,%rdi
    60db:	31 f6                	xor    %esi,%esi
    60dd:	48 8b 14 24          	mov    (%rsp),%rdx
    60e1:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    60e6:	e8 05 c0 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    60eb:	be 01 00 00 00       	mov    $0x1,%esi
    60f0:	48 89 df             	mov    %rbx,%rdi
    60f3:	e8 a8 bf ff ff       	call   20a0 <omTensorSetDataType@plt>
    60f8:	48 89 df             	mov    %rbx,%rdi
    60fb:	e8 d0 c0 ff ff       	call   21d0 <omTensorGetShape@plt>
    6100:	49 89 c7             	mov    %rax,%r15
    6103:	48 89 df             	mov    %rbx,%rdi
    6106:	e8 65 bf ff ff       	call   2070 <omTensorGetStrides@plt>
    610b:	4d 89 37             	mov    %r14,(%r15)
    610e:	48 c7 00 00 24 00 00 	movq   $0x2400,(%rax)
    6115:	49 c7 47 08 40 00 00 	movq   $0x40,0x8(%r15)
    611c:	00 
    611d:	48 c7 40 08 90 00 00 	movq   $0x90,0x8(%rax)
    6124:	00 
    6125:	49 c7 47 10 0c 00 00 	movq   $0xc,0x10(%r15)
    612c:	00 
    612d:	48 c7 40 10 0c 00 00 	movq   $0xc,0x10(%rax)
    6134:	00 
    6135:	49 c7 47 18 0c 00 00 	movq   $0xc,0x18(%r15)
    613c:	00 
    613d:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    6144:	00 
    6145:	4c 8d 25 84 92 00 00 	lea    0x9284(%rip),%r12        # f3d0 <constant_65_mobilenet>
    614c:	bf 04 00 00 00       	mov    $0x4,%edi
    6151:	e8 4a c0 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    6156:	49 89 c7             	mov    %rax,%r15
    6159:	48 89 c7             	mov    %rax,%rdi
    615c:	31 f6                	xor    %esi,%esi
    615e:	4c 89 e2             	mov    %r12,%rdx
    6161:	4c 89 e1             	mov    %r12,%rcx
    6164:	e8 87 bf ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    6169:	be 01 00 00 00       	mov    $0x1,%esi
    616e:	4c 89 ff             	mov    %r15,%rdi
    6171:	e8 2a bf ff ff       	call   20a0 <omTensorSetDataType@plt>
    6176:	4c 89 ff             	mov    %r15,%rdi
    6179:	e8 52 c0 ff ff       	call   21d0 <omTensorGetShape@plt>
    617e:	49 89 c4             	mov    %rax,%r12
    6181:	4c 89 ff             	mov    %r15,%rdi
    6184:	e8 e7 be ff ff       	call   2070 <omTensorGetStrides@plt>
    6189:	49 c7 04 24 40 00 00 	movq   $0x40,(%r12)
    6190:	00 
    6191:	48 c7 00 09 00 00 00 	movq   $0x9,(%rax)
    6198:	49 c7 44 24 08 01 00 	movq   $0x1,0x8(%r12)
    619f:	00 00 
    61a1:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    61a8:	00 
    61a9:	49 c7 44 24 10 03 00 	movq   $0x3,0x10(%r12)
    61b0:	00 00 
    61b2:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    61b9:	00 
    61ba:	49 c7 44 24 18 03 00 	movq   $0x3,0x18(%r12)
    61c1:	00 00 
    61c3:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    61ca:	00 
    61cb:	4c 8d 2d 3e 4d 01 00 	lea    0x14d3e(%rip),%r13        # 1af10 <constant_39_mobilenet>
    61d2:	bf 01 00 00 00       	mov    $0x1,%edi
    61d7:	e8 c4 bf ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    61dc:	49 89 c4             	mov    %rax,%r12
    61df:	48 89 c7             	mov    %rax,%rdi
    61e2:	31 f6                	xor    %esi,%esi
    61e4:	4c 89 ea             	mov    %r13,%rdx
    61e7:	4c 89 e9             	mov    %r13,%rcx
    61ea:	e8 01 bf ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    61ef:	be 01 00 00 00       	mov    $0x1,%esi
    61f4:	4c 89 e7             	mov    %r12,%rdi
    61f7:	e8 a4 be ff ff       	call   20a0 <omTensorSetDataType@plt>
    61fc:	4c 89 e7             	mov    %r12,%rdi
    61ff:	e8 cc bf ff ff       	call   21d0 <omTensorGetShape@plt>
    6204:	49 89 c5             	mov    %rax,%r13
    6207:	4c 89 e7             	mov    %r12,%rdi
    620a:	e8 61 be ff ff       	call   2070 <omTensorGetStrides@plt>
    620f:	49 c7 45 00 40 00 00 	movq   $0x40,0x0(%r13)
    6216:	00 
    6217:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    621e:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    6223:	4c 89 ef             	mov    %r13,%rdi
    6226:	48 89 de             	mov    %rbx,%rsi
    6229:	4c 89 fa             	mov    %r15,%rdx
    622c:	4c 89 e1             	mov    %r12,%rcx
    622f:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    6235:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    623b:	6a 01                	push   $0x1
    623d:	6a 01                	push   $0x1
    623f:	6a 01                	push   $0x1
    6241:	6a 01                	push   $0x1
    6243:	e8 18 bf ff ff       	call   2160 <Conv2D_bias@plt>
    6248:	48 83 c4 20          	add    $0x20,%rsp
    624c:	4c 89 ef             	mov    %r13,%rdi
    624f:	e8 5c bf ff ff       	call   21b0 <omTensorDestroy@plt>
    6254:	48 89 df             	mov    %rbx,%rdi
    6257:	e8 54 bf ff ff       	call   21b0 <omTensorDestroy@plt>
    625c:	4c 89 ff             	mov    %r15,%rdi
    625f:	e8 4c bf ff ff       	call   21b0 <omTensorDestroy@plt>
    6264:	4c 89 e7             	mov    %r12,%rdi
    6267:	e8 44 bf ff ff       	call   21b0 <omTensorDestroy@plt>
    626c:	48 8b 3c 24          	mov    (%rsp),%rdi
    6270:	e8 9b be ff ff       	call   2110 <free@plt>
    6275:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    627a:	e8 51 be ff ff       	call   20d0 <malloc@plt>
    627f:	49 89 c7             	mov    %rax,%r15
    6282:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    6286:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    628b:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    628f:	4c 89 24 24          	mov    %r12,(%rsp)
    6293:	bf 04 00 00 00       	mov    $0x4,%edi
    6298:	e8 03 bf ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    629d:	48 89 c3             	mov    %rax,%rbx
    62a0:	48 89 c7             	mov    %rax,%rdi
    62a3:	31 f6                	xor    %esi,%esi
    62a5:	4c 89 fa             	mov    %r15,%rdx
    62a8:	4c 89 e1             	mov    %r12,%rcx
    62ab:	e8 40 be ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    62b0:	be 01 00 00 00       	mov    $0x1,%esi
    62b5:	48 89 df             	mov    %rbx,%rdi
    62b8:	e8 e3 bd ff ff       	call   20a0 <omTensorSetDataType@plt>
    62bd:	48 89 df             	mov    %rbx,%rdi
    62c0:	e8 0b bf ff ff       	call   21d0 <omTensorGetShape@plt>
    62c5:	49 89 c4             	mov    %rax,%r12
    62c8:	48 89 df             	mov    %rbx,%rdi
    62cb:	e8 a0 bd ff ff       	call   2070 <omTensorGetStrides@plt>
    62d0:	4d 89 34 24          	mov    %r14,(%r12)
    62d4:	48 c7 00 00 24 00 00 	movq   $0x2400,(%rax)
    62db:	49 c7 44 24 08 40 00 	movq   $0x40,0x8(%r12)
    62e2:	00 00 
    62e4:	48 c7 40 08 90 00 00 	movq   $0x90,0x8(%rax)
    62eb:	00 
    62ec:	49 c7 44 24 10 0c 00 	movq   $0xc,0x10(%r12)
    62f3:	00 00 
    62f5:	48 c7 40 10 0c 00 00 	movq   $0xc,0x10(%rax)
    62fc:	00 
    62fd:	49 c7 44 24 18 0c 00 	movq   $0xc,0x18(%r12)
    6304:	00 00 
    6306:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    630d:	00 
    630e:	bf 04 00 00 00       	mov    $0x4,%edi
    6313:	e8 88 be ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    6318:	49 89 c4             	mov    %rax,%r12
    631b:	48 89 c7             	mov    %rax,%rdi
    631e:	31 f6                	xor    %esi,%esi
    6320:	4c 8b 7c 24 30       	mov    0x30(%rsp),%r15
    6325:	4c 89 fa             	mov    %r15,%rdx
    6328:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    632d:	e8 be bd ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    6332:	be 01 00 00 00       	mov    $0x1,%esi
    6337:	4c 89 e7             	mov    %r12,%rdi
    633a:	e8 61 bd ff ff       	call   20a0 <omTensorSetDataType@plt>
    633f:	4c 89 e7             	mov    %r12,%rdi
    6342:	e8 89 be ff ff       	call   21d0 <omTensorGetShape@plt>
    6347:	49 89 c5             	mov    %rax,%r13
    634a:	4c 89 e7             	mov    %r12,%rdi
    634d:	e8 1e bd ff ff       	call   2070 <omTensorGetStrides@plt>
    6352:	4d 89 75 00          	mov    %r14,0x0(%r13)
    6356:	48 c7 00 00 24 00 00 	movq   $0x2400,(%rax)
    635d:	49 c7 45 08 40 00 00 	movq   $0x40,0x8(%r13)
    6364:	00 
    6365:	48 c7 40 08 90 00 00 	movq   $0x90,0x8(%rax)
    636c:	00 
    636d:	49 c7 45 10 0c 00 00 	movq   $0xc,0x10(%r13)
    6374:	00 
    6375:	48 c7 40 10 0c 00 00 	movq   $0xc,0x10(%rax)
    637c:	00 
    637d:	49 c7 45 18 0c 00 00 	movq   $0xc,0x18(%r13)
    6384:	00 
    6385:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    638c:	00 
    638d:	48 89 df             	mov    %rbx,%rdi
    6390:	4c 89 e6             	mov    %r12,%rsi
    6393:	e8 58 be ff ff       	call   21f0 <Relu@plt>
    6398:	48 89 df             	mov    %rbx,%rdi
    639b:	e8 10 be ff ff       	call   21b0 <omTensorDestroy@plt>
    63a0:	4c 89 e7             	mov    %r12,%rdi
    63a3:	e8 08 be ff ff       	call   21b0 <omTensorDestroy@plt>
    63a8:	4c 89 ff             	mov    %r15,%rdi
    63ab:	e8 60 bd ff ff       	call   2110 <free@plt>
    63b0:	bf 18 00 00 00       	mov    $0x18,%edi
    63b5:	e8 16 bd ff ff       	call   20d0 <malloc@plt>
    63ba:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    63bf:	48 83 c0 0f          	add    $0xf,%rax
    63c3:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    63c7:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    63cc:	4c 89 30             	mov    %r14,(%rax)
    63cf:	bf 28 00 00 00       	mov    $0x28,%edi
    63d4:	e8 f7 bc ff ff       	call   20d0 <malloc@plt>
    63d9:	48 8d 48 0f          	lea    0xf(%rax),%rcx
    63dd:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
    63e1:	4c 89 31             	mov    %r14,(%rcx)
    63e4:	48 c7 41 08 40 00 00 	movq   $0x40,0x8(%rcx)
    63eb:	00 
    63ec:	48 c7 41 10 ff ff ff 	movq   $0xffffffffffffffff,0x10(%rcx)
    63f3:	ff 
    63f4:	48 89 c7             	mov    %rax,%rdi
    63f7:	e8 14 bd ff ff       	call   2110 <free@plt>
    63fc:	48 89 e9             	mov    %rbp,%rcx
    63ff:	48 c1 e1 06          	shl    $0x6,%rcx
    6403:	48 8b 5c 24 60       	mov    0x60(%rsp),%rbx
    6408:	48 89 d8             	mov    %rbx,%rax
    640b:	48 0f af c1          	imul   %rcx,%rax
    640f:	49 89 cf             	mov    %rcx,%r15
    6412:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
    6417:	48 8d 3c 85 10 00 00 	lea    0x10(,%rax,4),%rdi
    641e:	00 
    641f:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
    6424:	e8 a7 bc ff ff       	call   20d0 <malloc@plt>
    6429:	49 89 c4             	mov    %rax,%r12
    642c:	4c 8d 68 0f          	lea    0xf(%rax),%r13
    6430:	49 83 e5 f0          	and    $0xfffffffffffffff0,%r13
    6434:	bf 03 00 00 00       	mov    $0x3,%edi
    6439:	e8 62 bd ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    643e:	49 89 c6             	mov    %rax,%r14
    6441:	48 89 c7             	mov    %rax,%rdi
    6444:	31 f6                	xor    %esi,%esi
    6446:	4c 89 64 24 70       	mov    %r12,0x70(%rsp)
    644b:	4c 89 e2             	mov    %r12,%rdx
    644e:	4c 89 6c 24 08       	mov    %r13,0x8(%rsp)
    6453:	4c 89 e9             	mov    %r13,%rcx
    6456:	e8 95 bc ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    645b:	be 01 00 00 00       	mov    $0x1,%esi
    6460:	4c 89 f7             	mov    %r14,%rdi
    6463:	e8 38 bc ff ff       	call   20a0 <omTensorSetDataType@plt>
    6468:	4c 89 f7             	mov    %r14,%rdi
    646b:	e8 60 bd ff ff       	call   21d0 <omTensorGetShape@plt>
    6470:	49 89 c4             	mov    %rax,%r12
    6473:	4c 89 f7             	mov    %r14,%rdi
    6476:	e8 f5 bb ff ff       	call   2070 <omTensorGetStrides@plt>
    647b:	49 89 1c 24          	mov    %rbx,(%r12)
    647f:	4c 89 38             	mov    %r15,(%rax)
    6482:	49 c7 44 24 08 40 00 	movq   $0x40,0x8(%r12)
    6489:	00 00 
    648b:	48 89 68 08          	mov    %rbp,0x8(%rax)
    648f:	49 89 6c 24 10       	mov    %rbp,0x10(%r12)
    6494:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    649b:	00 
    649c:	bf 02 00 00 00       	mov    $0x2,%edi
    64a1:	e8 fa bc ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    64a6:	49 89 c5             	mov    %rax,%r13
    64a9:	48 8d 0d 60 e2 0c 00 	lea    0xce260(%rip),%rcx        # d4710 <constant_15_mobilenet>
    64b0:	48 89 c7             	mov    %rax,%rdi
    64b3:	31 f6                	xor    %esi,%esi
    64b5:	48 89 ca             	mov    %rcx,%rdx
    64b8:	e8 33 bc ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    64bd:	be 01 00 00 00       	mov    $0x1,%esi
    64c2:	4c 89 ef             	mov    %r13,%rdi
    64c5:	e8 d6 bb ff ff       	call   20a0 <omTensorSetDataType@plt>
    64ca:	4c 89 ef             	mov    %r13,%rdi
    64cd:	e8 fe bc ff ff       	call   21d0 <omTensorGetShape@plt>
    64d2:	49 89 c4             	mov    %rax,%r12
    64d5:	4c 89 ef             	mov    %r13,%rdi
    64d8:	e8 93 bb ff ff       	call   2070 <omTensorGetStrides@plt>
    64dd:	49 c7 04 24 40 00 00 	movq   $0x40,(%r12)
    64e4:	00 
    64e5:	48 c7 00 40 00 00 00 	movq   $0x40,(%rax)
    64ec:	49 c7 44 24 08 40 00 	movq   $0x40,0x8(%r12)
    64f3:	00 00 
    64f5:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    64fc:	00 
    64fd:	bf 03 00 00 00       	mov    $0x3,%edi
    6502:	e8 99 bc ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    6507:	49 89 c4             	mov    %rax,%r12
    650a:	48 89 c7             	mov    %rax,%rdi
    650d:	31 f6                	xor    %esi,%esi
    650f:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    6514:	48 8b 0c 24          	mov    (%rsp),%rcx
    6518:	e8 d3 bb ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    651d:	be 01 00 00 00       	mov    $0x1,%esi
    6522:	4c 89 e7             	mov    %r12,%rdi
    6525:	e8 76 bb ff ff       	call   20a0 <omTensorSetDataType@plt>
    652a:	4c 89 e7             	mov    %r12,%rdi
    652d:	e8 9e bc ff ff       	call   21d0 <omTensorGetShape@plt>
    6532:	49 89 c7             	mov    %rax,%r15
    6535:	4c 89 e7             	mov    %r12,%rdi
    6538:	e8 33 bb ff ff       	call   2070 <omTensorGetStrides@plt>
    653d:	49 89 1f             	mov    %rbx,(%r15)
    6540:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    6545:	48 89 08             	mov    %rcx,(%rax)
    6548:	49 c7 47 08 40 00 00 	movq   $0x40,0x8(%r15)
    654f:	00 
    6550:	48 89 68 08          	mov    %rbp,0x8(%rax)
    6554:	49 89 6f 10          	mov    %rbp,0x10(%r15)
    6558:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    655f:	00 
    6560:	4c 89 f7             	mov    %r14,%rdi
    6563:	4c 89 ee             	mov    %r13,%rsi
    6566:	4c 89 e2             	mov    %r12,%rdx
    6569:	e8 e2 ba ff ff       	call   2050 <MatMul@plt>
    656e:	4c 89 f7             	mov    %r14,%rdi
    6571:	e8 3a bc ff ff       	call   21b0 <omTensorDestroy@plt>
    6576:	4c 89 ef             	mov    %r13,%rdi
    6579:	e8 32 bc ff ff       	call   21b0 <omTensorDestroy@plt>
    657e:	4c 89 e7             	mov    %r12,%rdi
    6581:	e8 2a bc ff ff       	call   21b0 <omTensorDestroy@plt>
    6586:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    658b:	e8 80 bb ff ff       	call   2110 <free@plt>
    6590:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    6595:	e8 36 bb ff ff       	call   20d0 <malloc@plt>
    659a:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    659f:	48 83 c0 0f          	add    $0xf,%rax
    65a3:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    65a7:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    65ac:	48 85 db             	test   %rbx,%rbx
    65af:	0f 8e 58 01 00 00    	jle    670d <main_graph_mobilenet+0x443d>
    65b5:	48 89 e8             	mov    %rbp,%rax
    65b8:	48 c1 e0 08          	shl    $0x8,%rax
    65bc:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    65c1:	48 8d 0c ad 00 00 00 	lea    0x0(,%rbp,4),%rcx
    65c8:	00 
    65c9:	48 89 eb             	mov    %rbp,%rbx
    65cc:	48 83 e3 f8          	and    $0xfffffffffffffff8,%rbx
    65d0:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    65d5:	49 8d 45 10          	lea    0x10(%r13),%rax
    65d9:	48 89 04 24          	mov    %rax,(%rsp)
    65dd:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    65e2:	48 8d 50 10          	lea    0x10(%rax),%rdx
    65e6:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
    65eb:	45 31 c0             	xor    %r8d,%r8d
    65ee:	4c 8d 0d 1b 5e 01 00 	lea    0x15e1b(%rip),%r9        # 1c410 <constant_28_mobilenet>
    65f5:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    65fa:	eb 28                	jmp    6624 <main_graph_mobilenet+0x4354>
    65fc:	0f 1f 40 00          	nopl   0x0(%rax)
    6600:	49 ff c0             	inc    %r8
    6603:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    6608:	48 01 04 24          	add    %rax,(%rsp)
    660c:	48 01 44 24 20       	add    %rax,0x20(%rsp)
    6611:	49 01 c5             	add    %rax,%r13
    6614:	48 01 44 24 30       	add    %rax,0x30(%rsp)
    6619:	4c 3b 44 24 60       	cmp    0x60(%rsp),%r8
    661e:	0f 84 e9 00 00 00    	je     670d <main_graph_mobilenet+0x443d>
    6624:	48 85 ed             	test   %rbp,%rbp
    6627:	7e d7                	jle    6600 <main_graph_mobilenet+0x4330>
    6629:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
    662e:	4d 0f af f0          	imul   %r8,%r14
    6632:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    6637:	4e 8d 3c 30          	lea    (%rax,%r14,1),%r15
    663b:	4c 03 74 24 08       	add    0x8(%rsp),%r14
    6640:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
    6645:	4d 89 eb             	mov    %r13,%r11
    6648:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    664d:	48 8b 34 24          	mov    (%rsp),%rsi
    6651:	31 c0                	xor    %eax,%eax
    6653:	eb 20                	jmp    6675 <main_graph_mobilenet+0x43a5>
    6655:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    665c:	00 00 00 
    665f:	90                   	nop
    6660:	48 ff c0             	inc    %rax
    6663:	48 01 ce             	add    %rcx,%rsi
    6666:	48 01 cf             	add    %rcx,%rdi
    6669:	49 01 cb             	add    %rcx,%r11
    666c:	49 01 cc             	add    %rcx,%r12
    666f:	48 83 f8 40          	cmp    $0x40,%rax
    6673:	74 8b                	je     6600 <main_graph_mobilenet+0x4330>
    6675:	48 83 fd 08          	cmp    $0x8,%rbp
    6679:	f3 41 0f 10 04 81    	movss  (%r9,%rax,4),%xmm0
    667f:	73 0f                	jae    6690 <main_graph_mobilenet+0x43c0>
    6681:	31 d2                	xor    %edx,%edx
    6683:	eb 6b                	jmp    66f0 <main_graph_mobilenet+0x4420>
    6685:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    668c:	00 00 00 
    668f:	90                   	nop
    6690:	48 89 ca             	mov    %rcx,%rdx
    6693:	48 0f af d0          	imul   %rax,%rdx
    6697:	4d 8d 14 16          	lea    (%r14,%rdx,1),%r10
    669b:	4c 01 fa             	add    %r15,%rdx
    669e:	4c 29 d2             	sub    %r10,%rdx
    66a1:	48 83 fa 20          	cmp    $0x20,%rdx
    66a5:	73 09                	jae    66b0 <main_graph_mobilenet+0x43e0>
    66a7:	31 d2                	xor    %edx,%edx
    66a9:	eb 45                	jmp    66f0 <main_graph_mobilenet+0x4420>
    66ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    66b0:	0f 28 c8             	movaps %xmm0,%xmm1
    66b3:	0f c6 c8 00          	shufps $0x0,%xmm0,%xmm1
    66b7:	31 d2                	xor    %edx,%edx
    66b9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    66c0:	0f 10 54 97 f0       	movups -0x10(%rdi,%rdx,4),%xmm2
    66c5:	0f 10 1c 97          	movups (%rdi,%rdx,4),%xmm3
    66c9:	0f 58 d1             	addps  %xmm1,%xmm2
    66cc:	0f 58 d9             	addps  %xmm1,%xmm3
    66cf:	0f 11 54 96 f0       	movups %xmm2,-0x10(%rsi,%rdx,4)
    66d4:	0f 11 1c 96          	movups %xmm3,(%rsi,%rdx,4)
    66d8:	48 83 c2 08          	add    $0x8,%rdx
    66dc:	48 39 d3             	cmp    %rdx,%rbx
    66df:	75 df                	jne    66c0 <main_graph_mobilenet+0x43f0>
    66e1:	48 39 dd             	cmp    %rbx,%rbp
    66e4:	48 89 da             	mov    %rbx,%rdx
    66e7:	0f 84 73 ff ff ff    	je     6660 <main_graph_mobilenet+0x4390>
    66ed:	0f 1f 00             	nopl   (%rax)
    66f0:	f3 41 0f 10 0c 94    	movss  (%r12,%rdx,4),%xmm1
    66f6:	f3 0f 58 c8          	addss  %xmm0,%xmm1
    66fa:	f3 41 0f 11 0c 93    	movss  %xmm1,(%r11,%rdx,4)
    6700:	48 ff c2             	inc    %rdx
    6703:	48 39 d5             	cmp    %rdx,%rbp
    6706:	75 e8                	jne    66f0 <main_graph_mobilenet+0x4420>
    6708:	e9 53 ff ff ff       	jmp    6660 <main_graph_mobilenet+0x4390>
    670d:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    6712:	e8 f9 b9 ff ff       	call   2110 <free@plt>
    6717:	bf 30 00 00 00       	mov    $0x30,%edi
    671c:	e8 af b9 ff ff       	call   20d0 <malloc@plt>
    6721:	49 89 c4             	mov    %rax,%r12
    6724:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    6728:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    672c:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    6731:	48 8b 00             	mov    (%rax),%rax
    6734:	49 89 07             	mov    %rax,(%r15)
    6737:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    673c:	e8 cf b9 ff ff       	call   2110 <free@plt>
    6741:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    6746:	48 0f af e8          	imul   %rax,%rbp
    674a:	48 c1 e5 06          	shl    $0x6,%rbp
    674e:	49 8b 0f             	mov    (%r15),%rcx
    6751:	48 85 c9             	test   %rcx,%rcx
    6754:	48 0f 44 c8          	cmove  %rax,%rcx
    6758:	48 89 c8             	mov    %rcx,%rax
    675b:	48 c1 e0 0a          	shl    $0xa,%rax
    675f:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    6763:	48 8d 34 c0          	lea    (%rax,%rax,8),%rsi
    6767:	b8 00 24 00 00       	mov    $0x2400,%eax
    676c:	48 0f 44 f0          	cmove  %rax,%rsi
    6770:	48 89 f0             	mov    %rsi,%rax
    6773:	48 f7 d0             	not    %rax
    6776:	48 c1 f8 3f          	sar    $0x3f,%rax
    677a:	48 83 c8 01          	or     $0x1,%rax
    677e:	48 29 e8             	sub    %rbp,%rax
    6781:	48 99                	cqto   
    6783:	48 f7 fe             	idiv   %rsi
    6786:	49 89 c6             	mov    %rax,%r14
    6789:	49 f7 d6             	not    %r14
    678c:	48 89 e8             	mov    %rbp,%rax
    678f:	48 99                	cqto   
    6791:	48 f7 fe             	idiv   %rsi
    6794:	48 85 f6             	test   %rsi,%rsi
    6797:	48 89 c2             	mov    %rax,%rdx
    679a:	49 0f 48 d6          	cmovs  %r14,%rdx
    679e:	48 85 ed             	test   %rbp,%rbp
    67a1:	48 0f 4e d0          	cmovle %rax,%rdx
    67a5:	48 85 f6             	test   %rsi,%rsi
    67a8:	4c 0f 4e f2          	cmovle %rdx,%r14
    67ac:	48 85 ed             	test   %rbp,%rbp
    67af:	4c 0f 49 f2          	cmovns %rdx,%r14
    67b3:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    67b7:	4c 0f 45 f1          	cmovne %rcx,%r14
    67bb:	4c 89 f0             	mov    %r14,%rax
    67be:	48 c1 e0 08          	shl    $0x8,%rax
    67c2:	48 8d 0c c0          	lea    (%rax,%rax,8),%rcx
    67c6:	4c 89 f6             	mov    %r14,%rsi
    67c9:	48 c1 e6 06          	shl    $0x6,%rsi
    67cd:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    67d1:	b8 40 00 00 00       	mov    $0x40,%eax
    67d6:	48 0f 44 f0          	cmove  %rax,%rsi
    67da:	48 89 f0             	mov    %rsi,%rax
    67dd:	48 f7 d0             	not    %rax
    67e0:	48 c1 f8 3f          	sar    $0x3f,%rax
    67e4:	48 83 c8 01          	or     $0x1,%rax
    67e8:	48 29 c8             	sub    %rcx,%rax
    67eb:	48 99                	cqto   
    67ed:	48 f7 fe             	idiv   %rsi
    67f0:	49 89 c5             	mov    %rax,%r13
    67f3:	49 f7 d5             	not    %r13
    67f6:	48 89 c8             	mov    %rcx,%rax
    67f9:	48 99                	cqto   
    67fb:	48 f7 fe             	idiv   %rsi
    67fe:	48 85 f6             	test   %rsi,%rsi
    6801:	48 89 c2             	mov    %rax,%rdx
    6804:	49 0f 48 d5          	cmovs  %r13,%rdx
    6808:	48 85 c9             	test   %rcx,%rcx
    680b:	48 0f 4e d0          	cmovle %rax,%rdx
    680f:	48 85 f6             	test   %rsi,%rsi
    6812:	4c 0f 4e ea          	cmovle %rdx,%r13
    6816:	48 85 c9             	test   %rcx,%rcx
    6819:	4c 0f 49 ea          	cmovns %rdx,%r13
    681d:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    6821:	49 c7 47 08 40 00 00 	movq   $0x40,0x8(%r15)
    6828:	00 
    6829:	49 c7 47 10 0c 00 00 	movq   $0xc,0x10(%r15)
    6830:	00 
    6831:	49 c7 47 18 0c 00 00 	movq   $0xc,0x18(%r15)
    6838:	00 
    6839:	4c 89 f0             	mov    %r14,%rax
    683c:	49 0f 44 c5          	cmove  %r13,%rax
    6840:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    6845:	4c 89 e7             	mov    %r12,%rdi
    6848:	e8 c3 b8 ff ff       	call   2110 <free@plt>
    684d:	4c 89 f0             	mov    %r14,%rax
    6850:	48 c1 e0 0c          	shl    $0xc,%rax
    6854:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    6859:	e8 72 b8 ff ff       	call   20d0 <malloc@plt>
    685e:	48 89 c3             	mov    %rax,%rbx
    6861:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    6865:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    686a:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    686e:	4c 89 7c 24 08       	mov    %r15,0x8(%rsp)
    6873:	bf 04 00 00 00       	mov    $0x4,%edi
    6878:	e8 23 b9 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    687d:	48 89 c5             	mov    %rax,%rbp
    6880:	48 89 c7             	mov    %rax,%rdi
    6883:	31 f6                	xor    %esi,%esi
    6885:	48 89 da             	mov    %rbx,%rdx
    6888:	4c 89 f9             	mov    %r15,%rcx
    688b:	e8 60 b8 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    6890:	be 01 00 00 00       	mov    $0x1,%esi
    6895:	48 89 ef             	mov    %rbp,%rdi
    6898:	e8 03 b8 ff ff       	call   20a0 <omTensorSetDataType@plt>
    689d:	48 89 ef             	mov    %rbp,%rdi
    68a0:	e8 2b b9 ff ff       	call   21d0 <omTensorGetShape@plt>
    68a5:	49 89 c7             	mov    %rax,%r15
    68a8:	48 89 ef             	mov    %rbp,%rdi
    68ab:	e8 c0 b7 ff ff       	call   2070 <omTensorGetStrides@plt>
    68b0:	4d 89 37             	mov    %r14,(%r15)
    68b3:	48 c7 00 00 24 00 00 	movq   $0x2400,(%rax)
    68ba:	49 c7 47 08 40 00 00 	movq   $0x40,0x8(%r15)
    68c1:	00 
    68c2:	48 c7 40 08 90 00 00 	movq   $0x90,0x8(%rax)
    68c9:	00 
    68ca:	49 c7 47 10 0c 00 00 	movq   $0xc,0x10(%r15)
    68d1:	00 
    68d2:	48 c7 40 10 0c 00 00 	movq   $0xc,0x10(%rax)
    68d9:	00 
    68da:	49 c7 47 18 0c 00 00 	movq   $0xc,0x18(%r15)
    68e1:	00 
    68e2:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    68e9:	00 
    68ea:	bf 04 00 00 00       	mov    $0x4,%edi
    68ef:	e8 ac b8 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    68f4:	49 89 c7             	mov    %rax,%r15
    68f7:	48 89 c7             	mov    %rax,%rdi
    68fa:	31 f6                	xor    %esi,%esi
    68fc:	4c 8b 64 24 50       	mov    0x50(%rsp),%r12
    6901:	4c 89 e2             	mov    %r12,%rdx
    6904:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    6909:	e8 e2 b7 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    690e:	be 01 00 00 00       	mov    $0x1,%esi
    6913:	4c 89 ff             	mov    %r15,%rdi
    6916:	e8 85 b7 ff ff       	call   20a0 <omTensorSetDataType@plt>
    691b:	4c 89 ff             	mov    %r15,%rdi
    691e:	e8 ad b8 ff ff       	call   21d0 <omTensorGetShape@plt>
    6923:	48 89 c3             	mov    %rax,%rbx
    6926:	4c 89 ff             	mov    %r15,%rdi
    6929:	e8 42 b7 ff ff       	call   2070 <omTensorGetStrides@plt>
    692e:	4c 89 33             	mov    %r14,(%rbx)
    6931:	48 c7 00 00 24 00 00 	movq   $0x2400,(%rax)
    6938:	48 c7 43 08 40 00 00 	movq   $0x40,0x8(%rbx)
    693f:	00 
    6940:	48 c7 40 08 90 00 00 	movq   $0x90,0x8(%rax)
    6947:	00 
    6948:	48 c7 43 10 0c 00 00 	movq   $0xc,0x10(%rbx)
    694f:	00 
    6950:	48 c7 40 10 0c 00 00 	movq   $0xc,0x10(%rax)
    6957:	00 
    6958:	48 c7 43 18 0c 00 00 	movq   $0xc,0x18(%rbx)
    695f:	00 
    6960:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    6967:	00 
    6968:	48 89 ef             	mov    %rbp,%rdi
    696b:	4c 89 fe             	mov    %r15,%rsi
    696e:	e8 7d b8 ff ff       	call   21f0 <Relu@plt>
    6973:	48 89 ef             	mov    %rbp,%rdi
    6976:	e8 35 b8 ff ff       	call   21b0 <omTensorDestroy@plt>
    697b:	4c 89 ff             	mov    %r15,%rdi
    697e:	e8 2d b8 ff ff       	call   21b0 <omTensorDestroy@plt>
    6983:	4c 89 e7             	mov    %r12,%rdi
    6986:	e8 85 b7 ff ff       	call   2110 <free@plt>
    698b:	4c 89 f0             	mov    %r14,%rax
    698e:	48 c1 e0 0a          	shl    $0xa,%rax
    6992:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    6997:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
    699c:	e8 2f b7 ff ff       	call   20d0 <malloc@plt>
    69a1:	48 89 c3             	mov    %rax,%rbx
    69a4:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    69a8:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    69ad:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    69b1:	4c 89 3c 24          	mov    %r15,(%rsp)
    69b5:	bf 04 00 00 00       	mov    $0x4,%edi
    69ba:	e8 e1 b7 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    69bf:	49 89 c4             	mov    %rax,%r12
    69c2:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    69c7:	48 89 c7             	mov    %rax,%rdi
    69ca:	31 f6                	xor    %esi,%esi
    69cc:	48 89 da             	mov    %rbx,%rdx
    69cf:	4c 89 f9             	mov    %r15,%rcx
    69d2:	e8 19 b7 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    69d7:	be 01 00 00 00       	mov    $0x1,%esi
    69dc:	4c 89 e7             	mov    %r12,%rdi
    69df:	e8 bc b6 ff ff       	call   20a0 <omTensorSetDataType@plt>
    69e4:	4c 89 e7             	mov    %r12,%rdi
    69e7:	e8 e4 b7 ff ff       	call   21d0 <omTensorGetShape@plt>
    69ec:	49 89 c7             	mov    %rax,%r15
    69ef:	4c 89 e7             	mov    %r12,%rdi
    69f2:	e8 79 b6 ff ff       	call   2070 <omTensorGetStrides@plt>
    69f7:	4d 89 37             	mov    %r14,(%r15)
    69fa:	48 c7 00 00 09 00 00 	movq   $0x900,(%rax)
    6a01:	49 c7 47 08 40 00 00 	movq   $0x40,0x8(%r15)
    6a08:	00 
    6a09:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    6a10:	00 
    6a11:	49 c7 47 10 06 00 00 	movq   $0x6,0x10(%r15)
    6a18:	00 
    6a19:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    6a20:	00 
    6a21:	49 c7 47 18 06 00 00 	movq   $0x6,0x18(%r15)
    6a28:	00 
    6a29:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    6a30:	00 
    6a31:	bf 04 00 00 00       	mov    $0x4,%edi
    6a36:	e8 65 b7 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    6a3b:	49 89 c7             	mov    %rax,%r15
    6a3e:	48 89 c7             	mov    %rax,%rdi
    6a41:	31 f6                	xor    %esi,%esi
    6a43:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    6a48:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    6a4d:	e8 9e b6 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    6a52:	be 01 00 00 00       	mov    $0x1,%esi
    6a57:	4c 89 ff             	mov    %r15,%rdi
    6a5a:	e8 41 b6 ff ff       	call   20a0 <omTensorSetDataType@plt>
    6a5f:	4c 89 ff             	mov    %r15,%rdi
    6a62:	e8 69 b7 ff ff       	call   21d0 <omTensorGetShape@plt>
    6a67:	49 89 c4             	mov    %rax,%r12
    6a6a:	4c 89 ff             	mov    %r15,%rdi
    6a6d:	e8 fe b5 ff ff       	call   2070 <omTensorGetStrides@plt>
    6a72:	4d 89 34 24          	mov    %r14,(%r12)
    6a76:	48 c7 00 00 24 00 00 	movq   $0x2400,(%rax)
    6a7d:	49 c7 44 24 08 40 00 	movq   $0x40,0x8(%r12)
    6a84:	00 00 
    6a86:	48 c7 40 08 90 00 00 	movq   $0x90,0x8(%rax)
    6a8d:	00 
    6a8e:	49 c7 44 24 10 0c 00 	movq   $0xc,0x10(%r12)
    6a95:	00 00 
    6a97:	48 c7 40 10 0c 00 00 	movq   $0xc,0x10(%rax)
    6a9e:	00 
    6a9f:	49 c7 44 24 18 0c 00 	movq   $0xc,0x18(%r12)
    6aa6:	00 00 
    6aa8:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    6aaf:	00 
    6ab0:	48 8d 2d 19 78 00 00 	lea    0x7819(%rip),%rbp        # e2d0 <constant_67_mobilenet>
    6ab7:	bf 04 00 00 00       	mov    $0x4,%edi
    6abc:	e8 df b6 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    6ac1:	49 89 c4             	mov    %rax,%r12
    6ac4:	48 89 c7             	mov    %rax,%rdi
    6ac7:	31 f6                	xor    %esi,%esi
    6ac9:	48 89 ea             	mov    %rbp,%rdx
    6acc:	48 89 e9             	mov    %rbp,%rcx
    6acf:	e8 1c b6 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    6ad4:	be 01 00 00 00       	mov    $0x1,%esi
    6ad9:	4c 89 e7             	mov    %r12,%rdi
    6adc:	e8 bf b5 ff ff       	call   20a0 <omTensorSetDataType@plt>
    6ae1:	4c 89 e7             	mov    %r12,%rdi
    6ae4:	e8 e7 b6 ff ff       	call   21d0 <omTensorGetShape@plt>
    6ae9:	48 89 c5             	mov    %rax,%rbp
    6aec:	4c 89 e7             	mov    %r12,%rdi
    6aef:	e8 7c b5 ff ff       	call   2070 <omTensorGetStrides@plt>
    6af4:	48 c7 45 00 40 00 00 	movq   $0x40,0x0(%rbp)
    6afb:	00 
    6afc:	48 c7 00 09 00 00 00 	movq   $0x9,(%rax)
    6b03:	48 c7 45 08 01 00 00 	movq   $0x1,0x8(%rbp)
    6b0a:	00 
    6b0b:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    6b12:	00 
    6b13:	48 c7 45 10 03 00 00 	movq   $0x3,0x10(%rbp)
    6b1a:	00 
    6b1b:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    6b22:	00 
    6b23:	48 c7 45 18 03 00 00 	movq   $0x3,0x18(%rbp)
    6b2a:	00 
    6b2b:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    6b32:	00 
    6b33:	48 8d 1d 96 31 01 00 	lea    0x13196(%rip),%rbx        # 19cd0 <constant_50_mobilenet>
    6b3a:	bf 01 00 00 00       	mov    $0x1,%edi
    6b3f:	e8 5c b6 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    6b44:	48 89 c5             	mov    %rax,%rbp
    6b47:	48 89 c7             	mov    %rax,%rdi
    6b4a:	31 f6                	xor    %esi,%esi
    6b4c:	48 89 da             	mov    %rbx,%rdx
    6b4f:	48 89 d9             	mov    %rbx,%rcx
    6b52:	e8 99 b5 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    6b57:	be 01 00 00 00       	mov    $0x1,%esi
    6b5c:	48 89 ef             	mov    %rbp,%rdi
    6b5f:	e8 3c b5 ff ff       	call   20a0 <omTensorSetDataType@plt>
    6b64:	48 89 ef             	mov    %rbp,%rdi
    6b67:	e8 64 b6 ff ff       	call   21d0 <omTensorGetShape@plt>
    6b6c:	48 89 c3             	mov    %rax,%rbx
    6b6f:	48 89 ef             	mov    %rbp,%rdi
    6b72:	e8 f9 b4 ff ff       	call   2070 <omTensorGetStrides@plt>
    6b77:	48 c7 03 40 00 00 00 	movq   $0x40,(%rbx)
    6b7e:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    6b85:	48 8b 5c 24 40       	mov    0x40(%rsp),%rbx
    6b8a:	48 89 df             	mov    %rbx,%rdi
    6b8d:	4c 89 fe             	mov    %r15,%rsi
    6b90:	4c 89 e2             	mov    %r12,%rdx
    6b93:	48 89 e9             	mov    %rbp,%rcx
    6b96:	45 31 c0             	xor    %r8d,%r8d
    6b99:	45 31 c9             	xor    %r9d,%r9d
    6b9c:	6a 02                	push   $0x2
    6b9e:	6a 02                	push   $0x2
    6ba0:	6a 01                	push   $0x1
    6ba2:	6a 01                	push   $0x1
    6ba4:	e8 b7 b5 ff ff       	call   2160 <Conv2D_bias@plt>
    6ba9:	48 83 c4 20          	add    $0x20,%rsp
    6bad:	48 89 df             	mov    %rbx,%rdi
    6bb0:	e8 fb b5 ff ff       	call   21b0 <omTensorDestroy@plt>
    6bb5:	4c 89 ff             	mov    %r15,%rdi
    6bb8:	e8 f3 b5 ff ff       	call   21b0 <omTensorDestroy@plt>
    6bbd:	4c 89 e7             	mov    %r12,%rdi
    6bc0:	e8 eb b5 ff ff       	call   21b0 <omTensorDestroy@plt>
    6bc5:	48 89 ef             	mov    %rbp,%rdi
    6bc8:	e8 e3 b5 ff ff       	call   21b0 <omTensorDestroy@plt>
    6bcd:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    6bd2:	e8 39 b5 ff ff       	call   2110 <free@plt>
    6bd7:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    6bdc:	e8 ef b4 ff ff       	call   20d0 <malloc@plt>
    6be1:	48 89 c3             	mov    %rax,%rbx
    6be4:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    6be8:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    6bed:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    6bf1:	4c 89 64 24 30       	mov    %r12,0x30(%rsp)
    6bf6:	bf 04 00 00 00       	mov    $0x4,%edi
    6bfb:	e8 a0 b5 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    6c00:	49 89 c7             	mov    %rax,%r15
    6c03:	48 89 c7             	mov    %rax,%rdi
    6c06:	31 f6                	xor    %esi,%esi
    6c08:	48 89 da             	mov    %rbx,%rdx
    6c0b:	4c 89 e1             	mov    %r12,%rcx
    6c0e:	e8 dd b4 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    6c13:	be 01 00 00 00       	mov    $0x1,%esi
    6c18:	4c 89 ff             	mov    %r15,%rdi
    6c1b:	e8 80 b4 ff ff       	call   20a0 <omTensorSetDataType@plt>
    6c20:	4c 89 ff             	mov    %r15,%rdi
    6c23:	e8 a8 b5 ff ff       	call   21d0 <omTensorGetShape@plt>
    6c28:	48 89 c3             	mov    %rax,%rbx
    6c2b:	4c 89 ff             	mov    %r15,%rdi
    6c2e:	e8 3d b4 ff ff       	call   2070 <omTensorGetStrides@plt>
    6c33:	4c 89 33             	mov    %r14,(%rbx)
    6c36:	48 c7 00 00 09 00 00 	movq   $0x900,(%rax)
    6c3d:	48 c7 43 08 40 00 00 	movq   $0x40,0x8(%rbx)
    6c44:	00 
    6c45:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    6c4c:	00 
    6c4d:	48 c7 43 10 06 00 00 	movq   $0x6,0x10(%rbx)
    6c54:	00 
    6c55:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    6c5c:	00 
    6c5d:	48 c7 43 18 06 00 00 	movq   $0x6,0x18(%rbx)
    6c64:	00 
    6c65:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    6c6c:	00 
    6c6d:	bf 04 00 00 00       	mov    $0x4,%edi
    6c72:	e8 29 b5 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    6c77:	48 89 c3             	mov    %rax,%rbx
    6c7a:	48 89 c7             	mov    %rax,%rdi
    6c7d:	31 f6                	xor    %esi,%esi
    6c7f:	4c 8b 64 24 20       	mov    0x20(%rsp),%r12
    6c84:	4c 89 e2             	mov    %r12,%rdx
    6c87:	48 8b 0c 24          	mov    (%rsp),%rcx
    6c8b:	e8 60 b4 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    6c90:	be 01 00 00 00       	mov    $0x1,%esi
    6c95:	48 89 df             	mov    %rbx,%rdi
    6c98:	e8 03 b4 ff ff       	call   20a0 <omTensorSetDataType@plt>
    6c9d:	48 89 df             	mov    %rbx,%rdi
    6ca0:	e8 2b b5 ff ff       	call   21d0 <omTensorGetShape@plt>
    6ca5:	48 89 c5             	mov    %rax,%rbp
    6ca8:	48 89 df             	mov    %rbx,%rdi
    6cab:	e8 c0 b3 ff ff       	call   2070 <omTensorGetStrides@plt>
    6cb0:	4c 89 75 00          	mov    %r14,0x0(%rbp)
    6cb4:	48 c7 00 00 09 00 00 	movq   $0x900,(%rax)
    6cbb:	48 c7 45 08 40 00 00 	movq   $0x40,0x8(%rbp)
    6cc2:	00 
    6cc3:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    6cca:	00 
    6ccb:	48 c7 45 10 06 00 00 	movq   $0x6,0x10(%rbp)
    6cd2:	00 
    6cd3:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    6cda:	00 
    6cdb:	48 c7 45 18 06 00 00 	movq   $0x6,0x18(%rbp)
    6ce2:	00 
    6ce3:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    6cea:	00 
    6ceb:	4c 89 ff             	mov    %r15,%rdi
    6cee:	48 89 de             	mov    %rbx,%rsi
    6cf1:	e8 fa b4 ff ff       	call   21f0 <Relu@plt>
    6cf6:	4c 89 ff             	mov    %r15,%rdi
    6cf9:	e8 b2 b4 ff ff       	call   21b0 <omTensorDestroy@plt>
    6cfe:	48 89 df             	mov    %rbx,%rdi
    6d01:	e8 aa b4 ff ff       	call   21b0 <omTensorDestroy@plt>
    6d06:	4c 89 e7             	mov    %r12,%rdi
    6d09:	e8 02 b4 ff ff       	call   2110 <free@plt>
    6d0e:	bf 18 00 00 00       	mov    $0x18,%edi
    6d13:	e8 b8 b3 ff ff       	call   20d0 <malloc@plt>
    6d18:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    6d1d:	48 83 c0 0f          	add    $0xf,%rax
    6d21:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    6d25:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    6d2a:	4c 89 30             	mov    %r14,(%rax)
    6d2d:	bf 28 00 00 00       	mov    $0x28,%edi
    6d32:	e8 99 b3 ff ff       	call   20d0 <malloc@plt>
    6d37:	48 8d 48 0f          	lea    0xf(%rax),%rcx
    6d3b:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
    6d3f:	4c 89 31             	mov    %r14,(%rcx)
    6d42:	48 c7 41 08 40 00 00 	movq   $0x40,0x8(%rcx)
    6d49:	00 
    6d4a:	48 c7 41 10 ff ff ff 	movq   $0xffffffffffffffff,0x10(%rcx)
    6d51:	ff 
    6d52:	48 89 c7             	mov    %rax,%rdi
    6d55:	e8 b6 b3 ff ff       	call   2110 <free@plt>
    6d5a:	4c 89 e8             	mov    %r13,%rax
    6d5d:	48 c1 e0 06          	shl    $0x6,%rax
    6d61:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    6d66:	4c 89 eb             	mov    %r13,%rbx
    6d69:	48 c1 e3 07          	shl    $0x7,%rbx
    6d6d:	48 8b 6c 24 60       	mov    0x60(%rsp),%rbp
    6d72:	48 89 e8             	mov    %rbp,%rax
    6d75:	48 0f af c3          	imul   %rbx,%rax
    6d79:	48 8d 3c 85 10 00 00 	lea    0x10(,%rax,4),%rdi
    6d80:	00 
    6d81:	48 89 3c 24          	mov    %rdi,(%rsp)
    6d85:	e8 46 b3 ff ff       	call   20d0 <malloc@plt>
    6d8a:	49 89 c7             	mov    %rax,%r15
    6d8d:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    6d91:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    6d95:	bf 03 00 00 00       	mov    $0x3,%edi
    6d9a:	e8 01 b4 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    6d9f:	49 89 c6             	mov    %rax,%r14
    6da2:	48 89 c7             	mov    %rax,%rdi
    6da5:	31 f6                	xor    %esi,%esi
    6da7:	4c 89 7c 24 70       	mov    %r15,0x70(%rsp)
    6dac:	4c 89 fa             	mov    %r15,%rdx
    6daf:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    6db4:	4c 89 e1             	mov    %r12,%rcx
    6db7:	e8 34 b3 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    6dbc:	be 01 00 00 00       	mov    $0x1,%esi
    6dc1:	4c 89 f7             	mov    %r14,%rdi
    6dc4:	e8 d7 b2 ff ff       	call   20a0 <omTensorSetDataType@plt>
    6dc9:	4c 89 f7             	mov    %r14,%rdi
    6dcc:	e8 ff b3 ff ff       	call   21d0 <omTensorGetShape@plt>
    6dd1:	49 89 c7             	mov    %rax,%r15
    6dd4:	4c 89 f7             	mov    %r14,%rdi
    6dd7:	e8 94 b2 ff ff       	call   2070 <omTensorGetStrides@plt>
    6ddc:	49 89 2f             	mov    %rbp,(%r15)
    6ddf:	48 89 18             	mov    %rbx,(%rax)
    6de2:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    6de9:	00 
    6dea:	4c 89 68 08          	mov    %r13,0x8(%rax)
    6dee:	4d 89 6f 10          	mov    %r13,0x10(%r15)
    6df2:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    6df9:	00 
    6dfa:	bf 02 00 00 00       	mov    $0x2,%edi
    6dff:	e8 9c b3 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    6e04:	48 89 c3             	mov    %rax,%rbx
    6e07:	48 8d 0d 02 59 0c 00 	lea    0xc5902(%rip),%rcx        # cc710 <constant_16_mobilenet>
    6e0e:	48 89 c7             	mov    %rax,%rdi
    6e11:	31 f6                	xor    %esi,%esi
    6e13:	48 89 ca             	mov    %rcx,%rdx
    6e16:	e8 d5 b2 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    6e1b:	be 01 00 00 00       	mov    $0x1,%esi
    6e20:	48 89 df             	mov    %rbx,%rdi
    6e23:	e8 78 b2 ff ff       	call   20a0 <omTensorSetDataType@plt>
    6e28:	48 89 df             	mov    %rbx,%rdi
    6e2b:	e8 a0 b3 ff ff       	call   21d0 <omTensorGetShape@plt>
    6e30:	49 89 c7             	mov    %rax,%r15
    6e33:	48 89 df             	mov    %rbx,%rdi
    6e36:	e8 35 b2 ff ff       	call   2070 <omTensorGetStrides@plt>
    6e3b:	49 c7 07 80 00 00 00 	movq   $0x80,(%r15)
    6e42:	48 c7 00 40 00 00 00 	movq   $0x40,(%rax)
    6e49:	49 c7 47 08 40 00 00 	movq   $0x40,0x8(%r15)
    6e50:	00 
    6e51:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    6e58:	00 
    6e59:	bf 03 00 00 00       	mov    $0x3,%edi
    6e5e:	e8 3d b3 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    6e63:	49 89 c7             	mov    %rax,%r15
    6e66:	48 89 c7             	mov    %rax,%rdi
    6e69:	31 f6                	xor    %esi,%esi
    6e6b:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    6e70:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    6e75:	e8 76 b2 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    6e7a:	be 01 00 00 00       	mov    $0x1,%esi
    6e7f:	4c 89 ff             	mov    %r15,%rdi
    6e82:	e8 19 b2 ff ff       	call   20a0 <omTensorSetDataType@plt>
    6e87:	4c 89 ff             	mov    %r15,%rdi
    6e8a:	e8 41 b3 ff ff       	call   21d0 <omTensorGetShape@plt>
    6e8f:	49 89 c4             	mov    %rax,%r12
    6e92:	4c 89 ff             	mov    %r15,%rdi
    6e95:	e8 d6 b1 ff ff       	call   2070 <omTensorGetStrides@plt>
    6e9a:	49 89 2c 24          	mov    %rbp,(%r12)
    6e9e:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    6ea3:	48 89 08             	mov    %rcx,(%rax)
    6ea6:	49 c7 44 24 08 40 00 	movq   $0x40,0x8(%r12)
    6ead:	00 00 
    6eaf:	4c 89 68 08          	mov    %r13,0x8(%rax)
    6eb3:	4d 89 6c 24 10       	mov    %r13,0x10(%r12)
    6eb8:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    6ebf:	00 
    6ec0:	4c 89 f7             	mov    %r14,%rdi
    6ec3:	48 89 de             	mov    %rbx,%rsi
    6ec6:	4c 89 fa             	mov    %r15,%rdx
    6ec9:	e8 82 b1 ff ff       	call   2050 <MatMul@plt>
    6ece:	4c 89 f7             	mov    %r14,%rdi
    6ed1:	e8 da b2 ff ff       	call   21b0 <omTensorDestroy@plt>
    6ed6:	48 89 df             	mov    %rbx,%rdi
    6ed9:	e8 d2 b2 ff ff       	call   21b0 <omTensorDestroy@plt>
    6ede:	4c 89 ff             	mov    %r15,%rdi
    6ee1:	e8 ca b2 ff ff       	call   21b0 <omTensorDestroy@plt>
    6ee6:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    6eeb:	e8 20 b2 ff ff       	call   2110 <free@plt>
    6ef0:	48 8b 3c 24          	mov    (%rsp),%rdi
    6ef4:	e8 d7 b1 ff ff       	call   20d0 <malloc@plt>
    6ef9:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    6efe:	48 83 c0 0f          	add    $0xf,%rax
    6f02:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    6f06:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    6f0b:	48 85 ed             	test   %rbp,%rbp
    6f0e:	0f 8e 59 01 00 00    	jle    706d <main_graph_mobilenet+0x4d9d>
    6f14:	4c 89 e8             	mov    %r13,%rax
    6f17:	48 c1 e0 09          	shl    $0x9,%rax
    6f1b:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    6f20:	4a 8d 0c ad 00 00 00 	lea    0x0(,%r13,4),%rcx
    6f27:	00 
    6f28:	4c 89 eb             	mov    %r13,%rbx
    6f2b:	48 83 e3 f8          	and    $0xfffffffffffffff8,%rbx
    6f2f:	48 8b 6c 24 40       	mov    0x40(%rsp),%rbp
    6f34:	48 8d 45 10          	lea    0x10(%rbp),%rax
    6f38:	48 89 04 24          	mov    %rax,(%rsp)
    6f3c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    6f41:	48 8d 50 10          	lea    0x10(%rax),%rdx
    6f45:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
    6f4a:	45 31 c0             	xor    %r8d,%r8d
    6f4d:	4c 8d 0d bc 52 01 00 	lea    0x152bc(%rip),%r9        # 1c210 <constant_29_mobilenet>
    6f54:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    6f59:	eb 29                	jmp    6f84 <main_graph_mobilenet+0x4cb4>
    6f5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    6f60:	49 ff c0             	inc    %r8
    6f63:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    6f68:	48 01 04 24          	add    %rax,(%rsp)
    6f6c:	48 01 44 24 20       	add    %rax,0x20(%rsp)
    6f71:	48 01 c5             	add    %rax,%rbp
    6f74:	48 01 44 24 30       	add    %rax,0x30(%rsp)
    6f79:	4c 3b 44 24 60       	cmp    0x60(%rsp),%r8
    6f7e:	0f 84 e9 00 00 00    	je     706d <main_graph_mobilenet+0x4d9d>
    6f84:	4d 85 ed             	test   %r13,%r13
    6f87:	7e d7                	jle    6f60 <main_graph_mobilenet+0x4c90>
    6f89:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
    6f8e:	4d 0f af f0          	imul   %r8,%r14
    6f92:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    6f97:	4e 8d 3c 30          	lea    (%rax,%r14,1),%r15
    6f9b:	4c 03 74 24 08       	add    0x8(%rsp),%r14
    6fa0:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
    6fa5:	49 89 eb             	mov    %rbp,%r11
    6fa8:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    6fad:	48 8b 34 24          	mov    (%rsp),%rsi
    6fb1:	31 c0                	xor    %eax,%eax
    6fb3:	eb 22                	jmp    6fd7 <main_graph_mobilenet+0x4d07>
    6fb5:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    6fbc:	00 00 00 
    6fbf:	90                   	nop
    6fc0:	48 ff c0             	inc    %rax
    6fc3:	48 01 ce             	add    %rcx,%rsi
    6fc6:	48 01 cf             	add    %rcx,%rdi
    6fc9:	49 01 cb             	add    %rcx,%r11
    6fcc:	49 01 cc             	add    %rcx,%r12
    6fcf:	48 3d 80 00 00 00    	cmp    $0x80,%rax
    6fd5:	74 89                	je     6f60 <main_graph_mobilenet+0x4c90>
    6fd7:	49 83 fd 08          	cmp    $0x8,%r13
    6fdb:	f3 41 0f 10 04 81    	movss  (%r9,%rax,4),%xmm0
    6fe1:	73 0d                	jae    6ff0 <main_graph_mobilenet+0x4d20>
    6fe3:	31 d2                	xor    %edx,%edx
    6fe5:	eb 69                	jmp    7050 <main_graph_mobilenet+0x4d80>
    6fe7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    6fee:	00 00 
    6ff0:	48 89 ca             	mov    %rcx,%rdx
    6ff3:	48 0f af d0          	imul   %rax,%rdx
    6ff7:	4d 8d 14 16          	lea    (%r14,%rdx,1),%r10
    6ffb:	4c 01 fa             	add    %r15,%rdx
    6ffe:	4c 29 d2             	sub    %r10,%rdx
    7001:	48 83 fa 20          	cmp    $0x20,%rdx
    7005:	73 09                	jae    7010 <main_graph_mobilenet+0x4d40>
    7007:	31 d2                	xor    %edx,%edx
    7009:	eb 45                	jmp    7050 <main_graph_mobilenet+0x4d80>
    700b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    7010:	0f 28 c8             	movaps %xmm0,%xmm1
    7013:	0f c6 c8 00          	shufps $0x0,%xmm0,%xmm1
    7017:	31 d2                	xor    %edx,%edx
    7019:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7020:	0f 10 54 97 f0       	movups -0x10(%rdi,%rdx,4),%xmm2
    7025:	0f 10 1c 97          	movups (%rdi,%rdx,4),%xmm3
    7029:	0f 58 d1             	addps  %xmm1,%xmm2
    702c:	0f 58 d9             	addps  %xmm1,%xmm3
    702f:	0f 11 54 96 f0       	movups %xmm2,-0x10(%rsi,%rdx,4)
    7034:	0f 11 1c 96          	movups %xmm3,(%rsi,%rdx,4)
    7038:	48 83 c2 08          	add    $0x8,%rdx
    703c:	48 39 d3             	cmp    %rdx,%rbx
    703f:	75 df                	jne    7020 <main_graph_mobilenet+0x4d50>
    7041:	49 39 dd             	cmp    %rbx,%r13
    7044:	48 89 da             	mov    %rbx,%rdx
    7047:	0f 84 73 ff ff ff    	je     6fc0 <main_graph_mobilenet+0x4cf0>
    704d:	0f 1f 00             	nopl   (%rax)
    7050:	f3 41 0f 10 0c 94    	movss  (%r12,%rdx,4),%xmm1
    7056:	f3 0f 58 c8          	addss  %xmm0,%xmm1
    705a:	f3 41 0f 11 0c 93    	movss  %xmm1,(%r11,%rdx,4)
    7060:	48 ff c2             	inc    %rdx
    7063:	49 39 d5             	cmp    %rdx,%r13
    7066:	75 e8                	jne    7050 <main_graph_mobilenet+0x4d80>
    7068:	e9 53 ff ff ff       	jmp    6fc0 <main_graph_mobilenet+0x4cf0>
    706d:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    7072:	e8 99 b0 ff ff       	call   2110 <free@plt>
    7077:	bf 30 00 00 00       	mov    $0x30,%edi
    707c:	e8 4f b0 ff ff       	call   20d0 <malloc@plt>
    7081:	49 89 c7             	mov    %rax,%r15
    7084:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    7088:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    708c:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    7091:	48 8b 00             	mov    (%rax),%rax
    7094:	49 89 04 24          	mov    %rax,(%r12)
    7098:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    709d:	e8 6e b0 ff ff       	call   2110 <free@plt>
    70a2:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    70a7:	4c 0f af e8          	imul   %rax,%r13
    70ab:	49 c1 e5 07          	shl    $0x7,%r13
    70af:	49 8b 0c 24          	mov    (%r12),%rcx
    70b3:	48 85 c9             	test   %rcx,%rcx
    70b6:	48 0f 44 c8          	cmove  %rax,%rcx
    70ba:	48 89 c8             	mov    %rcx,%rax
    70bd:	48 c1 e0 09          	shl    $0x9,%rax
    70c1:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    70c5:	48 8d 34 c0          	lea    (%rax,%rax,8),%rsi
    70c9:	b8 00 12 00 00       	mov    $0x1200,%eax
    70ce:	48 0f 44 f0          	cmove  %rax,%rsi
    70d2:	48 89 f0             	mov    %rsi,%rax
    70d5:	48 f7 d0             	not    %rax
    70d8:	48 c1 f8 3f          	sar    $0x3f,%rax
    70dc:	48 83 c8 01          	or     $0x1,%rax
    70e0:	4c 29 e8             	sub    %r13,%rax
    70e3:	48 99                	cqto   
    70e5:	48 f7 fe             	idiv   %rsi
    70e8:	49 89 c6             	mov    %rax,%r14
    70eb:	49 f7 d6             	not    %r14
    70ee:	4c 89 e8             	mov    %r13,%rax
    70f1:	48 99                	cqto   
    70f3:	48 f7 fe             	idiv   %rsi
    70f6:	48 85 f6             	test   %rsi,%rsi
    70f9:	48 89 c2             	mov    %rax,%rdx
    70fc:	49 0f 48 d6          	cmovs  %r14,%rdx
    7100:	4d 85 ed             	test   %r13,%r13
    7103:	48 0f 4e d0          	cmovle %rax,%rdx
    7107:	48 85 f6             	test   %rsi,%rsi
    710a:	4c 0f 4e f2          	cmovle %rdx,%r14
    710e:	4d 85 ed             	test   %r13,%r13
    7111:	4c 0f 49 f2          	cmovns %rdx,%r14
    7115:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    7119:	4c 0f 45 f1          	cmovne %rcx,%r14
    711d:	4c 89 f0             	mov    %r14,%rax
    7120:	48 c1 e0 09          	shl    $0x9,%rax
    7124:	48 8d 0c c0          	lea    (%rax,%rax,8),%rcx
    7128:	4c 89 f6             	mov    %r14,%rsi
    712b:	48 c1 e6 07          	shl    $0x7,%rsi
    712f:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    7133:	b8 80 00 00 00       	mov    $0x80,%eax
    7138:	48 0f 44 f0          	cmove  %rax,%rsi
    713c:	48 89 f0             	mov    %rsi,%rax
    713f:	48 f7 d0             	not    %rax
    7142:	48 c1 f8 3f          	sar    $0x3f,%rax
    7146:	48 83 c8 01          	or     $0x1,%rax
    714a:	48 29 c8             	sub    %rcx,%rax
    714d:	48 99                	cqto   
    714f:	48 f7 fe             	idiv   %rsi
    7152:	48 89 c5             	mov    %rax,%rbp
    7155:	48 f7 d5             	not    %rbp
    7158:	48 89 c8             	mov    %rcx,%rax
    715b:	48 99                	cqto   
    715d:	48 f7 fe             	idiv   %rsi
    7160:	48 85 f6             	test   %rsi,%rsi
    7163:	48 89 c2             	mov    %rax,%rdx
    7166:	48 0f 48 d5          	cmovs  %rbp,%rdx
    716a:	48 85 c9             	test   %rcx,%rcx
    716d:	48 0f 4e d0          	cmovle %rax,%rdx
    7171:	48 85 f6             	test   %rsi,%rsi
    7174:	48 0f 4e ea          	cmovle %rdx,%rbp
    7178:	48 85 c9             	test   %rcx,%rcx
    717b:	48 0f 49 ea          	cmovns %rdx,%rbp
    717f:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    7183:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    718a:	00 00 
    718c:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    7193:	00 00 
    7195:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    719c:	00 00 
    719e:	4c 89 f0             	mov    %r14,%rax
    71a1:	48 0f 44 c5          	cmove  %rbp,%rax
    71a5:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    71aa:	4c 89 ff             	mov    %r15,%rdi
    71ad:	e8 5e af ff ff       	call   2110 <free@plt>
    71b2:	4c 89 f0             	mov    %r14,%rax
    71b5:	48 c1 e0 0b          	shl    $0xb,%rax
    71b9:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    71be:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    71c3:	e8 08 af ff ff       	call   20d0 <malloc@plt>
    71c8:	49 89 c7             	mov    %rax,%r15
    71cb:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    71cf:	48 89 04 24          	mov    %rax,(%rsp)
    71d3:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    71d7:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    71dc:	bf 04 00 00 00       	mov    $0x4,%edi
    71e1:	e8 ba af ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    71e6:	48 89 c3             	mov    %rax,%rbx
    71e9:	48 89 c7             	mov    %rax,%rdi
    71ec:	31 f6                	xor    %esi,%esi
    71ee:	4c 89 fa             	mov    %r15,%rdx
    71f1:	4c 89 e1             	mov    %r12,%rcx
    71f4:	e8 f7 ae ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    71f9:	be 01 00 00 00       	mov    $0x1,%esi
    71fe:	48 89 df             	mov    %rbx,%rdi
    7201:	e8 9a ae ff ff       	call   20a0 <omTensorSetDataType@plt>
    7206:	48 89 df             	mov    %rbx,%rdi
    7209:	e8 c2 af ff ff       	call   21d0 <omTensorGetShape@plt>
    720e:	49 89 c4             	mov    %rax,%r12
    7211:	48 89 df             	mov    %rbx,%rdi
    7214:	e8 57 ae ff ff       	call   2070 <omTensorGetStrides@plt>
    7219:	4d 89 34 24          	mov    %r14,(%r12)
    721d:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    7224:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    722b:	00 00 
    722d:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    7234:	00 
    7235:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    723c:	00 00 
    723e:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    7245:	00 
    7246:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    724d:	00 00 
    724f:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    7256:	00 
    7257:	bf 04 00 00 00       	mov    $0x4,%edi
    725c:	e8 3f af ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    7261:	49 89 c4             	mov    %rax,%r12
    7264:	48 89 c7             	mov    %rax,%rdi
    7267:	31 f6                	xor    %esi,%esi
    7269:	4c 8b 7c 24 50       	mov    0x50(%rsp),%r15
    726e:	4c 89 fa             	mov    %r15,%rdx
    7271:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    7276:	e8 75 ae ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    727b:	be 01 00 00 00       	mov    $0x1,%esi
    7280:	4c 89 e7             	mov    %r12,%rdi
    7283:	e8 18 ae ff ff       	call   20a0 <omTensorSetDataType@plt>
    7288:	4c 89 e7             	mov    %r12,%rdi
    728b:	e8 40 af ff ff       	call   21d0 <omTensorGetShape@plt>
    7290:	49 89 c5             	mov    %rax,%r13
    7293:	4c 89 e7             	mov    %r12,%rdi
    7296:	e8 d5 ad ff ff       	call   2070 <omTensorGetStrides@plt>
    729b:	4d 89 75 00          	mov    %r14,0x0(%r13)
    729f:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    72a6:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    72ad:	00 
    72ae:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    72b5:	00 
    72b6:	49 c7 45 10 06 00 00 	movq   $0x6,0x10(%r13)
    72bd:	00 
    72be:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    72c5:	00 
    72c6:	49 c7 45 18 06 00 00 	movq   $0x6,0x18(%r13)
    72cd:	00 
    72ce:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    72d5:	00 
    72d6:	48 89 df             	mov    %rbx,%rdi
    72d9:	4c 89 e6             	mov    %r12,%rsi
    72dc:	e8 0f af ff ff       	call   21f0 <Relu@plt>
    72e1:	48 89 df             	mov    %rbx,%rdi
    72e4:	e8 c7 ae ff ff       	call   21b0 <omTensorDestroy@plt>
    72e9:	4c 89 e7             	mov    %r12,%rdi
    72ec:	e8 bf ae ff ff       	call   21b0 <omTensorDestroy@plt>
    72f1:	4c 89 ff             	mov    %r15,%rdi
    72f4:	e8 17 ae ff ff       	call   2110 <free@plt>
    72f9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    72fe:	e8 cd ad ff ff       	call   20d0 <malloc@plt>
    7303:	48 89 c3             	mov    %rax,%rbx
    7306:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    730a:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    730f:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    7313:	4c 89 7c 24 20       	mov    %r15,0x20(%rsp)
    7318:	bf 04 00 00 00       	mov    $0x4,%edi
    731d:	e8 7e ae ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    7322:	49 89 c4             	mov    %rax,%r12
    7325:	48 89 c7             	mov    %rax,%rdi
    7328:	31 f6                	xor    %esi,%esi
    732a:	48 89 da             	mov    %rbx,%rdx
    732d:	4c 89 f9             	mov    %r15,%rcx
    7330:	e8 bb ad ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    7335:	be 01 00 00 00       	mov    $0x1,%esi
    733a:	4c 89 e3             	mov    %r12,%rbx
    733d:	4c 89 64 24 40       	mov    %r12,0x40(%rsp)
    7342:	4c 89 e7             	mov    %r12,%rdi
    7345:	e8 56 ad ff ff       	call   20a0 <omTensorSetDataType@plt>
    734a:	4c 89 e7             	mov    %r12,%rdi
    734d:	e8 7e ae ff ff       	call   21d0 <omTensorGetShape@plt>
    7352:	49 89 c4             	mov    %rax,%r12
    7355:	48 89 df             	mov    %rbx,%rdi
    7358:	e8 13 ad ff ff       	call   2070 <omTensorGetStrides@plt>
    735d:	4d 89 34 24          	mov    %r14,(%r12)
    7361:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    7368:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    736f:	00 00 
    7371:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    7378:	00 
    7379:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    7380:	00 00 
    7382:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    7389:	00 
    738a:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    7391:	00 00 
    7393:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    739a:	00 
    739b:	bf 04 00 00 00       	mov    $0x4,%edi
    73a0:	e8 fb ad ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    73a5:	48 89 c3             	mov    %rax,%rbx
    73a8:	48 89 c7             	mov    %rax,%rdi
    73ab:	31 f6                	xor    %esi,%esi
    73ad:	48 8b 14 24          	mov    (%rsp),%rdx
    73b1:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    73b6:	e8 35 ad ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    73bb:	be 01 00 00 00       	mov    $0x1,%esi
    73c0:	48 89 df             	mov    %rbx,%rdi
    73c3:	e8 d8 ac ff ff       	call   20a0 <omTensorSetDataType@plt>
    73c8:	48 89 df             	mov    %rbx,%rdi
    73cb:	e8 00 ae ff ff       	call   21d0 <omTensorGetShape@plt>
    73d0:	49 89 c7             	mov    %rax,%r15
    73d3:	48 89 df             	mov    %rbx,%rdi
    73d6:	e8 95 ac ff ff       	call   2070 <omTensorGetStrides@plt>
    73db:	4d 89 37             	mov    %r14,(%r15)
    73de:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    73e5:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    73ec:	00 
    73ed:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    73f4:	00 
    73f5:	49 c7 47 10 06 00 00 	movq   $0x6,0x10(%r15)
    73fc:	00 
    73fd:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    7404:	00 
    7405:	49 c7 47 18 06 00 00 	movq   $0x6,0x18(%r15)
    740c:	00 
    740d:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    7414:	00 
    7415:	4c 8d 25 94 ea 00 00 	lea    0xea94(%rip),%r12        # 15eb0 <constant_57_mobilenet>
    741c:	bf 04 00 00 00       	mov    $0x4,%edi
    7421:	e8 7a ad ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    7426:	49 89 c7             	mov    %rax,%r15
    7429:	48 89 c7             	mov    %rax,%rdi
    742c:	31 f6                	xor    %esi,%esi
    742e:	4c 89 e2             	mov    %r12,%rdx
    7431:	4c 89 e1             	mov    %r12,%rcx
    7434:	e8 b7 ac ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    7439:	be 01 00 00 00       	mov    $0x1,%esi
    743e:	4c 89 ff             	mov    %r15,%rdi
    7441:	e8 5a ac ff ff       	call   20a0 <omTensorSetDataType@plt>
    7446:	4c 89 ff             	mov    %r15,%rdi
    7449:	e8 82 ad ff ff       	call   21d0 <omTensorGetShape@plt>
    744e:	49 89 c4             	mov    %rax,%r12
    7451:	4c 89 ff             	mov    %r15,%rdi
    7454:	e8 17 ac ff ff       	call   2070 <omTensorGetStrides@plt>
    7459:	49 c7 04 24 80 00 00 	movq   $0x80,(%r12)
    7460:	00 
    7461:	48 c7 00 09 00 00 00 	movq   $0x9,(%rax)
    7468:	49 c7 44 24 08 01 00 	movq   $0x1,0x8(%r12)
    746f:	00 00 
    7471:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    7478:	00 
    7479:	49 c7 44 24 10 03 00 	movq   $0x3,0x10(%r12)
    7480:	00 00 
    7482:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    7489:	00 
    748a:	49 c7 44 24 18 03 00 	movq   $0x3,0x18(%r12)
    7491:	00 00 
    7493:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    749a:	00 
    749b:	4c 8d 2d 2e 29 01 00 	lea    0x1292e(%rip),%r13        # 19dd0 <constant_49_mobilenet>
    74a2:	bf 01 00 00 00       	mov    $0x1,%edi
    74a7:	e8 f4 ac ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    74ac:	49 89 c4             	mov    %rax,%r12
    74af:	48 89 c7             	mov    %rax,%rdi
    74b2:	31 f6                	xor    %esi,%esi
    74b4:	4c 89 ea             	mov    %r13,%rdx
    74b7:	4c 89 e9             	mov    %r13,%rcx
    74ba:	e8 31 ac ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    74bf:	be 01 00 00 00       	mov    $0x1,%esi
    74c4:	4c 89 e7             	mov    %r12,%rdi
    74c7:	e8 d4 ab ff ff       	call   20a0 <omTensorSetDataType@plt>
    74cc:	4c 89 e7             	mov    %r12,%rdi
    74cf:	e8 fc ac ff ff       	call   21d0 <omTensorGetShape@plt>
    74d4:	49 89 c5             	mov    %rax,%r13
    74d7:	4c 89 e7             	mov    %r12,%rdi
    74da:	e8 91 ab ff ff       	call   2070 <omTensorGetStrides@plt>
    74df:	49 c7 45 00 80 00 00 	movq   $0x80,0x0(%r13)
    74e6:	00 
    74e7:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    74ee:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    74f3:	4c 89 ef             	mov    %r13,%rdi
    74f6:	48 89 de             	mov    %rbx,%rsi
    74f9:	4c 89 fa             	mov    %r15,%rdx
    74fc:	4c 89 e1             	mov    %r12,%rcx
    74ff:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    7505:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    750b:	6a 01                	push   $0x1
    750d:	6a 01                	push   $0x1
    750f:	6a 01                	push   $0x1
    7511:	6a 01                	push   $0x1
    7513:	e8 48 ac ff ff       	call   2160 <Conv2D_bias@plt>
    7518:	48 83 c4 20          	add    $0x20,%rsp
    751c:	4c 89 ef             	mov    %r13,%rdi
    751f:	e8 8c ac ff ff       	call   21b0 <omTensorDestroy@plt>
    7524:	48 89 df             	mov    %rbx,%rdi
    7527:	e8 84 ac ff ff       	call   21b0 <omTensorDestroy@plt>
    752c:	4c 89 ff             	mov    %r15,%rdi
    752f:	e8 7c ac ff ff       	call   21b0 <omTensorDestroy@plt>
    7534:	4c 89 e7             	mov    %r12,%rdi
    7537:	e8 74 ac ff ff       	call   21b0 <omTensorDestroy@plt>
    753c:	48 8b 3c 24          	mov    (%rsp),%rdi
    7540:	e8 cb ab ff ff       	call   2110 <free@plt>
    7545:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    754a:	e8 81 ab ff ff       	call   20d0 <malloc@plt>
    754f:	49 89 c7             	mov    %rax,%r15
    7552:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    7556:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    755b:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    755f:	4c 89 24 24          	mov    %r12,(%rsp)
    7563:	bf 04 00 00 00       	mov    $0x4,%edi
    7568:	e8 33 ac ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    756d:	48 89 c3             	mov    %rax,%rbx
    7570:	48 89 c7             	mov    %rax,%rdi
    7573:	31 f6                	xor    %esi,%esi
    7575:	4c 89 fa             	mov    %r15,%rdx
    7578:	4c 89 e1             	mov    %r12,%rcx
    757b:	e8 70 ab ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    7580:	be 01 00 00 00       	mov    $0x1,%esi
    7585:	48 89 df             	mov    %rbx,%rdi
    7588:	e8 13 ab ff ff       	call   20a0 <omTensorSetDataType@plt>
    758d:	48 89 df             	mov    %rbx,%rdi
    7590:	e8 3b ac ff ff       	call   21d0 <omTensorGetShape@plt>
    7595:	49 89 c4             	mov    %rax,%r12
    7598:	48 89 df             	mov    %rbx,%rdi
    759b:	e8 d0 aa ff ff       	call   2070 <omTensorGetStrides@plt>
    75a0:	4d 89 34 24          	mov    %r14,(%r12)
    75a4:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    75ab:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    75b2:	00 00 
    75b4:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    75bb:	00 
    75bc:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    75c3:	00 00 
    75c5:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    75cc:	00 
    75cd:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    75d4:	00 00 
    75d6:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    75dd:	00 
    75de:	bf 04 00 00 00       	mov    $0x4,%edi
    75e3:	e8 b8 ab ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    75e8:	49 89 c4             	mov    %rax,%r12
    75eb:	48 89 c7             	mov    %rax,%rdi
    75ee:	31 f6                	xor    %esi,%esi
    75f0:	4c 8b 7c 24 30       	mov    0x30(%rsp),%r15
    75f5:	4c 89 fa             	mov    %r15,%rdx
    75f8:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    75fd:	e8 ee aa ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    7602:	be 01 00 00 00       	mov    $0x1,%esi
    7607:	4c 89 e7             	mov    %r12,%rdi
    760a:	e8 91 aa ff ff       	call   20a0 <omTensorSetDataType@plt>
    760f:	4c 89 e7             	mov    %r12,%rdi
    7612:	e8 b9 ab ff ff       	call   21d0 <omTensorGetShape@plt>
    7617:	49 89 c5             	mov    %rax,%r13
    761a:	4c 89 e7             	mov    %r12,%rdi
    761d:	e8 4e aa ff ff       	call   2070 <omTensorGetStrides@plt>
    7622:	4d 89 75 00          	mov    %r14,0x0(%r13)
    7626:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    762d:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    7634:	00 
    7635:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    763c:	00 
    763d:	49 c7 45 10 06 00 00 	movq   $0x6,0x10(%r13)
    7644:	00 
    7645:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    764c:	00 
    764d:	49 c7 45 18 06 00 00 	movq   $0x6,0x18(%r13)
    7654:	00 
    7655:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    765c:	00 
    765d:	48 89 df             	mov    %rbx,%rdi
    7660:	4c 89 e6             	mov    %r12,%rsi
    7663:	e8 88 ab ff ff       	call   21f0 <Relu@plt>
    7668:	48 89 df             	mov    %rbx,%rdi
    766b:	e8 40 ab ff ff       	call   21b0 <omTensorDestroy@plt>
    7670:	4c 89 e7             	mov    %r12,%rdi
    7673:	e8 38 ab ff ff       	call   21b0 <omTensorDestroy@plt>
    7678:	4c 89 ff             	mov    %r15,%rdi
    767b:	e8 90 aa ff ff       	call   2110 <free@plt>
    7680:	bf 18 00 00 00       	mov    $0x18,%edi
    7685:	e8 46 aa ff ff       	call   20d0 <malloc@plt>
    768a:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    768f:	48 83 c0 0f          	add    $0xf,%rax
    7693:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    7697:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    769c:	4c 89 30             	mov    %r14,(%rax)
    769f:	bf 28 00 00 00       	mov    $0x28,%edi
    76a4:	e8 27 aa ff ff       	call   20d0 <malloc@plt>
    76a9:	48 8d 48 0f          	lea    0xf(%rax),%rcx
    76ad:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
    76b1:	4c 89 31             	mov    %r14,(%rcx)
    76b4:	48 c7 41 08 80 00 00 	movq   $0x80,0x8(%rcx)
    76bb:	00 
    76bc:	48 c7 41 10 ff ff ff 	movq   $0xffffffffffffffff,0x10(%rcx)
    76c3:	ff 
    76c4:	48 89 c7             	mov    %rax,%rdi
    76c7:	e8 44 aa ff ff       	call   2110 <free@plt>
    76cc:	48 89 e9             	mov    %rbp,%rcx
    76cf:	48 c1 e1 07          	shl    $0x7,%rcx
    76d3:	48 8b 5c 24 60       	mov    0x60(%rsp),%rbx
    76d8:	48 89 d8             	mov    %rbx,%rax
    76db:	48 0f af c1          	imul   %rcx,%rax
    76df:	49 89 cf             	mov    %rcx,%r15
    76e2:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
    76e7:	48 8d 3c 85 10 00 00 	lea    0x10(,%rax,4),%rdi
    76ee:	00 
    76ef:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
    76f4:	e8 d7 a9 ff ff       	call   20d0 <malloc@plt>
    76f9:	49 89 c4             	mov    %rax,%r12
    76fc:	4c 8d 68 0f          	lea    0xf(%rax),%r13
    7700:	49 83 e5 f0          	and    $0xfffffffffffffff0,%r13
    7704:	bf 03 00 00 00       	mov    $0x3,%edi
    7709:	e8 92 aa ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    770e:	49 89 c6             	mov    %rax,%r14
    7711:	48 89 c7             	mov    %rax,%rdi
    7714:	31 f6                	xor    %esi,%esi
    7716:	4c 89 64 24 70       	mov    %r12,0x70(%rsp)
    771b:	4c 89 e2             	mov    %r12,%rdx
    771e:	4c 89 6c 24 08       	mov    %r13,0x8(%rsp)
    7723:	4c 89 e9             	mov    %r13,%rcx
    7726:	e8 c5 a9 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    772b:	be 01 00 00 00       	mov    $0x1,%esi
    7730:	4c 89 f7             	mov    %r14,%rdi
    7733:	e8 68 a9 ff ff       	call   20a0 <omTensorSetDataType@plt>
    7738:	4c 89 f7             	mov    %r14,%rdi
    773b:	e8 90 aa ff ff       	call   21d0 <omTensorGetShape@plt>
    7740:	49 89 c4             	mov    %rax,%r12
    7743:	4c 89 f7             	mov    %r14,%rdi
    7746:	e8 25 a9 ff ff       	call   2070 <omTensorGetStrides@plt>
    774b:	49 89 1c 24          	mov    %rbx,(%r12)
    774f:	4c 89 38             	mov    %r15,(%rax)
    7752:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    7759:	00 00 
    775b:	48 89 68 08          	mov    %rbp,0x8(%rax)
    775f:	49 89 6c 24 10       	mov    %rbp,0x10(%r12)
    7764:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    776b:	00 
    776c:	bf 02 00 00 00       	mov    $0x2,%edi
    7771:	e8 2a aa ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    7776:	49 89 c5             	mov    %rax,%r13
    7779:	48 8d 0d 90 4f 0b 00 	lea    0xb4f90(%rip),%rcx        # bc710 <constant_17_mobilenet>
    7780:	48 89 c7             	mov    %rax,%rdi
    7783:	31 f6                	xor    %esi,%esi
    7785:	48 89 ca             	mov    %rcx,%rdx
    7788:	e8 63 a9 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    778d:	be 01 00 00 00       	mov    $0x1,%esi
    7792:	4c 89 ef             	mov    %r13,%rdi
    7795:	e8 06 a9 ff ff       	call   20a0 <omTensorSetDataType@plt>
    779a:	4c 89 ef             	mov    %r13,%rdi
    779d:	e8 2e aa ff ff       	call   21d0 <omTensorGetShape@plt>
    77a2:	49 89 c4             	mov    %rax,%r12
    77a5:	4c 89 ef             	mov    %r13,%rdi
    77a8:	e8 c3 a8 ff ff       	call   2070 <omTensorGetStrides@plt>
    77ad:	49 c7 04 24 80 00 00 	movq   $0x80,(%r12)
    77b4:	00 
    77b5:	48 c7 00 80 00 00 00 	movq   $0x80,(%rax)
    77bc:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    77c3:	00 00 
    77c5:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    77cc:	00 
    77cd:	bf 03 00 00 00       	mov    $0x3,%edi
    77d2:	e8 c9 a9 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    77d7:	49 89 c4             	mov    %rax,%r12
    77da:	48 89 c7             	mov    %rax,%rdi
    77dd:	31 f6                	xor    %esi,%esi
    77df:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    77e4:	48 8b 0c 24          	mov    (%rsp),%rcx
    77e8:	e8 03 a9 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    77ed:	be 01 00 00 00       	mov    $0x1,%esi
    77f2:	4c 89 e7             	mov    %r12,%rdi
    77f5:	e8 a6 a8 ff ff       	call   20a0 <omTensorSetDataType@plt>
    77fa:	4c 89 e7             	mov    %r12,%rdi
    77fd:	e8 ce a9 ff ff       	call   21d0 <omTensorGetShape@plt>
    7802:	49 89 c7             	mov    %rax,%r15
    7805:	4c 89 e7             	mov    %r12,%rdi
    7808:	e8 63 a8 ff ff       	call   2070 <omTensorGetStrides@plt>
    780d:	49 89 1f             	mov    %rbx,(%r15)
    7810:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    7815:	48 89 08             	mov    %rcx,(%rax)
    7818:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    781f:	00 
    7820:	48 89 68 08          	mov    %rbp,0x8(%rax)
    7824:	49 89 6f 10          	mov    %rbp,0x10(%r15)
    7828:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    782f:	00 
    7830:	4c 89 f7             	mov    %r14,%rdi
    7833:	4c 89 ee             	mov    %r13,%rsi
    7836:	4c 89 e2             	mov    %r12,%rdx
    7839:	e8 12 a8 ff ff       	call   2050 <MatMul@plt>
    783e:	4c 89 f7             	mov    %r14,%rdi
    7841:	e8 6a a9 ff ff       	call   21b0 <omTensorDestroy@plt>
    7846:	4c 89 ef             	mov    %r13,%rdi
    7849:	e8 62 a9 ff ff       	call   21b0 <omTensorDestroy@plt>
    784e:	4c 89 e7             	mov    %r12,%rdi
    7851:	e8 5a a9 ff ff       	call   21b0 <omTensorDestroy@plt>
    7856:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    785b:	e8 b0 a8 ff ff       	call   2110 <free@plt>
    7860:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    7865:	e8 66 a8 ff ff       	call   20d0 <malloc@plt>
    786a:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    786f:	48 83 c0 0f          	add    $0xf,%rax
    7873:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    7877:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    787c:	48 85 db             	test   %rbx,%rbx
    787f:	0f 8e 58 01 00 00    	jle    79dd <main_graph_mobilenet+0x570d>
    7885:	48 89 e8             	mov    %rbp,%rax
    7888:	48 c1 e0 09          	shl    $0x9,%rax
    788c:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    7891:	48 8d 0c ad 00 00 00 	lea    0x0(,%rbp,4),%rcx
    7898:	00 
    7899:	48 89 eb             	mov    %rbp,%rbx
    789c:	48 83 e3 f8          	and    $0xfffffffffffffff8,%rbx
    78a0:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    78a5:	49 8d 45 10          	lea    0x10(%r13),%rax
    78a9:	48 89 04 24          	mov    %rax,(%rsp)
    78ad:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    78b2:	48 8d 50 10          	lea    0x10(%rax),%rdx
    78b6:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
    78bb:	45 31 c0             	xor    %r8d,%r8d
    78be:	4c 8d 0d 4b 47 01 00 	lea    0x1474b(%rip),%r9        # 1c010 <constant_30_mobilenet>
    78c5:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    78ca:	eb 28                	jmp    78f4 <main_graph_mobilenet+0x5624>
    78cc:	0f 1f 40 00          	nopl   0x0(%rax)
    78d0:	49 ff c0             	inc    %r8
    78d3:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    78d8:	48 01 04 24          	add    %rax,(%rsp)
    78dc:	48 01 44 24 20       	add    %rax,0x20(%rsp)
    78e1:	49 01 c5             	add    %rax,%r13
    78e4:	48 01 44 24 30       	add    %rax,0x30(%rsp)
    78e9:	4c 3b 44 24 60       	cmp    0x60(%rsp),%r8
    78ee:	0f 84 e9 00 00 00    	je     79dd <main_graph_mobilenet+0x570d>
    78f4:	48 85 ed             	test   %rbp,%rbp
    78f7:	7e d7                	jle    78d0 <main_graph_mobilenet+0x5600>
    78f9:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
    78fe:	4d 0f af f0          	imul   %r8,%r14
    7902:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    7907:	4e 8d 3c 30          	lea    (%rax,%r14,1),%r15
    790b:	4c 03 74 24 08       	add    0x8(%rsp),%r14
    7910:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
    7915:	4d 89 eb             	mov    %r13,%r11
    7918:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    791d:	48 8b 34 24          	mov    (%rsp),%rsi
    7921:	31 c0                	xor    %eax,%eax
    7923:	eb 22                	jmp    7947 <main_graph_mobilenet+0x5677>
    7925:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    792c:	00 00 00 
    792f:	90                   	nop
    7930:	48 ff c0             	inc    %rax
    7933:	48 01 ce             	add    %rcx,%rsi
    7936:	48 01 cf             	add    %rcx,%rdi
    7939:	49 01 cb             	add    %rcx,%r11
    793c:	49 01 cc             	add    %rcx,%r12
    793f:	48 3d 80 00 00 00    	cmp    $0x80,%rax
    7945:	74 89                	je     78d0 <main_graph_mobilenet+0x5600>
    7947:	48 83 fd 08          	cmp    $0x8,%rbp
    794b:	f3 41 0f 10 04 81    	movss  (%r9,%rax,4),%xmm0
    7951:	73 0d                	jae    7960 <main_graph_mobilenet+0x5690>
    7953:	31 d2                	xor    %edx,%edx
    7955:	eb 69                	jmp    79c0 <main_graph_mobilenet+0x56f0>
    7957:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    795e:	00 00 
    7960:	48 89 ca             	mov    %rcx,%rdx
    7963:	48 0f af d0          	imul   %rax,%rdx
    7967:	4d 8d 14 16          	lea    (%r14,%rdx,1),%r10
    796b:	4c 01 fa             	add    %r15,%rdx
    796e:	4c 29 d2             	sub    %r10,%rdx
    7971:	48 83 fa 20          	cmp    $0x20,%rdx
    7975:	73 09                	jae    7980 <main_graph_mobilenet+0x56b0>
    7977:	31 d2                	xor    %edx,%edx
    7979:	eb 45                	jmp    79c0 <main_graph_mobilenet+0x56f0>
    797b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    7980:	0f 28 c8             	movaps %xmm0,%xmm1
    7983:	0f c6 c8 00          	shufps $0x0,%xmm0,%xmm1
    7987:	31 d2                	xor    %edx,%edx
    7989:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    7990:	0f 10 54 97 f0       	movups -0x10(%rdi,%rdx,4),%xmm2
    7995:	0f 10 1c 97          	movups (%rdi,%rdx,4),%xmm3
    7999:	0f 58 d1             	addps  %xmm1,%xmm2
    799c:	0f 58 d9             	addps  %xmm1,%xmm3
    799f:	0f 11 54 96 f0       	movups %xmm2,-0x10(%rsi,%rdx,4)
    79a4:	0f 11 1c 96          	movups %xmm3,(%rsi,%rdx,4)
    79a8:	48 83 c2 08          	add    $0x8,%rdx
    79ac:	48 39 d3             	cmp    %rdx,%rbx
    79af:	75 df                	jne    7990 <main_graph_mobilenet+0x56c0>
    79b1:	48 39 dd             	cmp    %rbx,%rbp
    79b4:	48 89 da             	mov    %rbx,%rdx
    79b7:	0f 84 73 ff ff ff    	je     7930 <main_graph_mobilenet+0x5660>
    79bd:	0f 1f 00             	nopl   (%rax)
    79c0:	f3 41 0f 10 0c 94    	movss  (%r12,%rdx,4),%xmm1
    79c6:	f3 0f 58 c8          	addss  %xmm0,%xmm1
    79ca:	f3 41 0f 11 0c 93    	movss  %xmm1,(%r11,%rdx,4)
    79d0:	48 ff c2             	inc    %rdx
    79d3:	48 39 d5             	cmp    %rdx,%rbp
    79d6:	75 e8                	jne    79c0 <main_graph_mobilenet+0x56f0>
    79d8:	e9 53 ff ff ff       	jmp    7930 <main_graph_mobilenet+0x5660>
    79dd:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    79e2:	e8 29 a7 ff ff       	call   2110 <free@plt>
    79e7:	bf 30 00 00 00       	mov    $0x30,%edi
    79ec:	e8 df a6 ff ff       	call   20d0 <malloc@plt>
    79f1:	49 89 c7             	mov    %rax,%r15
    79f4:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    79f8:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    79fc:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    7a01:	48 8b 00             	mov    (%rax),%rax
    7a04:	49 89 04 24          	mov    %rax,(%r12)
    7a08:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    7a0d:	e8 fe a6 ff ff       	call   2110 <free@plt>
    7a12:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    7a17:	48 0f af e8          	imul   %rax,%rbp
    7a1b:	48 c1 e5 07          	shl    $0x7,%rbp
    7a1f:	49 8b 0c 24          	mov    (%r12),%rcx
    7a23:	48 85 c9             	test   %rcx,%rcx
    7a26:	48 0f 44 c8          	cmove  %rax,%rcx
    7a2a:	48 89 c8             	mov    %rcx,%rax
    7a2d:	48 c1 e0 09          	shl    $0x9,%rax
    7a31:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    7a35:	48 8d 34 c0          	lea    (%rax,%rax,8),%rsi
    7a39:	b8 00 12 00 00       	mov    $0x1200,%eax
    7a3e:	48 0f 44 f0          	cmove  %rax,%rsi
    7a42:	48 89 f0             	mov    %rsi,%rax
    7a45:	48 f7 d0             	not    %rax
    7a48:	48 c1 f8 3f          	sar    $0x3f,%rax
    7a4c:	48 83 c8 01          	or     $0x1,%rax
    7a50:	48 29 e8             	sub    %rbp,%rax
    7a53:	48 99                	cqto   
    7a55:	48 f7 fe             	idiv   %rsi
    7a58:	49 89 c6             	mov    %rax,%r14
    7a5b:	49 f7 d6             	not    %r14
    7a5e:	48 89 e8             	mov    %rbp,%rax
    7a61:	48 99                	cqto   
    7a63:	48 f7 fe             	idiv   %rsi
    7a66:	48 85 f6             	test   %rsi,%rsi
    7a69:	48 89 c2             	mov    %rax,%rdx
    7a6c:	49 0f 48 d6          	cmovs  %r14,%rdx
    7a70:	48 85 ed             	test   %rbp,%rbp
    7a73:	48 0f 4e d0          	cmovle %rax,%rdx
    7a77:	48 85 f6             	test   %rsi,%rsi
    7a7a:	4c 0f 4e f2          	cmovle %rdx,%r14
    7a7e:	48 85 ed             	test   %rbp,%rbp
    7a81:	4c 0f 49 f2          	cmovns %rdx,%r14
    7a85:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    7a89:	4c 0f 45 f1          	cmovne %rcx,%r14
    7a8d:	4c 89 f0             	mov    %r14,%rax
    7a90:	48 c1 e0 09          	shl    $0x9,%rax
    7a94:	48 8d 0c c0          	lea    (%rax,%rax,8),%rcx
    7a98:	4c 89 f6             	mov    %r14,%rsi
    7a9b:	48 c1 e6 07          	shl    $0x7,%rsi
    7a9f:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    7aa3:	b8 80 00 00 00       	mov    $0x80,%eax
    7aa8:	48 0f 44 f0          	cmove  %rax,%rsi
    7aac:	48 89 f0             	mov    %rsi,%rax
    7aaf:	48 f7 d0             	not    %rax
    7ab2:	48 c1 f8 3f          	sar    $0x3f,%rax
    7ab6:	48 83 c8 01          	or     $0x1,%rax
    7aba:	48 29 c8             	sub    %rcx,%rax
    7abd:	48 99                	cqto   
    7abf:	48 f7 fe             	idiv   %rsi
    7ac2:	48 89 c5             	mov    %rax,%rbp
    7ac5:	48 f7 d5             	not    %rbp
    7ac8:	48 89 c8             	mov    %rcx,%rax
    7acb:	48 99                	cqto   
    7acd:	48 f7 fe             	idiv   %rsi
    7ad0:	48 85 f6             	test   %rsi,%rsi
    7ad3:	48 89 c2             	mov    %rax,%rdx
    7ad6:	48 0f 48 d5          	cmovs  %rbp,%rdx
    7ada:	48 85 c9             	test   %rcx,%rcx
    7add:	48 0f 4e d0          	cmovle %rax,%rdx
    7ae1:	48 85 f6             	test   %rsi,%rsi
    7ae4:	48 0f 4e ea          	cmovle %rdx,%rbp
    7ae8:	48 85 c9             	test   %rcx,%rcx
    7aeb:	48 0f 49 ea          	cmovns %rdx,%rbp
    7aef:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    7af3:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    7afa:	00 00 
    7afc:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    7b03:	00 00 
    7b05:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    7b0c:	00 00 
    7b0e:	4c 89 f0             	mov    %r14,%rax
    7b11:	48 0f 44 c5          	cmove  %rbp,%rax
    7b15:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    7b1a:	4c 89 ff             	mov    %r15,%rdi
    7b1d:	e8 ee a5 ff ff       	call   2110 <free@plt>
    7b22:	4c 89 f0             	mov    %r14,%rax
    7b25:	48 c1 e0 0b          	shl    $0xb,%rax
    7b29:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    7b2e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    7b33:	e8 98 a5 ff ff       	call   20d0 <malloc@plt>
    7b38:	49 89 c7             	mov    %rax,%r15
    7b3b:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    7b3f:	48 89 04 24          	mov    %rax,(%rsp)
    7b43:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    7b47:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    7b4c:	bf 04 00 00 00       	mov    $0x4,%edi
    7b51:	e8 4a a6 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    7b56:	48 89 c3             	mov    %rax,%rbx
    7b59:	48 89 c7             	mov    %rax,%rdi
    7b5c:	31 f6                	xor    %esi,%esi
    7b5e:	4c 89 fa             	mov    %r15,%rdx
    7b61:	4c 89 e1             	mov    %r12,%rcx
    7b64:	e8 87 a5 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    7b69:	be 01 00 00 00       	mov    $0x1,%esi
    7b6e:	48 89 df             	mov    %rbx,%rdi
    7b71:	e8 2a a5 ff ff       	call   20a0 <omTensorSetDataType@plt>
    7b76:	48 89 df             	mov    %rbx,%rdi
    7b79:	e8 52 a6 ff ff       	call   21d0 <omTensorGetShape@plt>
    7b7e:	49 89 c4             	mov    %rax,%r12
    7b81:	48 89 df             	mov    %rbx,%rdi
    7b84:	e8 e7 a4 ff ff       	call   2070 <omTensorGetStrides@plt>
    7b89:	4d 89 34 24          	mov    %r14,(%r12)
    7b8d:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    7b94:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    7b9b:	00 00 
    7b9d:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    7ba4:	00 
    7ba5:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    7bac:	00 00 
    7bae:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    7bb5:	00 
    7bb6:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    7bbd:	00 00 
    7bbf:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    7bc6:	00 
    7bc7:	bf 04 00 00 00       	mov    $0x4,%edi
    7bcc:	e8 cf a5 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    7bd1:	49 89 c4             	mov    %rax,%r12
    7bd4:	48 89 c7             	mov    %rax,%rdi
    7bd7:	31 f6                	xor    %esi,%esi
    7bd9:	4c 8b 7c 24 50       	mov    0x50(%rsp),%r15
    7bde:	4c 89 fa             	mov    %r15,%rdx
    7be1:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    7be6:	e8 05 a5 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    7beb:	be 01 00 00 00       	mov    $0x1,%esi
    7bf0:	4c 89 e7             	mov    %r12,%rdi
    7bf3:	e8 a8 a4 ff ff       	call   20a0 <omTensorSetDataType@plt>
    7bf8:	4c 89 e7             	mov    %r12,%rdi
    7bfb:	e8 d0 a5 ff ff       	call   21d0 <omTensorGetShape@plt>
    7c00:	49 89 c5             	mov    %rax,%r13
    7c03:	4c 89 e7             	mov    %r12,%rdi
    7c06:	e8 65 a4 ff ff       	call   2070 <omTensorGetStrides@plt>
    7c0b:	4d 89 75 00          	mov    %r14,0x0(%r13)
    7c0f:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    7c16:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    7c1d:	00 
    7c1e:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    7c25:	00 
    7c26:	49 c7 45 10 06 00 00 	movq   $0x6,0x10(%r13)
    7c2d:	00 
    7c2e:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    7c35:	00 
    7c36:	49 c7 45 18 06 00 00 	movq   $0x6,0x18(%r13)
    7c3d:	00 
    7c3e:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    7c45:	00 
    7c46:	48 89 df             	mov    %rbx,%rdi
    7c49:	4c 89 e6             	mov    %r12,%rsi
    7c4c:	e8 9f a5 ff ff       	call   21f0 <Relu@plt>
    7c51:	48 89 df             	mov    %rbx,%rdi
    7c54:	e8 57 a5 ff ff       	call   21b0 <omTensorDestroy@plt>
    7c59:	4c 89 e7             	mov    %r12,%rdi
    7c5c:	e8 4f a5 ff ff       	call   21b0 <omTensorDestroy@plt>
    7c61:	4c 89 ff             	mov    %r15,%rdi
    7c64:	e8 a7 a4 ff ff       	call   2110 <free@plt>
    7c69:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    7c6e:	e8 5d a4 ff ff       	call   20d0 <malloc@plt>
    7c73:	48 89 c3             	mov    %rax,%rbx
    7c76:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    7c7a:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    7c7f:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    7c83:	4c 89 7c 24 20       	mov    %r15,0x20(%rsp)
    7c88:	bf 04 00 00 00       	mov    $0x4,%edi
    7c8d:	e8 0e a5 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    7c92:	49 89 c4             	mov    %rax,%r12
    7c95:	48 89 c7             	mov    %rax,%rdi
    7c98:	31 f6                	xor    %esi,%esi
    7c9a:	48 89 da             	mov    %rbx,%rdx
    7c9d:	4c 89 f9             	mov    %r15,%rcx
    7ca0:	e8 4b a4 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    7ca5:	be 01 00 00 00       	mov    $0x1,%esi
    7caa:	4c 89 e3             	mov    %r12,%rbx
    7cad:	4c 89 64 24 40       	mov    %r12,0x40(%rsp)
    7cb2:	4c 89 e7             	mov    %r12,%rdi
    7cb5:	e8 e6 a3 ff ff       	call   20a0 <omTensorSetDataType@plt>
    7cba:	4c 89 e7             	mov    %r12,%rdi
    7cbd:	e8 0e a5 ff ff       	call   21d0 <omTensorGetShape@plt>
    7cc2:	49 89 c4             	mov    %rax,%r12
    7cc5:	48 89 df             	mov    %rbx,%rdi
    7cc8:	e8 a3 a3 ff ff       	call   2070 <omTensorGetStrides@plt>
    7ccd:	4d 89 34 24          	mov    %r14,(%r12)
    7cd1:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    7cd8:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    7cdf:	00 00 
    7ce1:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    7ce8:	00 
    7ce9:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    7cf0:	00 00 
    7cf2:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    7cf9:	00 
    7cfa:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    7d01:	00 00 
    7d03:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    7d0a:	00 
    7d0b:	bf 04 00 00 00       	mov    $0x4,%edi
    7d10:	e8 8b a4 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    7d15:	48 89 c3             	mov    %rax,%rbx
    7d18:	48 89 c7             	mov    %rax,%rdi
    7d1b:	31 f6                	xor    %esi,%esi
    7d1d:	48 8b 14 24          	mov    (%rsp),%rdx
    7d21:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    7d26:	e8 c5 a3 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    7d2b:	be 01 00 00 00       	mov    $0x1,%esi
    7d30:	48 89 df             	mov    %rbx,%rdi
    7d33:	e8 68 a3 ff ff       	call   20a0 <omTensorSetDataType@plt>
    7d38:	48 89 df             	mov    %rbx,%rdi
    7d3b:	e8 90 a4 ff ff       	call   21d0 <omTensorGetShape@plt>
    7d40:	49 89 c7             	mov    %rax,%r15
    7d43:	48 89 df             	mov    %rbx,%rdi
    7d46:	e8 25 a3 ff ff       	call   2070 <omTensorGetStrides@plt>
    7d4b:	4d 89 37             	mov    %r14,(%r15)
    7d4e:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    7d55:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    7d5c:	00 
    7d5d:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    7d64:	00 
    7d65:	49 c7 47 10 06 00 00 	movq   $0x6,0x10(%r15)
    7d6c:	00 
    7d6d:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    7d74:	00 
    7d75:	49 c7 47 18 06 00 00 	movq   $0x6,0x18(%r15)
    7d7c:	00 
    7d7d:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    7d84:	00 
    7d85:	4c 8d 25 c4 cb 00 00 	lea    0xcbc4(%rip),%r12        # 14950 <constant_59_mobilenet>
    7d8c:	bf 04 00 00 00       	mov    $0x4,%edi
    7d91:	e8 0a a4 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    7d96:	49 89 c7             	mov    %rax,%r15
    7d99:	48 89 c7             	mov    %rax,%rdi
    7d9c:	31 f6                	xor    %esi,%esi
    7d9e:	4c 89 e2             	mov    %r12,%rdx
    7da1:	4c 89 e1             	mov    %r12,%rcx
    7da4:	e8 47 a3 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    7da9:	be 01 00 00 00       	mov    $0x1,%esi
    7dae:	4c 89 ff             	mov    %r15,%rdi
    7db1:	e8 ea a2 ff ff       	call   20a0 <omTensorSetDataType@plt>
    7db6:	4c 89 ff             	mov    %r15,%rdi
    7db9:	e8 12 a4 ff ff       	call   21d0 <omTensorGetShape@plt>
    7dbe:	49 89 c4             	mov    %rax,%r12
    7dc1:	4c 89 ff             	mov    %r15,%rdi
    7dc4:	e8 a7 a2 ff ff       	call   2070 <omTensorGetStrides@plt>
    7dc9:	49 c7 04 24 80 00 00 	movq   $0x80,(%r12)
    7dd0:	00 
    7dd1:	48 c7 00 09 00 00 00 	movq   $0x9,(%rax)
    7dd8:	49 c7 44 24 08 01 00 	movq   $0x1,0x8(%r12)
    7ddf:	00 00 
    7de1:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    7de8:	00 
    7de9:	49 c7 44 24 10 03 00 	movq   $0x3,0x10(%r12)
    7df0:	00 00 
    7df2:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    7df9:	00 
    7dfa:	49 c7 44 24 18 03 00 	movq   $0x3,0x18(%r12)
    7e01:	00 00 
    7e03:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    7e0a:	00 
    7e0b:	4c 8d 2d be 21 01 00 	lea    0x121be(%rip),%r13        # 19fd0 <constant_48_mobilenet>
    7e12:	bf 01 00 00 00       	mov    $0x1,%edi
    7e17:	e8 84 a3 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    7e1c:	49 89 c4             	mov    %rax,%r12
    7e1f:	48 89 c7             	mov    %rax,%rdi
    7e22:	31 f6                	xor    %esi,%esi
    7e24:	4c 89 ea             	mov    %r13,%rdx
    7e27:	4c 89 e9             	mov    %r13,%rcx
    7e2a:	e8 c1 a2 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    7e2f:	be 01 00 00 00       	mov    $0x1,%esi
    7e34:	4c 89 e7             	mov    %r12,%rdi
    7e37:	e8 64 a2 ff ff       	call   20a0 <omTensorSetDataType@plt>
    7e3c:	4c 89 e7             	mov    %r12,%rdi
    7e3f:	e8 8c a3 ff ff       	call   21d0 <omTensorGetShape@plt>
    7e44:	49 89 c5             	mov    %rax,%r13
    7e47:	4c 89 e7             	mov    %r12,%rdi
    7e4a:	e8 21 a2 ff ff       	call   2070 <omTensorGetStrides@plt>
    7e4f:	49 c7 45 00 80 00 00 	movq   $0x80,0x0(%r13)
    7e56:	00 
    7e57:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    7e5e:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    7e63:	4c 89 ef             	mov    %r13,%rdi
    7e66:	48 89 de             	mov    %rbx,%rsi
    7e69:	4c 89 fa             	mov    %r15,%rdx
    7e6c:	4c 89 e1             	mov    %r12,%rcx
    7e6f:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    7e75:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    7e7b:	6a 01                	push   $0x1
    7e7d:	6a 01                	push   $0x1
    7e7f:	6a 01                	push   $0x1
    7e81:	6a 01                	push   $0x1
    7e83:	e8 d8 a2 ff ff       	call   2160 <Conv2D_bias@plt>
    7e88:	48 83 c4 20          	add    $0x20,%rsp
    7e8c:	4c 89 ef             	mov    %r13,%rdi
    7e8f:	e8 1c a3 ff ff       	call   21b0 <omTensorDestroy@plt>
    7e94:	48 89 df             	mov    %rbx,%rdi
    7e97:	e8 14 a3 ff ff       	call   21b0 <omTensorDestroy@plt>
    7e9c:	4c 89 ff             	mov    %r15,%rdi
    7e9f:	e8 0c a3 ff ff       	call   21b0 <omTensorDestroy@plt>
    7ea4:	4c 89 e7             	mov    %r12,%rdi
    7ea7:	e8 04 a3 ff ff       	call   21b0 <omTensorDestroy@plt>
    7eac:	48 8b 3c 24          	mov    (%rsp),%rdi
    7eb0:	e8 5b a2 ff ff       	call   2110 <free@plt>
    7eb5:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    7eba:	e8 11 a2 ff ff       	call   20d0 <malloc@plt>
    7ebf:	49 89 c7             	mov    %rax,%r15
    7ec2:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    7ec6:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    7ecb:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    7ecf:	4c 89 24 24          	mov    %r12,(%rsp)
    7ed3:	bf 04 00 00 00       	mov    $0x4,%edi
    7ed8:	e8 c3 a2 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    7edd:	48 89 c3             	mov    %rax,%rbx
    7ee0:	48 89 c7             	mov    %rax,%rdi
    7ee3:	31 f6                	xor    %esi,%esi
    7ee5:	4c 89 fa             	mov    %r15,%rdx
    7ee8:	4c 89 e1             	mov    %r12,%rcx
    7eeb:	e8 00 a2 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    7ef0:	be 01 00 00 00       	mov    $0x1,%esi
    7ef5:	48 89 df             	mov    %rbx,%rdi
    7ef8:	e8 a3 a1 ff ff       	call   20a0 <omTensorSetDataType@plt>
    7efd:	48 89 df             	mov    %rbx,%rdi
    7f00:	e8 cb a2 ff ff       	call   21d0 <omTensorGetShape@plt>
    7f05:	49 89 c4             	mov    %rax,%r12
    7f08:	48 89 df             	mov    %rbx,%rdi
    7f0b:	e8 60 a1 ff ff       	call   2070 <omTensorGetStrides@plt>
    7f10:	4d 89 34 24          	mov    %r14,(%r12)
    7f14:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    7f1b:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    7f22:	00 00 
    7f24:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    7f2b:	00 
    7f2c:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    7f33:	00 00 
    7f35:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    7f3c:	00 
    7f3d:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    7f44:	00 00 
    7f46:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    7f4d:	00 
    7f4e:	bf 04 00 00 00       	mov    $0x4,%edi
    7f53:	e8 48 a2 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    7f58:	49 89 c4             	mov    %rax,%r12
    7f5b:	48 89 c7             	mov    %rax,%rdi
    7f5e:	31 f6                	xor    %esi,%esi
    7f60:	4c 8b 7c 24 30       	mov    0x30(%rsp),%r15
    7f65:	4c 89 fa             	mov    %r15,%rdx
    7f68:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    7f6d:	e8 7e a1 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    7f72:	be 01 00 00 00       	mov    $0x1,%esi
    7f77:	4c 89 e7             	mov    %r12,%rdi
    7f7a:	e8 21 a1 ff ff       	call   20a0 <omTensorSetDataType@plt>
    7f7f:	4c 89 e7             	mov    %r12,%rdi
    7f82:	e8 49 a2 ff ff       	call   21d0 <omTensorGetShape@plt>
    7f87:	49 89 c5             	mov    %rax,%r13
    7f8a:	4c 89 e7             	mov    %r12,%rdi
    7f8d:	e8 de a0 ff ff       	call   2070 <omTensorGetStrides@plt>
    7f92:	4d 89 75 00          	mov    %r14,0x0(%r13)
    7f96:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    7f9d:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    7fa4:	00 
    7fa5:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    7fac:	00 
    7fad:	49 c7 45 10 06 00 00 	movq   $0x6,0x10(%r13)
    7fb4:	00 
    7fb5:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    7fbc:	00 
    7fbd:	49 c7 45 18 06 00 00 	movq   $0x6,0x18(%r13)
    7fc4:	00 
    7fc5:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    7fcc:	00 
    7fcd:	48 89 df             	mov    %rbx,%rdi
    7fd0:	4c 89 e6             	mov    %r12,%rsi
    7fd3:	e8 18 a2 ff ff       	call   21f0 <Relu@plt>
    7fd8:	48 89 df             	mov    %rbx,%rdi
    7fdb:	e8 d0 a1 ff ff       	call   21b0 <omTensorDestroy@plt>
    7fe0:	4c 89 e7             	mov    %r12,%rdi
    7fe3:	e8 c8 a1 ff ff       	call   21b0 <omTensorDestroy@plt>
    7fe8:	4c 89 ff             	mov    %r15,%rdi
    7feb:	e8 20 a1 ff ff       	call   2110 <free@plt>
    7ff0:	bf 18 00 00 00       	mov    $0x18,%edi
    7ff5:	e8 d6 a0 ff ff       	call   20d0 <malloc@plt>
    7ffa:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    7fff:	48 83 c0 0f          	add    $0xf,%rax
    8003:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    8007:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    800c:	4c 89 30             	mov    %r14,(%rax)
    800f:	bf 28 00 00 00       	mov    $0x28,%edi
    8014:	e8 b7 a0 ff ff       	call   20d0 <malloc@plt>
    8019:	48 8d 48 0f          	lea    0xf(%rax),%rcx
    801d:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
    8021:	4c 89 31             	mov    %r14,(%rcx)
    8024:	48 c7 41 08 80 00 00 	movq   $0x80,0x8(%rcx)
    802b:	00 
    802c:	48 c7 41 10 ff ff ff 	movq   $0xffffffffffffffff,0x10(%rcx)
    8033:	ff 
    8034:	48 89 c7             	mov    %rax,%rdi
    8037:	e8 d4 a0 ff ff       	call   2110 <free@plt>
    803c:	48 89 e9             	mov    %rbp,%rcx
    803f:	48 c1 e1 07          	shl    $0x7,%rcx
    8043:	48 8b 5c 24 60       	mov    0x60(%rsp),%rbx
    8048:	48 89 d8             	mov    %rbx,%rax
    804b:	48 0f af c1          	imul   %rcx,%rax
    804f:	49 89 cf             	mov    %rcx,%r15
    8052:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
    8057:	48 8d 3c 85 10 00 00 	lea    0x10(,%rax,4),%rdi
    805e:	00 
    805f:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
    8064:	e8 67 a0 ff ff       	call   20d0 <malloc@plt>
    8069:	49 89 c4             	mov    %rax,%r12
    806c:	4c 8d 68 0f          	lea    0xf(%rax),%r13
    8070:	49 83 e5 f0          	and    $0xfffffffffffffff0,%r13
    8074:	bf 03 00 00 00       	mov    $0x3,%edi
    8079:	e8 22 a1 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    807e:	49 89 c6             	mov    %rax,%r14
    8081:	48 89 c7             	mov    %rax,%rdi
    8084:	31 f6                	xor    %esi,%esi
    8086:	4c 89 64 24 70       	mov    %r12,0x70(%rsp)
    808b:	4c 89 e2             	mov    %r12,%rdx
    808e:	4c 89 6c 24 08       	mov    %r13,0x8(%rsp)
    8093:	4c 89 e9             	mov    %r13,%rcx
    8096:	e8 55 a0 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    809b:	be 01 00 00 00       	mov    $0x1,%esi
    80a0:	4c 89 f7             	mov    %r14,%rdi
    80a3:	e8 f8 9f ff ff       	call   20a0 <omTensorSetDataType@plt>
    80a8:	4c 89 f7             	mov    %r14,%rdi
    80ab:	e8 20 a1 ff ff       	call   21d0 <omTensorGetShape@plt>
    80b0:	49 89 c4             	mov    %rax,%r12
    80b3:	4c 89 f7             	mov    %r14,%rdi
    80b6:	e8 b5 9f ff ff       	call   2070 <omTensorGetStrides@plt>
    80bb:	49 89 1c 24          	mov    %rbx,(%r12)
    80bf:	4c 89 38             	mov    %r15,(%rax)
    80c2:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    80c9:	00 00 
    80cb:	48 89 68 08          	mov    %rbp,0x8(%rax)
    80cf:	49 89 6c 24 10       	mov    %rbp,0x10(%r12)
    80d4:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    80db:	00 
    80dc:	bf 02 00 00 00       	mov    $0x2,%edi
    80e1:	e8 ba a0 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    80e6:	49 89 c5             	mov    %rax,%r13
    80e9:	48 8d 0d 20 46 0a 00 	lea    0xa4620(%rip),%rcx        # ac710 <constant_18_mobilenet>
    80f0:	48 89 c7             	mov    %rax,%rdi
    80f3:	31 f6                	xor    %esi,%esi
    80f5:	48 89 ca             	mov    %rcx,%rdx
    80f8:	e8 f3 9f ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    80fd:	be 01 00 00 00       	mov    $0x1,%esi
    8102:	4c 89 ef             	mov    %r13,%rdi
    8105:	e8 96 9f ff ff       	call   20a0 <omTensorSetDataType@plt>
    810a:	4c 89 ef             	mov    %r13,%rdi
    810d:	e8 be a0 ff ff       	call   21d0 <omTensorGetShape@plt>
    8112:	49 89 c4             	mov    %rax,%r12
    8115:	4c 89 ef             	mov    %r13,%rdi
    8118:	e8 53 9f ff ff       	call   2070 <omTensorGetStrides@plt>
    811d:	49 c7 04 24 80 00 00 	movq   $0x80,(%r12)
    8124:	00 
    8125:	48 c7 00 80 00 00 00 	movq   $0x80,(%rax)
    812c:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    8133:	00 00 
    8135:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    813c:	00 
    813d:	bf 03 00 00 00       	mov    $0x3,%edi
    8142:	e8 59 a0 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    8147:	49 89 c4             	mov    %rax,%r12
    814a:	48 89 c7             	mov    %rax,%rdi
    814d:	31 f6                	xor    %esi,%esi
    814f:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    8154:	48 8b 0c 24          	mov    (%rsp),%rcx
    8158:	e8 93 9f ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    815d:	be 01 00 00 00       	mov    $0x1,%esi
    8162:	4c 89 e7             	mov    %r12,%rdi
    8165:	e8 36 9f ff ff       	call   20a0 <omTensorSetDataType@plt>
    816a:	4c 89 e7             	mov    %r12,%rdi
    816d:	e8 5e a0 ff ff       	call   21d0 <omTensorGetShape@plt>
    8172:	49 89 c7             	mov    %rax,%r15
    8175:	4c 89 e7             	mov    %r12,%rdi
    8178:	e8 f3 9e ff ff       	call   2070 <omTensorGetStrides@plt>
    817d:	49 89 1f             	mov    %rbx,(%r15)
    8180:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    8185:	48 89 08             	mov    %rcx,(%rax)
    8188:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    818f:	00 
    8190:	48 89 68 08          	mov    %rbp,0x8(%rax)
    8194:	49 89 6f 10          	mov    %rbp,0x10(%r15)
    8198:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    819f:	00 
    81a0:	4c 89 f7             	mov    %r14,%rdi
    81a3:	4c 89 ee             	mov    %r13,%rsi
    81a6:	4c 89 e2             	mov    %r12,%rdx
    81a9:	e8 a2 9e ff ff       	call   2050 <MatMul@plt>
    81ae:	4c 89 f7             	mov    %r14,%rdi
    81b1:	e8 fa 9f ff ff       	call   21b0 <omTensorDestroy@plt>
    81b6:	4c 89 ef             	mov    %r13,%rdi
    81b9:	e8 f2 9f ff ff       	call   21b0 <omTensorDestroy@plt>
    81be:	4c 89 e7             	mov    %r12,%rdi
    81c1:	e8 ea 9f ff ff       	call   21b0 <omTensorDestroy@plt>
    81c6:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    81cb:	e8 40 9f ff ff       	call   2110 <free@plt>
    81d0:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    81d5:	e8 f6 9e ff ff       	call   20d0 <malloc@plt>
    81da:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    81df:	48 83 c0 0f          	add    $0xf,%rax
    81e3:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    81e7:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    81ec:	48 85 db             	test   %rbx,%rbx
    81ef:	0f 8e 58 01 00 00    	jle    834d <main_graph_mobilenet+0x607d>
    81f5:	48 89 e8             	mov    %rbp,%rax
    81f8:	48 c1 e0 09          	shl    $0x9,%rax
    81fc:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    8201:	48 8d 0c ad 00 00 00 	lea    0x0(,%rbp,4),%rcx
    8208:	00 
    8209:	48 89 eb             	mov    %rbp,%rbx
    820c:	48 83 e3 f8          	and    $0xfffffffffffffff8,%rbx
    8210:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    8215:	49 8d 45 10          	lea    0x10(%r13),%rax
    8219:	48 89 04 24          	mov    %rax,(%rsp)
    821d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    8222:	48 8d 50 10          	lea    0x10(%rax),%rdx
    8226:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
    822b:	45 31 c0             	xor    %r8d,%r8d
    822e:	4c 8d 0d db 3b 01 00 	lea    0x13bdb(%rip),%r9        # 1be10 <constant_31_mobilenet>
    8235:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    823a:	eb 28                	jmp    8264 <main_graph_mobilenet+0x5f94>
    823c:	0f 1f 40 00          	nopl   0x0(%rax)
    8240:	49 ff c0             	inc    %r8
    8243:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    8248:	48 01 04 24          	add    %rax,(%rsp)
    824c:	48 01 44 24 20       	add    %rax,0x20(%rsp)
    8251:	49 01 c5             	add    %rax,%r13
    8254:	48 01 44 24 30       	add    %rax,0x30(%rsp)
    8259:	4c 3b 44 24 60       	cmp    0x60(%rsp),%r8
    825e:	0f 84 e9 00 00 00    	je     834d <main_graph_mobilenet+0x607d>
    8264:	48 85 ed             	test   %rbp,%rbp
    8267:	7e d7                	jle    8240 <main_graph_mobilenet+0x5f70>
    8269:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
    826e:	4d 0f af f0          	imul   %r8,%r14
    8272:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    8277:	4e 8d 3c 30          	lea    (%rax,%r14,1),%r15
    827b:	4c 03 74 24 08       	add    0x8(%rsp),%r14
    8280:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
    8285:	4d 89 eb             	mov    %r13,%r11
    8288:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    828d:	48 8b 34 24          	mov    (%rsp),%rsi
    8291:	31 c0                	xor    %eax,%eax
    8293:	eb 22                	jmp    82b7 <main_graph_mobilenet+0x5fe7>
    8295:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    829c:	00 00 00 
    829f:	90                   	nop
    82a0:	48 ff c0             	inc    %rax
    82a3:	48 01 ce             	add    %rcx,%rsi
    82a6:	48 01 cf             	add    %rcx,%rdi
    82a9:	49 01 cb             	add    %rcx,%r11
    82ac:	49 01 cc             	add    %rcx,%r12
    82af:	48 3d 80 00 00 00    	cmp    $0x80,%rax
    82b5:	74 89                	je     8240 <main_graph_mobilenet+0x5f70>
    82b7:	48 83 fd 08          	cmp    $0x8,%rbp
    82bb:	f3 41 0f 10 04 81    	movss  (%r9,%rax,4),%xmm0
    82c1:	73 0d                	jae    82d0 <main_graph_mobilenet+0x6000>
    82c3:	31 d2                	xor    %edx,%edx
    82c5:	eb 69                	jmp    8330 <main_graph_mobilenet+0x6060>
    82c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    82ce:	00 00 
    82d0:	48 89 ca             	mov    %rcx,%rdx
    82d3:	48 0f af d0          	imul   %rax,%rdx
    82d7:	4d 8d 14 16          	lea    (%r14,%rdx,1),%r10
    82db:	4c 01 fa             	add    %r15,%rdx
    82de:	4c 29 d2             	sub    %r10,%rdx
    82e1:	48 83 fa 20          	cmp    $0x20,%rdx
    82e5:	73 09                	jae    82f0 <main_graph_mobilenet+0x6020>
    82e7:	31 d2                	xor    %edx,%edx
    82e9:	eb 45                	jmp    8330 <main_graph_mobilenet+0x6060>
    82eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    82f0:	0f 28 c8             	movaps %xmm0,%xmm1
    82f3:	0f c6 c8 00          	shufps $0x0,%xmm0,%xmm1
    82f7:	31 d2                	xor    %edx,%edx
    82f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8300:	0f 10 54 97 f0       	movups -0x10(%rdi,%rdx,4),%xmm2
    8305:	0f 10 1c 97          	movups (%rdi,%rdx,4),%xmm3
    8309:	0f 58 d1             	addps  %xmm1,%xmm2
    830c:	0f 58 d9             	addps  %xmm1,%xmm3
    830f:	0f 11 54 96 f0       	movups %xmm2,-0x10(%rsi,%rdx,4)
    8314:	0f 11 1c 96          	movups %xmm3,(%rsi,%rdx,4)
    8318:	48 83 c2 08          	add    $0x8,%rdx
    831c:	48 39 d3             	cmp    %rdx,%rbx
    831f:	75 df                	jne    8300 <main_graph_mobilenet+0x6030>
    8321:	48 39 dd             	cmp    %rbx,%rbp
    8324:	48 89 da             	mov    %rbx,%rdx
    8327:	0f 84 73 ff ff ff    	je     82a0 <main_graph_mobilenet+0x5fd0>
    832d:	0f 1f 00             	nopl   (%rax)
    8330:	f3 41 0f 10 0c 94    	movss  (%r12,%rdx,4),%xmm1
    8336:	f3 0f 58 c8          	addss  %xmm0,%xmm1
    833a:	f3 41 0f 11 0c 93    	movss  %xmm1,(%r11,%rdx,4)
    8340:	48 ff c2             	inc    %rdx
    8343:	48 39 d5             	cmp    %rdx,%rbp
    8346:	75 e8                	jne    8330 <main_graph_mobilenet+0x6060>
    8348:	e9 53 ff ff ff       	jmp    82a0 <main_graph_mobilenet+0x5fd0>
    834d:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    8352:	e8 b9 9d ff ff       	call   2110 <free@plt>
    8357:	bf 30 00 00 00       	mov    $0x30,%edi
    835c:	e8 6f 9d ff ff       	call   20d0 <malloc@plt>
    8361:	49 89 c7             	mov    %rax,%r15
    8364:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    8368:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    836c:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    8371:	48 8b 00             	mov    (%rax),%rax
    8374:	49 89 04 24          	mov    %rax,(%r12)
    8378:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    837d:	e8 8e 9d ff ff       	call   2110 <free@plt>
    8382:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    8387:	48 0f af e8          	imul   %rax,%rbp
    838b:	48 c1 e5 07          	shl    $0x7,%rbp
    838f:	49 8b 0c 24          	mov    (%r12),%rcx
    8393:	48 85 c9             	test   %rcx,%rcx
    8396:	48 0f 44 c8          	cmove  %rax,%rcx
    839a:	48 89 c8             	mov    %rcx,%rax
    839d:	48 c1 e0 09          	shl    $0x9,%rax
    83a1:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    83a5:	48 8d 34 c0          	lea    (%rax,%rax,8),%rsi
    83a9:	b8 00 12 00 00       	mov    $0x1200,%eax
    83ae:	48 0f 44 f0          	cmove  %rax,%rsi
    83b2:	48 89 f0             	mov    %rsi,%rax
    83b5:	48 f7 d0             	not    %rax
    83b8:	48 c1 f8 3f          	sar    $0x3f,%rax
    83bc:	48 83 c8 01          	or     $0x1,%rax
    83c0:	48 29 e8             	sub    %rbp,%rax
    83c3:	48 99                	cqto   
    83c5:	48 f7 fe             	idiv   %rsi
    83c8:	49 89 c6             	mov    %rax,%r14
    83cb:	49 f7 d6             	not    %r14
    83ce:	48 89 e8             	mov    %rbp,%rax
    83d1:	48 99                	cqto   
    83d3:	48 f7 fe             	idiv   %rsi
    83d6:	48 85 f6             	test   %rsi,%rsi
    83d9:	48 89 c2             	mov    %rax,%rdx
    83dc:	49 0f 48 d6          	cmovs  %r14,%rdx
    83e0:	48 85 ed             	test   %rbp,%rbp
    83e3:	48 0f 4e d0          	cmovle %rax,%rdx
    83e7:	48 85 f6             	test   %rsi,%rsi
    83ea:	4c 0f 4e f2          	cmovle %rdx,%r14
    83ee:	48 85 ed             	test   %rbp,%rbp
    83f1:	4c 0f 49 f2          	cmovns %rdx,%r14
    83f5:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    83f9:	4c 0f 45 f1          	cmovne %rcx,%r14
    83fd:	4c 89 f0             	mov    %r14,%rax
    8400:	48 c1 e0 09          	shl    $0x9,%rax
    8404:	48 8d 0c c0          	lea    (%rax,%rax,8),%rcx
    8408:	4c 89 f6             	mov    %r14,%rsi
    840b:	48 c1 e6 07          	shl    $0x7,%rsi
    840f:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    8413:	b8 80 00 00 00       	mov    $0x80,%eax
    8418:	48 0f 44 f0          	cmove  %rax,%rsi
    841c:	48 89 f0             	mov    %rsi,%rax
    841f:	48 f7 d0             	not    %rax
    8422:	48 c1 f8 3f          	sar    $0x3f,%rax
    8426:	48 83 c8 01          	or     $0x1,%rax
    842a:	48 29 c8             	sub    %rcx,%rax
    842d:	48 99                	cqto   
    842f:	48 f7 fe             	idiv   %rsi
    8432:	48 89 c5             	mov    %rax,%rbp
    8435:	48 f7 d5             	not    %rbp
    8438:	48 89 c8             	mov    %rcx,%rax
    843b:	48 99                	cqto   
    843d:	48 f7 fe             	idiv   %rsi
    8440:	48 85 f6             	test   %rsi,%rsi
    8443:	48 89 c2             	mov    %rax,%rdx
    8446:	48 0f 48 d5          	cmovs  %rbp,%rdx
    844a:	48 85 c9             	test   %rcx,%rcx
    844d:	48 0f 4e d0          	cmovle %rax,%rdx
    8451:	48 85 f6             	test   %rsi,%rsi
    8454:	48 0f 4e ea          	cmovle %rdx,%rbp
    8458:	48 85 c9             	test   %rcx,%rcx
    845b:	48 0f 49 ea          	cmovns %rdx,%rbp
    845f:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    8463:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    846a:	00 00 
    846c:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    8473:	00 00 
    8475:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    847c:	00 00 
    847e:	4c 89 f0             	mov    %r14,%rax
    8481:	48 0f 44 c5          	cmove  %rbp,%rax
    8485:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    848a:	4c 89 ff             	mov    %r15,%rdi
    848d:	e8 7e 9c ff ff       	call   2110 <free@plt>
    8492:	4c 89 f0             	mov    %r14,%rax
    8495:	48 c1 e0 0b          	shl    $0xb,%rax
    8499:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    849e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    84a3:	e8 28 9c ff ff       	call   20d0 <malloc@plt>
    84a8:	49 89 c7             	mov    %rax,%r15
    84ab:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    84af:	48 89 04 24          	mov    %rax,(%rsp)
    84b3:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    84b7:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    84bc:	bf 04 00 00 00       	mov    $0x4,%edi
    84c1:	e8 da 9c ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    84c6:	48 89 c3             	mov    %rax,%rbx
    84c9:	48 89 c7             	mov    %rax,%rdi
    84cc:	31 f6                	xor    %esi,%esi
    84ce:	4c 89 fa             	mov    %r15,%rdx
    84d1:	4c 89 e1             	mov    %r12,%rcx
    84d4:	e8 17 9c ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    84d9:	be 01 00 00 00       	mov    $0x1,%esi
    84de:	48 89 df             	mov    %rbx,%rdi
    84e1:	e8 ba 9b ff ff       	call   20a0 <omTensorSetDataType@plt>
    84e6:	48 89 df             	mov    %rbx,%rdi
    84e9:	e8 e2 9c ff ff       	call   21d0 <omTensorGetShape@plt>
    84ee:	49 89 c4             	mov    %rax,%r12
    84f1:	48 89 df             	mov    %rbx,%rdi
    84f4:	e8 77 9b ff ff       	call   2070 <omTensorGetStrides@plt>
    84f9:	4d 89 34 24          	mov    %r14,(%r12)
    84fd:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    8504:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    850b:	00 00 
    850d:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    8514:	00 
    8515:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    851c:	00 00 
    851e:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    8525:	00 
    8526:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    852d:	00 00 
    852f:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    8536:	00 
    8537:	bf 04 00 00 00       	mov    $0x4,%edi
    853c:	e8 5f 9c ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    8541:	49 89 c4             	mov    %rax,%r12
    8544:	48 89 c7             	mov    %rax,%rdi
    8547:	31 f6                	xor    %esi,%esi
    8549:	4c 8b 7c 24 50       	mov    0x50(%rsp),%r15
    854e:	4c 89 fa             	mov    %r15,%rdx
    8551:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    8556:	e8 95 9b ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    855b:	be 01 00 00 00       	mov    $0x1,%esi
    8560:	4c 89 e7             	mov    %r12,%rdi
    8563:	e8 38 9b ff ff       	call   20a0 <omTensorSetDataType@plt>
    8568:	4c 89 e7             	mov    %r12,%rdi
    856b:	e8 60 9c ff ff       	call   21d0 <omTensorGetShape@plt>
    8570:	49 89 c5             	mov    %rax,%r13
    8573:	4c 89 e7             	mov    %r12,%rdi
    8576:	e8 f5 9a ff ff       	call   2070 <omTensorGetStrides@plt>
    857b:	4d 89 75 00          	mov    %r14,0x0(%r13)
    857f:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    8586:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    858d:	00 
    858e:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    8595:	00 
    8596:	49 c7 45 10 06 00 00 	movq   $0x6,0x10(%r13)
    859d:	00 
    859e:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    85a5:	00 
    85a6:	49 c7 45 18 06 00 00 	movq   $0x6,0x18(%r13)
    85ad:	00 
    85ae:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    85b5:	00 
    85b6:	48 89 df             	mov    %rbx,%rdi
    85b9:	4c 89 e6             	mov    %r12,%rsi
    85bc:	e8 2f 9c ff ff       	call   21f0 <Relu@plt>
    85c1:	48 89 df             	mov    %rbx,%rdi
    85c4:	e8 e7 9b ff ff       	call   21b0 <omTensorDestroy@plt>
    85c9:	4c 89 e7             	mov    %r12,%rdi
    85cc:	e8 df 9b ff ff       	call   21b0 <omTensorDestroy@plt>
    85d1:	4c 89 ff             	mov    %r15,%rdi
    85d4:	e8 37 9b ff ff       	call   2110 <free@plt>
    85d9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    85de:	e8 ed 9a ff ff       	call   20d0 <malloc@plt>
    85e3:	48 89 c3             	mov    %rax,%rbx
    85e6:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    85ea:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    85ef:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    85f3:	4c 89 7c 24 20       	mov    %r15,0x20(%rsp)
    85f8:	bf 04 00 00 00       	mov    $0x4,%edi
    85fd:	e8 9e 9b ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    8602:	49 89 c4             	mov    %rax,%r12
    8605:	48 89 c7             	mov    %rax,%rdi
    8608:	31 f6                	xor    %esi,%esi
    860a:	48 89 da             	mov    %rbx,%rdx
    860d:	4c 89 f9             	mov    %r15,%rcx
    8610:	e8 db 9a ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    8615:	be 01 00 00 00       	mov    $0x1,%esi
    861a:	4c 89 e3             	mov    %r12,%rbx
    861d:	4c 89 64 24 40       	mov    %r12,0x40(%rsp)
    8622:	4c 89 e7             	mov    %r12,%rdi
    8625:	e8 76 9a ff ff       	call   20a0 <omTensorSetDataType@plt>
    862a:	4c 89 e7             	mov    %r12,%rdi
    862d:	e8 9e 9b ff ff       	call   21d0 <omTensorGetShape@plt>
    8632:	49 89 c4             	mov    %rax,%r12
    8635:	48 89 df             	mov    %rbx,%rdi
    8638:	e8 33 9a ff ff       	call   2070 <omTensorGetStrides@plt>
    863d:	4d 89 34 24          	mov    %r14,(%r12)
    8641:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    8648:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    864f:	00 00 
    8651:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    8658:	00 
    8659:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    8660:	00 00 
    8662:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    8669:	00 
    866a:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    8671:	00 00 
    8673:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    867a:	00 
    867b:	bf 04 00 00 00       	mov    $0x4,%edi
    8680:	e8 1b 9b ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    8685:	48 89 c3             	mov    %rax,%rbx
    8688:	48 89 c7             	mov    %rax,%rdi
    868b:	31 f6                	xor    %esi,%esi
    868d:	48 8b 14 24          	mov    (%rsp),%rdx
    8691:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    8696:	e8 55 9a ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    869b:	be 01 00 00 00       	mov    $0x1,%esi
    86a0:	48 89 df             	mov    %rbx,%rdi
    86a3:	e8 f8 99 ff ff       	call   20a0 <omTensorSetDataType@plt>
    86a8:	48 89 df             	mov    %rbx,%rdi
    86ab:	e8 20 9b ff ff       	call   21d0 <omTensorGetShape@plt>
    86b0:	49 89 c7             	mov    %rax,%r15
    86b3:	48 89 df             	mov    %rbx,%rdi
    86b6:	e8 b5 99 ff ff       	call   2070 <omTensorGetStrides@plt>
    86bb:	4d 89 37             	mov    %r14,(%r15)
    86be:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    86c5:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    86cc:	00 
    86cd:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    86d4:	00 
    86d5:	49 c7 47 10 06 00 00 	movq   $0x6,0x10(%r15)
    86dc:	00 
    86dd:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    86e4:	00 
    86e5:	49 c7 47 18 06 00 00 	movq   $0x6,0x18(%r15)
    86ec:	00 
    86ed:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    86f4:	00 
    86f5:	4c 8d 25 54 b0 00 00 	lea    0xb054(%rip),%r12        # 13750 <constant_60_mobilenet>
    86fc:	bf 04 00 00 00       	mov    $0x4,%edi
    8701:	e8 9a 9a ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    8706:	49 89 c7             	mov    %rax,%r15
    8709:	48 89 c7             	mov    %rax,%rdi
    870c:	31 f6                	xor    %esi,%esi
    870e:	4c 89 e2             	mov    %r12,%rdx
    8711:	4c 89 e1             	mov    %r12,%rcx
    8714:	e8 d7 99 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    8719:	be 01 00 00 00       	mov    $0x1,%esi
    871e:	4c 89 ff             	mov    %r15,%rdi
    8721:	e8 7a 99 ff ff       	call   20a0 <omTensorSetDataType@plt>
    8726:	4c 89 ff             	mov    %r15,%rdi
    8729:	e8 a2 9a ff ff       	call   21d0 <omTensorGetShape@plt>
    872e:	49 89 c4             	mov    %rax,%r12
    8731:	4c 89 ff             	mov    %r15,%rdi
    8734:	e8 37 99 ff ff       	call   2070 <omTensorGetStrides@plt>
    8739:	49 c7 04 24 80 00 00 	movq   $0x80,(%r12)
    8740:	00 
    8741:	48 c7 00 09 00 00 00 	movq   $0x9,(%rax)
    8748:	49 c7 44 24 08 01 00 	movq   $0x1,0x8(%r12)
    874f:	00 00 
    8751:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    8758:	00 
    8759:	49 c7 44 24 10 03 00 	movq   $0x3,0x10(%r12)
    8760:	00 00 
    8762:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    8769:	00 
    876a:	49 c7 44 24 18 03 00 	movq   $0x3,0x18(%r12)
    8771:	00 00 
    8773:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    877a:	00 
    877b:	4c 8d 2d 4e 1a 01 00 	lea    0x11a4e(%rip),%r13        # 1a1d0 <constant_47_mobilenet>
    8782:	bf 01 00 00 00       	mov    $0x1,%edi
    8787:	e8 14 9a ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    878c:	49 89 c4             	mov    %rax,%r12
    878f:	48 89 c7             	mov    %rax,%rdi
    8792:	31 f6                	xor    %esi,%esi
    8794:	4c 89 ea             	mov    %r13,%rdx
    8797:	4c 89 e9             	mov    %r13,%rcx
    879a:	e8 51 99 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    879f:	be 01 00 00 00       	mov    $0x1,%esi
    87a4:	4c 89 e7             	mov    %r12,%rdi
    87a7:	e8 f4 98 ff ff       	call   20a0 <omTensorSetDataType@plt>
    87ac:	4c 89 e7             	mov    %r12,%rdi
    87af:	e8 1c 9a ff ff       	call   21d0 <omTensorGetShape@plt>
    87b4:	49 89 c5             	mov    %rax,%r13
    87b7:	4c 89 e7             	mov    %r12,%rdi
    87ba:	e8 b1 98 ff ff       	call   2070 <omTensorGetStrides@plt>
    87bf:	49 c7 45 00 80 00 00 	movq   $0x80,0x0(%r13)
    87c6:	00 
    87c7:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    87ce:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    87d3:	4c 89 ef             	mov    %r13,%rdi
    87d6:	48 89 de             	mov    %rbx,%rsi
    87d9:	4c 89 fa             	mov    %r15,%rdx
    87dc:	4c 89 e1             	mov    %r12,%rcx
    87df:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    87e5:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    87eb:	6a 01                	push   $0x1
    87ed:	6a 01                	push   $0x1
    87ef:	6a 01                	push   $0x1
    87f1:	6a 01                	push   $0x1
    87f3:	e8 68 99 ff ff       	call   2160 <Conv2D_bias@plt>
    87f8:	48 83 c4 20          	add    $0x20,%rsp
    87fc:	4c 89 ef             	mov    %r13,%rdi
    87ff:	e8 ac 99 ff ff       	call   21b0 <omTensorDestroy@plt>
    8804:	48 89 df             	mov    %rbx,%rdi
    8807:	e8 a4 99 ff ff       	call   21b0 <omTensorDestroy@plt>
    880c:	4c 89 ff             	mov    %r15,%rdi
    880f:	e8 9c 99 ff ff       	call   21b0 <omTensorDestroy@plt>
    8814:	4c 89 e7             	mov    %r12,%rdi
    8817:	e8 94 99 ff ff       	call   21b0 <omTensorDestroy@plt>
    881c:	48 8b 3c 24          	mov    (%rsp),%rdi
    8820:	e8 eb 98 ff ff       	call   2110 <free@plt>
    8825:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    882a:	e8 a1 98 ff ff       	call   20d0 <malloc@plt>
    882f:	49 89 c7             	mov    %rax,%r15
    8832:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    8836:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    883b:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    883f:	4c 89 24 24          	mov    %r12,(%rsp)
    8843:	bf 04 00 00 00       	mov    $0x4,%edi
    8848:	e8 53 99 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    884d:	48 89 c3             	mov    %rax,%rbx
    8850:	48 89 c7             	mov    %rax,%rdi
    8853:	31 f6                	xor    %esi,%esi
    8855:	4c 89 fa             	mov    %r15,%rdx
    8858:	4c 89 e1             	mov    %r12,%rcx
    885b:	e8 90 98 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    8860:	be 01 00 00 00       	mov    $0x1,%esi
    8865:	48 89 df             	mov    %rbx,%rdi
    8868:	e8 33 98 ff ff       	call   20a0 <omTensorSetDataType@plt>
    886d:	48 89 df             	mov    %rbx,%rdi
    8870:	e8 5b 99 ff ff       	call   21d0 <omTensorGetShape@plt>
    8875:	49 89 c4             	mov    %rax,%r12
    8878:	48 89 df             	mov    %rbx,%rdi
    887b:	e8 f0 97 ff ff       	call   2070 <omTensorGetStrides@plt>
    8880:	4d 89 34 24          	mov    %r14,(%r12)
    8884:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    888b:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    8892:	00 00 
    8894:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    889b:	00 
    889c:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    88a3:	00 00 
    88a5:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    88ac:	00 
    88ad:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    88b4:	00 00 
    88b6:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    88bd:	00 
    88be:	bf 04 00 00 00       	mov    $0x4,%edi
    88c3:	e8 d8 98 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    88c8:	49 89 c4             	mov    %rax,%r12
    88cb:	48 89 c7             	mov    %rax,%rdi
    88ce:	31 f6                	xor    %esi,%esi
    88d0:	4c 8b 7c 24 30       	mov    0x30(%rsp),%r15
    88d5:	4c 89 fa             	mov    %r15,%rdx
    88d8:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    88dd:	e8 0e 98 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    88e2:	be 01 00 00 00       	mov    $0x1,%esi
    88e7:	4c 89 e7             	mov    %r12,%rdi
    88ea:	e8 b1 97 ff ff       	call   20a0 <omTensorSetDataType@plt>
    88ef:	4c 89 e7             	mov    %r12,%rdi
    88f2:	e8 d9 98 ff ff       	call   21d0 <omTensorGetShape@plt>
    88f7:	49 89 c5             	mov    %rax,%r13
    88fa:	4c 89 e7             	mov    %r12,%rdi
    88fd:	e8 6e 97 ff ff       	call   2070 <omTensorGetStrides@plt>
    8902:	4d 89 75 00          	mov    %r14,0x0(%r13)
    8906:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    890d:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    8914:	00 
    8915:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    891c:	00 
    891d:	49 c7 45 10 06 00 00 	movq   $0x6,0x10(%r13)
    8924:	00 
    8925:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    892c:	00 
    892d:	49 c7 45 18 06 00 00 	movq   $0x6,0x18(%r13)
    8934:	00 
    8935:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    893c:	00 
    893d:	48 89 df             	mov    %rbx,%rdi
    8940:	4c 89 e6             	mov    %r12,%rsi
    8943:	e8 a8 98 ff ff       	call   21f0 <Relu@plt>
    8948:	48 89 df             	mov    %rbx,%rdi
    894b:	e8 60 98 ff ff       	call   21b0 <omTensorDestroy@plt>
    8950:	4c 89 e7             	mov    %r12,%rdi
    8953:	e8 58 98 ff ff       	call   21b0 <omTensorDestroy@plt>
    8958:	4c 89 ff             	mov    %r15,%rdi
    895b:	e8 b0 97 ff ff       	call   2110 <free@plt>
    8960:	bf 18 00 00 00       	mov    $0x18,%edi
    8965:	e8 66 97 ff ff       	call   20d0 <malloc@plt>
    896a:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    896f:	48 83 c0 0f          	add    $0xf,%rax
    8973:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    8977:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    897c:	4c 89 30             	mov    %r14,(%rax)
    897f:	bf 28 00 00 00       	mov    $0x28,%edi
    8984:	e8 47 97 ff ff       	call   20d0 <malloc@plt>
    8989:	48 8d 48 0f          	lea    0xf(%rax),%rcx
    898d:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
    8991:	4c 89 31             	mov    %r14,(%rcx)
    8994:	48 c7 41 08 80 00 00 	movq   $0x80,0x8(%rcx)
    899b:	00 
    899c:	48 c7 41 10 ff ff ff 	movq   $0xffffffffffffffff,0x10(%rcx)
    89a3:	ff 
    89a4:	48 89 c7             	mov    %rax,%rdi
    89a7:	e8 64 97 ff ff       	call   2110 <free@plt>
    89ac:	48 89 e9             	mov    %rbp,%rcx
    89af:	48 c1 e1 07          	shl    $0x7,%rcx
    89b3:	48 8b 5c 24 60       	mov    0x60(%rsp),%rbx
    89b8:	48 89 d8             	mov    %rbx,%rax
    89bb:	48 0f af c1          	imul   %rcx,%rax
    89bf:	49 89 cf             	mov    %rcx,%r15
    89c2:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
    89c7:	48 8d 3c 85 10 00 00 	lea    0x10(,%rax,4),%rdi
    89ce:	00 
    89cf:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
    89d4:	e8 f7 96 ff ff       	call   20d0 <malloc@plt>
    89d9:	49 89 c4             	mov    %rax,%r12
    89dc:	4c 8d 68 0f          	lea    0xf(%rax),%r13
    89e0:	49 83 e5 f0          	and    $0xfffffffffffffff0,%r13
    89e4:	bf 03 00 00 00       	mov    $0x3,%edi
    89e9:	e8 b2 97 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    89ee:	49 89 c6             	mov    %rax,%r14
    89f1:	48 89 c7             	mov    %rax,%rdi
    89f4:	31 f6                	xor    %esi,%esi
    89f6:	4c 89 64 24 70       	mov    %r12,0x70(%rsp)
    89fb:	4c 89 e2             	mov    %r12,%rdx
    89fe:	4c 89 6c 24 08       	mov    %r13,0x8(%rsp)
    8a03:	4c 89 e9             	mov    %r13,%rcx
    8a06:	e8 e5 96 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    8a0b:	be 01 00 00 00       	mov    $0x1,%esi
    8a10:	4c 89 f7             	mov    %r14,%rdi
    8a13:	e8 88 96 ff ff       	call   20a0 <omTensorSetDataType@plt>
    8a18:	4c 89 f7             	mov    %r14,%rdi
    8a1b:	e8 b0 97 ff ff       	call   21d0 <omTensorGetShape@plt>
    8a20:	49 89 c4             	mov    %rax,%r12
    8a23:	4c 89 f7             	mov    %r14,%rdi
    8a26:	e8 45 96 ff ff       	call   2070 <omTensorGetStrides@plt>
    8a2b:	49 89 1c 24          	mov    %rbx,(%r12)
    8a2f:	4c 89 38             	mov    %r15,(%rax)
    8a32:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    8a39:	00 00 
    8a3b:	48 89 68 08          	mov    %rbp,0x8(%rax)
    8a3f:	49 89 6c 24 10       	mov    %rbp,0x10(%r12)
    8a44:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    8a4b:	00 
    8a4c:	bf 02 00 00 00       	mov    $0x2,%edi
    8a51:	e8 4a 97 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    8a56:	49 89 c5             	mov    %rax,%r13
    8a59:	48 8d 0d b0 3c 09 00 	lea    0x93cb0(%rip),%rcx        # 9c710 <constant_19_mobilenet>
    8a60:	48 89 c7             	mov    %rax,%rdi
    8a63:	31 f6                	xor    %esi,%esi
    8a65:	48 89 ca             	mov    %rcx,%rdx
    8a68:	e8 83 96 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    8a6d:	be 01 00 00 00       	mov    $0x1,%esi
    8a72:	4c 89 ef             	mov    %r13,%rdi
    8a75:	e8 26 96 ff ff       	call   20a0 <omTensorSetDataType@plt>
    8a7a:	4c 89 ef             	mov    %r13,%rdi
    8a7d:	e8 4e 97 ff ff       	call   21d0 <omTensorGetShape@plt>
    8a82:	49 89 c4             	mov    %rax,%r12
    8a85:	4c 89 ef             	mov    %r13,%rdi
    8a88:	e8 e3 95 ff ff       	call   2070 <omTensorGetStrides@plt>
    8a8d:	49 c7 04 24 80 00 00 	movq   $0x80,(%r12)
    8a94:	00 
    8a95:	48 c7 00 80 00 00 00 	movq   $0x80,(%rax)
    8a9c:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    8aa3:	00 00 
    8aa5:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    8aac:	00 
    8aad:	bf 03 00 00 00       	mov    $0x3,%edi
    8ab2:	e8 e9 96 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    8ab7:	49 89 c4             	mov    %rax,%r12
    8aba:	48 89 c7             	mov    %rax,%rdi
    8abd:	31 f6                	xor    %esi,%esi
    8abf:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    8ac4:	48 8b 0c 24          	mov    (%rsp),%rcx
    8ac8:	e8 23 96 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    8acd:	be 01 00 00 00       	mov    $0x1,%esi
    8ad2:	4c 89 e7             	mov    %r12,%rdi
    8ad5:	e8 c6 95 ff ff       	call   20a0 <omTensorSetDataType@plt>
    8ada:	4c 89 e7             	mov    %r12,%rdi
    8add:	e8 ee 96 ff ff       	call   21d0 <omTensorGetShape@plt>
    8ae2:	49 89 c7             	mov    %rax,%r15
    8ae5:	4c 89 e7             	mov    %r12,%rdi
    8ae8:	e8 83 95 ff ff       	call   2070 <omTensorGetStrides@plt>
    8aed:	49 89 1f             	mov    %rbx,(%r15)
    8af0:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    8af5:	48 89 08             	mov    %rcx,(%rax)
    8af8:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    8aff:	00 
    8b00:	48 89 68 08          	mov    %rbp,0x8(%rax)
    8b04:	49 89 6f 10          	mov    %rbp,0x10(%r15)
    8b08:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    8b0f:	00 
    8b10:	4c 89 f7             	mov    %r14,%rdi
    8b13:	4c 89 ee             	mov    %r13,%rsi
    8b16:	4c 89 e2             	mov    %r12,%rdx
    8b19:	e8 32 95 ff ff       	call   2050 <MatMul@plt>
    8b1e:	4c 89 f7             	mov    %r14,%rdi
    8b21:	e8 8a 96 ff ff       	call   21b0 <omTensorDestroy@plt>
    8b26:	4c 89 ef             	mov    %r13,%rdi
    8b29:	e8 82 96 ff ff       	call   21b0 <omTensorDestroy@plt>
    8b2e:	4c 89 e7             	mov    %r12,%rdi
    8b31:	e8 7a 96 ff ff       	call   21b0 <omTensorDestroy@plt>
    8b36:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    8b3b:	e8 d0 95 ff ff       	call   2110 <free@plt>
    8b40:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    8b45:	e8 86 95 ff ff       	call   20d0 <malloc@plt>
    8b4a:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    8b4f:	48 83 c0 0f          	add    $0xf,%rax
    8b53:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    8b57:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    8b5c:	48 85 db             	test   %rbx,%rbx
    8b5f:	0f 8e 58 01 00 00    	jle    8cbd <main_graph_mobilenet+0x69ed>
    8b65:	48 89 e8             	mov    %rbp,%rax
    8b68:	48 c1 e0 09          	shl    $0x9,%rax
    8b6c:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    8b71:	48 8d 0c ad 00 00 00 	lea    0x0(,%rbp,4),%rcx
    8b78:	00 
    8b79:	48 89 eb             	mov    %rbp,%rbx
    8b7c:	48 83 e3 f8          	and    $0xfffffffffffffff8,%rbx
    8b80:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    8b85:	49 8d 45 10          	lea    0x10(%r13),%rax
    8b89:	48 89 04 24          	mov    %rax,(%rsp)
    8b8d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    8b92:	48 8d 50 10          	lea    0x10(%rax),%rdx
    8b96:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
    8b9b:	45 31 c0             	xor    %r8d,%r8d
    8b9e:	4c 8d 0d 6b 30 01 00 	lea    0x1306b(%rip),%r9        # 1bc10 <constant_32_mobilenet>
    8ba5:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    8baa:	eb 28                	jmp    8bd4 <main_graph_mobilenet+0x6904>
    8bac:	0f 1f 40 00          	nopl   0x0(%rax)
    8bb0:	49 ff c0             	inc    %r8
    8bb3:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    8bb8:	48 01 04 24          	add    %rax,(%rsp)
    8bbc:	48 01 44 24 20       	add    %rax,0x20(%rsp)
    8bc1:	49 01 c5             	add    %rax,%r13
    8bc4:	48 01 44 24 30       	add    %rax,0x30(%rsp)
    8bc9:	4c 3b 44 24 60       	cmp    0x60(%rsp),%r8
    8bce:	0f 84 e9 00 00 00    	je     8cbd <main_graph_mobilenet+0x69ed>
    8bd4:	48 85 ed             	test   %rbp,%rbp
    8bd7:	7e d7                	jle    8bb0 <main_graph_mobilenet+0x68e0>
    8bd9:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
    8bde:	4d 0f af f0          	imul   %r8,%r14
    8be2:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    8be7:	4e 8d 3c 30          	lea    (%rax,%r14,1),%r15
    8beb:	4c 03 74 24 08       	add    0x8(%rsp),%r14
    8bf0:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
    8bf5:	4d 89 eb             	mov    %r13,%r11
    8bf8:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    8bfd:	48 8b 34 24          	mov    (%rsp),%rsi
    8c01:	31 c0                	xor    %eax,%eax
    8c03:	eb 22                	jmp    8c27 <main_graph_mobilenet+0x6957>
    8c05:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    8c0c:	00 00 00 
    8c0f:	90                   	nop
    8c10:	48 ff c0             	inc    %rax
    8c13:	48 01 ce             	add    %rcx,%rsi
    8c16:	48 01 cf             	add    %rcx,%rdi
    8c19:	49 01 cb             	add    %rcx,%r11
    8c1c:	49 01 cc             	add    %rcx,%r12
    8c1f:	48 3d 80 00 00 00    	cmp    $0x80,%rax
    8c25:	74 89                	je     8bb0 <main_graph_mobilenet+0x68e0>
    8c27:	48 83 fd 08          	cmp    $0x8,%rbp
    8c2b:	f3 41 0f 10 04 81    	movss  (%r9,%rax,4),%xmm0
    8c31:	73 0d                	jae    8c40 <main_graph_mobilenet+0x6970>
    8c33:	31 d2                	xor    %edx,%edx
    8c35:	eb 69                	jmp    8ca0 <main_graph_mobilenet+0x69d0>
    8c37:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    8c3e:	00 00 
    8c40:	48 89 ca             	mov    %rcx,%rdx
    8c43:	48 0f af d0          	imul   %rax,%rdx
    8c47:	4d 8d 14 16          	lea    (%r14,%rdx,1),%r10
    8c4b:	4c 01 fa             	add    %r15,%rdx
    8c4e:	4c 29 d2             	sub    %r10,%rdx
    8c51:	48 83 fa 20          	cmp    $0x20,%rdx
    8c55:	73 09                	jae    8c60 <main_graph_mobilenet+0x6990>
    8c57:	31 d2                	xor    %edx,%edx
    8c59:	eb 45                	jmp    8ca0 <main_graph_mobilenet+0x69d0>
    8c5b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    8c60:	0f 28 c8             	movaps %xmm0,%xmm1
    8c63:	0f c6 c8 00          	shufps $0x0,%xmm0,%xmm1
    8c67:	31 d2                	xor    %edx,%edx
    8c69:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    8c70:	0f 10 54 97 f0       	movups -0x10(%rdi,%rdx,4),%xmm2
    8c75:	0f 10 1c 97          	movups (%rdi,%rdx,4),%xmm3
    8c79:	0f 58 d1             	addps  %xmm1,%xmm2
    8c7c:	0f 58 d9             	addps  %xmm1,%xmm3
    8c7f:	0f 11 54 96 f0       	movups %xmm2,-0x10(%rsi,%rdx,4)
    8c84:	0f 11 1c 96          	movups %xmm3,(%rsi,%rdx,4)
    8c88:	48 83 c2 08          	add    $0x8,%rdx
    8c8c:	48 39 d3             	cmp    %rdx,%rbx
    8c8f:	75 df                	jne    8c70 <main_graph_mobilenet+0x69a0>
    8c91:	48 39 dd             	cmp    %rbx,%rbp
    8c94:	48 89 da             	mov    %rbx,%rdx
    8c97:	0f 84 73 ff ff ff    	je     8c10 <main_graph_mobilenet+0x6940>
    8c9d:	0f 1f 00             	nopl   (%rax)
    8ca0:	f3 41 0f 10 0c 94    	movss  (%r12,%rdx,4),%xmm1
    8ca6:	f3 0f 58 c8          	addss  %xmm0,%xmm1
    8caa:	f3 41 0f 11 0c 93    	movss  %xmm1,(%r11,%rdx,4)
    8cb0:	48 ff c2             	inc    %rdx
    8cb3:	48 39 d5             	cmp    %rdx,%rbp
    8cb6:	75 e8                	jne    8ca0 <main_graph_mobilenet+0x69d0>
    8cb8:	e9 53 ff ff ff       	jmp    8c10 <main_graph_mobilenet+0x6940>
    8cbd:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    8cc2:	e8 49 94 ff ff       	call   2110 <free@plt>
    8cc7:	bf 30 00 00 00       	mov    $0x30,%edi
    8ccc:	e8 ff 93 ff ff       	call   20d0 <malloc@plt>
    8cd1:	49 89 c7             	mov    %rax,%r15
    8cd4:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    8cd8:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    8cdc:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    8ce1:	48 8b 00             	mov    (%rax),%rax
    8ce4:	49 89 04 24          	mov    %rax,(%r12)
    8ce8:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    8ced:	e8 1e 94 ff ff       	call   2110 <free@plt>
    8cf2:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    8cf7:	48 0f af e8          	imul   %rax,%rbp
    8cfb:	48 c1 e5 07          	shl    $0x7,%rbp
    8cff:	49 8b 0c 24          	mov    (%r12),%rcx
    8d03:	48 85 c9             	test   %rcx,%rcx
    8d06:	48 0f 44 c8          	cmove  %rax,%rcx
    8d0a:	48 89 c8             	mov    %rcx,%rax
    8d0d:	48 c1 e0 09          	shl    $0x9,%rax
    8d11:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    8d15:	48 8d 34 c0          	lea    (%rax,%rax,8),%rsi
    8d19:	b8 00 12 00 00       	mov    $0x1200,%eax
    8d1e:	48 0f 44 f0          	cmove  %rax,%rsi
    8d22:	48 89 f0             	mov    %rsi,%rax
    8d25:	48 f7 d0             	not    %rax
    8d28:	48 c1 f8 3f          	sar    $0x3f,%rax
    8d2c:	48 83 c8 01          	or     $0x1,%rax
    8d30:	48 29 e8             	sub    %rbp,%rax
    8d33:	48 99                	cqto   
    8d35:	48 f7 fe             	idiv   %rsi
    8d38:	49 89 c6             	mov    %rax,%r14
    8d3b:	49 f7 d6             	not    %r14
    8d3e:	48 89 e8             	mov    %rbp,%rax
    8d41:	48 99                	cqto   
    8d43:	48 f7 fe             	idiv   %rsi
    8d46:	48 85 f6             	test   %rsi,%rsi
    8d49:	48 89 c2             	mov    %rax,%rdx
    8d4c:	49 0f 48 d6          	cmovs  %r14,%rdx
    8d50:	48 85 ed             	test   %rbp,%rbp
    8d53:	48 0f 4e d0          	cmovle %rax,%rdx
    8d57:	48 85 f6             	test   %rsi,%rsi
    8d5a:	4c 0f 4e f2          	cmovle %rdx,%r14
    8d5e:	48 85 ed             	test   %rbp,%rbp
    8d61:	4c 0f 49 f2          	cmovns %rdx,%r14
    8d65:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    8d69:	4c 0f 45 f1          	cmovne %rcx,%r14
    8d6d:	4c 89 f0             	mov    %r14,%rax
    8d70:	48 c1 e0 09          	shl    $0x9,%rax
    8d74:	48 8d 0c c0          	lea    (%rax,%rax,8),%rcx
    8d78:	4c 89 f6             	mov    %r14,%rsi
    8d7b:	48 c1 e6 07          	shl    $0x7,%rsi
    8d7f:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    8d83:	b8 80 00 00 00       	mov    $0x80,%eax
    8d88:	48 0f 44 f0          	cmove  %rax,%rsi
    8d8c:	48 89 f0             	mov    %rsi,%rax
    8d8f:	48 f7 d0             	not    %rax
    8d92:	48 c1 f8 3f          	sar    $0x3f,%rax
    8d96:	48 83 c8 01          	or     $0x1,%rax
    8d9a:	48 29 c8             	sub    %rcx,%rax
    8d9d:	48 99                	cqto   
    8d9f:	48 f7 fe             	idiv   %rsi
    8da2:	48 89 c5             	mov    %rax,%rbp
    8da5:	48 f7 d5             	not    %rbp
    8da8:	48 89 c8             	mov    %rcx,%rax
    8dab:	48 99                	cqto   
    8dad:	48 f7 fe             	idiv   %rsi
    8db0:	48 85 f6             	test   %rsi,%rsi
    8db3:	48 89 c2             	mov    %rax,%rdx
    8db6:	48 0f 48 d5          	cmovs  %rbp,%rdx
    8dba:	48 85 c9             	test   %rcx,%rcx
    8dbd:	48 0f 4e d0          	cmovle %rax,%rdx
    8dc1:	48 85 f6             	test   %rsi,%rsi
    8dc4:	48 0f 4e ea          	cmovle %rdx,%rbp
    8dc8:	48 85 c9             	test   %rcx,%rcx
    8dcb:	48 0f 49 ea          	cmovns %rdx,%rbp
    8dcf:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    8dd3:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    8dda:	00 00 
    8ddc:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    8de3:	00 00 
    8de5:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    8dec:	00 00 
    8dee:	4c 89 f0             	mov    %r14,%rax
    8df1:	48 0f 44 c5          	cmove  %rbp,%rax
    8df5:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    8dfa:	4c 89 ff             	mov    %r15,%rdi
    8dfd:	e8 0e 93 ff ff       	call   2110 <free@plt>
    8e02:	4c 89 f0             	mov    %r14,%rax
    8e05:	48 c1 e0 0b          	shl    $0xb,%rax
    8e09:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    8e0e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    8e13:	e8 b8 92 ff ff       	call   20d0 <malloc@plt>
    8e18:	49 89 c7             	mov    %rax,%r15
    8e1b:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    8e1f:	48 89 04 24          	mov    %rax,(%rsp)
    8e23:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    8e27:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    8e2c:	bf 04 00 00 00       	mov    $0x4,%edi
    8e31:	e8 6a 93 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    8e36:	48 89 c3             	mov    %rax,%rbx
    8e39:	48 89 c7             	mov    %rax,%rdi
    8e3c:	31 f6                	xor    %esi,%esi
    8e3e:	4c 89 fa             	mov    %r15,%rdx
    8e41:	4c 89 e1             	mov    %r12,%rcx
    8e44:	e8 a7 92 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    8e49:	be 01 00 00 00       	mov    $0x1,%esi
    8e4e:	48 89 df             	mov    %rbx,%rdi
    8e51:	e8 4a 92 ff ff       	call   20a0 <omTensorSetDataType@plt>
    8e56:	48 89 df             	mov    %rbx,%rdi
    8e59:	e8 72 93 ff ff       	call   21d0 <omTensorGetShape@plt>
    8e5e:	49 89 c4             	mov    %rax,%r12
    8e61:	48 89 df             	mov    %rbx,%rdi
    8e64:	e8 07 92 ff ff       	call   2070 <omTensorGetStrides@plt>
    8e69:	4d 89 34 24          	mov    %r14,(%r12)
    8e6d:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    8e74:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    8e7b:	00 00 
    8e7d:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    8e84:	00 
    8e85:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    8e8c:	00 00 
    8e8e:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    8e95:	00 
    8e96:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    8e9d:	00 00 
    8e9f:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    8ea6:	00 
    8ea7:	bf 04 00 00 00       	mov    $0x4,%edi
    8eac:	e8 ef 92 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    8eb1:	49 89 c4             	mov    %rax,%r12
    8eb4:	48 89 c7             	mov    %rax,%rdi
    8eb7:	31 f6                	xor    %esi,%esi
    8eb9:	4c 8b 7c 24 50       	mov    0x50(%rsp),%r15
    8ebe:	4c 89 fa             	mov    %r15,%rdx
    8ec1:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    8ec6:	e8 25 92 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    8ecb:	be 01 00 00 00       	mov    $0x1,%esi
    8ed0:	4c 89 e7             	mov    %r12,%rdi
    8ed3:	e8 c8 91 ff ff       	call   20a0 <omTensorSetDataType@plt>
    8ed8:	4c 89 e7             	mov    %r12,%rdi
    8edb:	e8 f0 92 ff ff       	call   21d0 <omTensorGetShape@plt>
    8ee0:	49 89 c5             	mov    %rax,%r13
    8ee3:	4c 89 e7             	mov    %r12,%rdi
    8ee6:	e8 85 91 ff ff       	call   2070 <omTensorGetStrides@plt>
    8eeb:	4d 89 75 00          	mov    %r14,0x0(%r13)
    8eef:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    8ef6:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    8efd:	00 
    8efe:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    8f05:	00 
    8f06:	49 c7 45 10 06 00 00 	movq   $0x6,0x10(%r13)
    8f0d:	00 
    8f0e:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    8f15:	00 
    8f16:	49 c7 45 18 06 00 00 	movq   $0x6,0x18(%r13)
    8f1d:	00 
    8f1e:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    8f25:	00 
    8f26:	48 89 df             	mov    %rbx,%rdi
    8f29:	4c 89 e6             	mov    %r12,%rsi
    8f2c:	e8 bf 92 ff ff       	call   21f0 <Relu@plt>
    8f31:	48 89 df             	mov    %rbx,%rdi
    8f34:	e8 77 92 ff ff       	call   21b0 <omTensorDestroy@plt>
    8f39:	4c 89 e7             	mov    %r12,%rdi
    8f3c:	e8 6f 92 ff ff       	call   21b0 <omTensorDestroy@plt>
    8f41:	4c 89 ff             	mov    %r15,%rdi
    8f44:	e8 c7 91 ff ff       	call   2110 <free@plt>
    8f49:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    8f4e:	e8 7d 91 ff ff       	call   20d0 <malloc@plt>
    8f53:	48 89 c3             	mov    %rax,%rbx
    8f56:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    8f5a:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    8f5f:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    8f63:	4c 89 7c 24 20       	mov    %r15,0x20(%rsp)
    8f68:	bf 04 00 00 00       	mov    $0x4,%edi
    8f6d:	e8 2e 92 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    8f72:	49 89 c4             	mov    %rax,%r12
    8f75:	48 89 c7             	mov    %rax,%rdi
    8f78:	31 f6                	xor    %esi,%esi
    8f7a:	48 89 da             	mov    %rbx,%rdx
    8f7d:	4c 89 f9             	mov    %r15,%rcx
    8f80:	e8 6b 91 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    8f85:	be 01 00 00 00       	mov    $0x1,%esi
    8f8a:	4c 89 e3             	mov    %r12,%rbx
    8f8d:	4c 89 64 24 40       	mov    %r12,0x40(%rsp)
    8f92:	4c 89 e7             	mov    %r12,%rdi
    8f95:	e8 06 91 ff ff       	call   20a0 <omTensorSetDataType@plt>
    8f9a:	4c 89 e7             	mov    %r12,%rdi
    8f9d:	e8 2e 92 ff ff       	call   21d0 <omTensorGetShape@plt>
    8fa2:	49 89 c4             	mov    %rax,%r12
    8fa5:	48 89 df             	mov    %rbx,%rdi
    8fa8:	e8 c3 90 ff ff       	call   2070 <omTensorGetStrides@plt>
    8fad:	4d 89 34 24          	mov    %r14,(%r12)
    8fb1:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    8fb8:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    8fbf:	00 00 
    8fc1:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    8fc8:	00 
    8fc9:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    8fd0:	00 00 
    8fd2:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    8fd9:	00 
    8fda:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    8fe1:	00 00 
    8fe3:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    8fea:	00 
    8feb:	bf 04 00 00 00       	mov    $0x4,%edi
    8ff0:	e8 ab 91 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    8ff5:	48 89 c3             	mov    %rax,%rbx
    8ff8:	48 89 c7             	mov    %rax,%rdi
    8ffb:	31 f6                	xor    %esi,%esi
    8ffd:	48 8b 14 24          	mov    (%rsp),%rdx
    9001:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    9006:	e8 e5 90 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    900b:	be 01 00 00 00       	mov    $0x1,%esi
    9010:	48 89 df             	mov    %rbx,%rdi
    9013:	e8 88 90 ff ff       	call   20a0 <omTensorSetDataType@plt>
    9018:	48 89 df             	mov    %rbx,%rdi
    901b:	e8 b0 91 ff ff       	call   21d0 <omTensorGetShape@plt>
    9020:	49 89 c7             	mov    %rax,%r15
    9023:	48 89 df             	mov    %rbx,%rdi
    9026:	e8 45 90 ff ff       	call   2070 <omTensorGetStrides@plt>
    902b:	4d 89 37             	mov    %r14,(%r15)
    902e:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    9035:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    903c:	00 
    903d:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    9044:	00 
    9045:	49 c7 47 10 06 00 00 	movq   $0x6,0x10(%r15)
    904c:	00 
    904d:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    9054:	00 
    9055:	49 c7 47 18 06 00 00 	movq   $0x6,0x18(%r15)
    905c:	00 
    905d:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    9064:	00 
    9065:	4c 8d 25 e4 94 00 00 	lea    0x94e4(%rip),%r12        # 12550 <constant_61_mobilenet>
    906c:	bf 04 00 00 00       	mov    $0x4,%edi
    9071:	e8 2a 91 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    9076:	49 89 c7             	mov    %rax,%r15
    9079:	48 89 c7             	mov    %rax,%rdi
    907c:	31 f6                	xor    %esi,%esi
    907e:	4c 89 e2             	mov    %r12,%rdx
    9081:	4c 89 e1             	mov    %r12,%rcx
    9084:	e8 67 90 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    9089:	be 01 00 00 00       	mov    $0x1,%esi
    908e:	4c 89 ff             	mov    %r15,%rdi
    9091:	e8 0a 90 ff ff       	call   20a0 <omTensorSetDataType@plt>
    9096:	4c 89 ff             	mov    %r15,%rdi
    9099:	e8 32 91 ff ff       	call   21d0 <omTensorGetShape@plt>
    909e:	49 89 c4             	mov    %rax,%r12
    90a1:	4c 89 ff             	mov    %r15,%rdi
    90a4:	e8 c7 8f ff ff       	call   2070 <omTensorGetStrides@plt>
    90a9:	49 c7 04 24 80 00 00 	movq   $0x80,(%r12)
    90b0:	00 
    90b1:	48 c7 00 09 00 00 00 	movq   $0x9,(%rax)
    90b8:	49 c7 44 24 08 01 00 	movq   $0x1,0x8(%r12)
    90bf:	00 00 
    90c1:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    90c8:	00 
    90c9:	49 c7 44 24 10 03 00 	movq   $0x3,0x10(%r12)
    90d0:	00 00 
    90d2:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    90d9:	00 
    90da:	49 c7 44 24 18 03 00 	movq   $0x3,0x18(%r12)
    90e1:	00 00 
    90e3:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    90ea:	00 
    90eb:	4c 8d 2d de 12 01 00 	lea    0x112de(%rip),%r13        # 1a3d0 <constant_46_mobilenet>
    90f2:	bf 01 00 00 00       	mov    $0x1,%edi
    90f7:	e8 a4 90 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    90fc:	49 89 c4             	mov    %rax,%r12
    90ff:	48 89 c7             	mov    %rax,%rdi
    9102:	31 f6                	xor    %esi,%esi
    9104:	4c 89 ea             	mov    %r13,%rdx
    9107:	4c 89 e9             	mov    %r13,%rcx
    910a:	e8 e1 8f ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    910f:	be 01 00 00 00       	mov    $0x1,%esi
    9114:	4c 89 e7             	mov    %r12,%rdi
    9117:	e8 84 8f ff ff       	call   20a0 <omTensorSetDataType@plt>
    911c:	4c 89 e7             	mov    %r12,%rdi
    911f:	e8 ac 90 ff ff       	call   21d0 <omTensorGetShape@plt>
    9124:	49 89 c5             	mov    %rax,%r13
    9127:	4c 89 e7             	mov    %r12,%rdi
    912a:	e8 41 8f ff ff       	call   2070 <omTensorGetStrides@plt>
    912f:	49 c7 45 00 80 00 00 	movq   $0x80,0x0(%r13)
    9136:	00 
    9137:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    913e:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    9143:	4c 89 ef             	mov    %r13,%rdi
    9146:	48 89 de             	mov    %rbx,%rsi
    9149:	4c 89 fa             	mov    %r15,%rdx
    914c:	4c 89 e1             	mov    %r12,%rcx
    914f:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    9155:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    915b:	6a 01                	push   $0x1
    915d:	6a 01                	push   $0x1
    915f:	6a 01                	push   $0x1
    9161:	6a 01                	push   $0x1
    9163:	e8 f8 8f ff ff       	call   2160 <Conv2D_bias@plt>
    9168:	48 83 c4 20          	add    $0x20,%rsp
    916c:	4c 89 ef             	mov    %r13,%rdi
    916f:	e8 3c 90 ff ff       	call   21b0 <omTensorDestroy@plt>
    9174:	48 89 df             	mov    %rbx,%rdi
    9177:	e8 34 90 ff ff       	call   21b0 <omTensorDestroy@plt>
    917c:	4c 89 ff             	mov    %r15,%rdi
    917f:	e8 2c 90 ff ff       	call   21b0 <omTensorDestroy@plt>
    9184:	4c 89 e7             	mov    %r12,%rdi
    9187:	e8 24 90 ff ff       	call   21b0 <omTensorDestroy@plt>
    918c:	48 8b 3c 24          	mov    (%rsp),%rdi
    9190:	e8 7b 8f ff ff       	call   2110 <free@plt>
    9195:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    919a:	e8 31 8f ff ff       	call   20d0 <malloc@plt>
    919f:	49 89 c7             	mov    %rax,%r15
    91a2:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    91a6:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    91ab:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    91af:	4c 89 24 24          	mov    %r12,(%rsp)
    91b3:	bf 04 00 00 00       	mov    $0x4,%edi
    91b8:	e8 e3 8f ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    91bd:	48 89 c3             	mov    %rax,%rbx
    91c0:	48 89 c7             	mov    %rax,%rdi
    91c3:	31 f6                	xor    %esi,%esi
    91c5:	4c 89 fa             	mov    %r15,%rdx
    91c8:	4c 89 e1             	mov    %r12,%rcx
    91cb:	e8 20 8f ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    91d0:	be 01 00 00 00       	mov    $0x1,%esi
    91d5:	48 89 df             	mov    %rbx,%rdi
    91d8:	e8 c3 8e ff ff       	call   20a0 <omTensorSetDataType@plt>
    91dd:	48 89 df             	mov    %rbx,%rdi
    91e0:	e8 eb 8f ff ff       	call   21d0 <omTensorGetShape@plt>
    91e5:	49 89 c4             	mov    %rax,%r12
    91e8:	48 89 df             	mov    %rbx,%rdi
    91eb:	e8 80 8e ff ff       	call   2070 <omTensorGetStrides@plt>
    91f0:	4d 89 34 24          	mov    %r14,(%r12)
    91f4:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    91fb:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    9202:	00 00 
    9204:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    920b:	00 
    920c:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    9213:	00 00 
    9215:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    921c:	00 
    921d:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    9224:	00 00 
    9226:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    922d:	00 
    922e:	bf 04 00 00 00       	mov    $0x4,%edi
    9233:	e8 68 8f ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    9238:	49 89 c4             	mov    %rax,%r12
    923b:	48 89 c7             	mov    %rax,%rdi
    923e:	31 f6                	xor    %esi,%esi
    9240:	4c 8b 7c 24 30       	mov    0x30(%rsp),%r15
    9245:	4c 89 fa             	mov    %r15,%rdx
    9248:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    924d:	e8 9e 8e ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    9252:	be 01 00 00 00       	mov    $0x1,%esi
    9257:	4c 89 e7             	mov    %r12,%rdi
    925a:	e8 41 8e ff ff       	call   20a0 <omTensorSetDataType@plt>
    925f:	4c 89 e7             	mov    %r12,%rdi
    9262:	e8 69 8f ff ff       	call   21d0 <omTensorGetShape@plt>
    9267:	49 89 c5             	mov    %rax,%r13
    926a:	4c 89 e7             	mov    %r12,%rdi
    926d:	e8 fe 8d ff ff       	call   2070 <omTensorGetStrides@plt>
    9272:	4d 89 75 00          	mov    %r14,0x0(%r13)
    9276:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    927d:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    9284:	00 
    9285:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    928c:	00 
    928d:	49 c7 45 10 06 00 00 	movq   $0x6,0x10(%r13)
    9294:	00 
    9295:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    929c:	00 
    929d:	49 c7 45 18 06 00 00 	movq   $0x6,0x18(%r13)
    92a4:	00 
    92a5:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    92ac:	00 
    92ad:	48 89 df             	mov    %rbx,%rdi
    92b0:	4c 89 e6             	mov    %r12,%rsi
    92b3:	e8 38 8f ff ff       	call   21f0 <Relu@plt>
    92b8:	48 89 df             	mov    %rbx,%rdi
    92bb:	e8 f0 8e ff ff       	call   21b0 <omTensorDestroy@plt>
    92c0:	4c 89 e7             	mov    %r12,%rdi
    92c3:	e8 e8 8e ff ff       	call   21b0 <omTensorDestroy@plt>
    92c8:	4c 89 ff             	mov    %r15,%rdi
    92cb:	e8 40 8e ff ff       	call   2110 <free@plt>
    92d0:	bf 18 00 00 00       	mov    $0x18,%edi
    92d5:	e8 f6 8d ff ff       	call   20d0 <malloc@plt>
    92da:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    92df:	48 83 c0 0f          	add    $0xf,%rax
    92e3:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    92e7:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    92ec:	4c 89 30             	mov    %r14,(%rax)
    92ef:	bf 28 00 00 00       	mov    $0x28,%edi
    92f4:	e8 d7 8d ff ff       	call   20d0 <malloc@plt>
    92f9:	48 8d 48 0f          	lea    0xf(%rax),%rcx
    92fd:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
    9301:	4c 89 31             	mov    %r14,(%rcx)
    9304:	48 c7 41 08 80 00 00 	movq   $0x80,0x8(%rcx)
    930b:	00 
    930c:	48 c7 41 10 ff ff ff 	movq   $0xffffffffffffffff,0x10(%rcx)
    9313:	ff 
    9314:	48 89 c7             	mov    %rax,%rdi
    9317:	e8 f4 8d ff ff       	call   2110 <free@plt>
    931c:	48 89 e9             	mov    %rbp,%rcx
    931f:	48 c1 e1 07          	shl    $0x7,%rcx
    9323:	48 8b 5c 24 60       	mov    0x60(%rsp),%rbx
    9328:	48 89 d8             	mov    %rbx,%rax
    932b:	48 0f af c1          	imul   %rcx,%rax
    932f:	49 89 cf             	mov    %rcx,%r15
    9332:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
    9337:	48 8d 3c 85 10 00 00 	lea    0x10(,%rax,4),%rdi
    933e:	00 
    933f:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
    9344:	e8 87 8d ff ff       	call   20d0 <malloc@plt>
    9349:	49 89 c4             	mov    %rax,%r12
    934c:	4c 8d 68 0f          	lea    0xf(%rax),%r13
    9350:	49 83 e5 f0          	and    $0xfffffffffffffff0,%r13
    9354:	bf 03 00 00 00       	mov    $0x3,%edi
    9359:	e8 42 8e ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    935e:	49 89 c6             	mov    %rax,%r14
    9361:	48 89 c7             	mov    %rax,%rdi
    9364:	31 f6                	xor    %esi,%esi
    9366:	4c 89 64 24 70       	mov    %r12,0x70(%rsp)
    936b:	4c 89 e2             	mov    %r12,%rdx
    936e:	4c 89 6c 24 08       	mov    %r13,0x8(%rsp)
    9373:	4c 89 e9             	mov    %r13,%rcx
    9376:	e8 75 8d ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    937b:	be 01 00 00 00       	mov    $0x1,%esi
    9380:	4c 89 f7             	mov    %r14,%rdi
    9383:	e8 18 8d ff ff       	call   20a0 <omTensorSetDataType@plt>
    9388:	4c 89 f7             	mov    %r14,%rdi
    938b:	e8 40 8e ff ff       	call   21d0 <omTensorGetShape@plt>
    9390:	49 89 c4             	mov    %rax,%r12
    9393:	4c 89 f7             	mov    %r14,%rdi
    9396:	e8 d5 8c ff ff       	call   2070 <omTensorGetStrides@plt>
    939b:	49 89 1c 24          	mov    %rbx,(%r12)
    939f:	4c 89 38             	mov    %r15,(%rax)
    93a2:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    93a9:	00 00 
    93ab:	48 89 68 08          	mov    %rbp,0x8(%rax)
    93af:	49 89 6c 24 10       	mov    %rbp,0x10(%r12)
    93b4:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    93bb:	00 
    93bc:	bf 02 00 00 00       	mov    $0x2,%edi
    93c1:	e8 da 8d ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    93c6:	49 89 c5             	mov    %rax,%r13
    93c9:	48 8d 0d 40 33 08 00 	lea    0x83340(%rip),%rcx        # 8c710 <constant_20_mobilenet>
    93d0:	48 89 c7             	mov    %rax,%rdi
    93d3:	31 f6                	xor    %esi,%esi
    93d5:	48 89 ca             	mov    %rcx,%rdx
    93d8:	e8 13 8d ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    93dd:	be 01 00 00 00       	mov    $0x1,%esi
    93e2:	4c 89 ef             	mov    %r13,%rdi
    93e5:	e8 b6 8c ff ff       	call   20a0 <omTensorSetDataType@plt>
    93ea:	4c 89 ef             	mov    %r13,%rdi
    93ed:	e8 de 8d ff ff       	call   21d0 <omTensorGetShape@plt>
    93f2:	49 89 c4             	mov    %rax,%r12
    93f5:	4c 89 ef             	mov    %r13,%rdi
    93f8:	e8 73 8c ff ff       	call   2070 <omTensorGetStrides@plt>
    93fd:	49 c7 04 24 80 00 00 	movq   $0x80,(%r12)
    9404:	00 
    9405:	48 c7 00 80 00 00 00 	movq   $0x80,(%rax)
    940c:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    9413:	00 00 
    9415:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    941c:	00 
    941d:	bf 03 00 00 00       	mov    $0x3,%edi
    9422:	e8 79 8d ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    9427:	49 89 c4             	mov    %rax,%r12
    942a:	48 89 c7             	mov    %rax,%rdi
    942d:	31 f6                	xor    %esi,%esi
    942f:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    9434:	48 8b 0c 24          	mov    (%rsp),%rcx
    9438:	e8 b3 8c ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    943d:	be 01 00 00 00       	mov    $0x1,%esi
    9442:	4c 89 e7             	mov    %r12,%rdi
    9445:	e8 56 8c ff ff       	call   20a0 <omTensorSetDataType@plt>
    944a:	4c 89 e7             	mov    %r12,%rdi
    944d:	e8 7e 8d ff ff       	call   21d0 <omTensorGetShape@plt>
    9452:	49 89 c7             	mov    %rax,%r15
    9455:	4c 89 e7             	mov    %r12,%rdi
    9458:	e8 13 8c ff ff       	call   2070 <omTensorGetStrides@plt>
    945d:	49 89 1f             	mov    %rbx,(%r15)
    9460:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    9465:	48 89 08             	mov    %rcx,(%rax)
    9468:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    946f:	00 
    9470:	48 89 68 08          	mov    %rbp,0x8(%rax)
    9474:	49 89 6f 10          	mov    %rbp,0x10(%r15)
    9478:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    947f:	00 
    9480:	4c 89 f7             	mov    %r14,%rdi
    9483:	4c 89 ee             	mov    %r13,%rsi
    9486:	4c 89 e2             	mov    %r12,%rdx
    9489:	e8 c2 8b ff ff       	call   2050 <MatMul@plt>
    948e:	4c 89 f7             	mov    %r14,%rdi
    9491:	e8 1a 8d ff ff       	call   21b0 <omTensorDestroy@plt>
    9496:	4c 89 ef             	mov    %r13,%rdi
    9499:	e8 12 8d ff ff       	call   21b0 <omTensorDestroy@plt>
    949e:	4c 89 e7             	mov    %r12,%rdi
    94a1:	e8 0a 8d ff ff       	call   21b0 <omTensorDestroy@plt>
    94a6:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    94ab:	e8 60 8c ff ff       	call   2110 <free@plt>
    94b0:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    94b5:	e8 16 8c ff ff       	call   20d0 <malloc@plt>
    94ba:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    94bf:	48 83 c0 0f          	add    $0xf,%rax
    94c3:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    94c7:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    94cc:	48 85 db             	test   %rbx,%rbx
    94cf:	0f 8e 58 01 00 00    	jle    962d <main_graph_mobilenet+0x735d>
    94d5:	48 89 e8             	mov    %rbp,%rax
    94d8:	48 c1 e0 09          	shl    $0x9,%rax
    94dc:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    94e1:	48 8d 0c ad 00 00 00 	lea    0x0(,%rbp,4),%rcx
    94e8:	00 
    94e9:	48 89 eb             	mov    %rbp,%rbx
    94ec:	48 83 e3 f8          	and    $0xfffffffffffffff8,%rbx
    94f0:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    94f5:	49 8d 45 10          	lea    0x10(%r13),%rax
    94f9:	48 89 04 24          	mov    %rax,(%rsp)
    94fd:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    9502:	48 8d 50 10          	lea    0x10(%rax),%rdx
    9506:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
    950b:	45 31 c0             	xor    %r8d,%r8d
    950e:	4c 8d 0d fb 24 01 00 	lea    0x124fb(%rip),%r9        # 1ba10 <constant_33_mobilenet>
    9515:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    951a:	eb 28                	jmp    9544 <main_graph_mobilenet+0x7274>
    951c:	0f 1f 40 00          	nopl   0x0(%rax)
    9520:	49 ff c0             	inc    %r8
    9523:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    9528:	48 01 04 24          	add    %rax,(%rsp)
    952c:	48 01 44 24 20       	add    %rax,0x20(%rsp)
    9531:	49 01 c5             	add    %rax,%r13
    9534:	48 01 44 24 30       	add    %rax,0x30(%rsp)
    9539:	4c 3b 44 24 60       	cmp    0x60(%rsp),%r8
    953e:	0f 84 e9 00 00 00    	je     962d <main_graph_mobilenet+0x735d>
    9544:	48 85 ed             	test   %rbp,%rbp
    9547:	7e d7                	jle    9520 <main_graph_mobilenet+0x7250>
    9549:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
    954e:	4d 0f af f0          	imul   %r8,%r14
    9552:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    9557:	4e 8d 3c 30          	lea    (%rax,%r14,1),%r15
    955b:	4c 03 74 24 08       	add    0x8(%rsp),%r14
    9560:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
    9565:	4d 89 eb             	mov    %r13,%r11
    9568:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    956d:	48 8b 34 24          	mov    (%rsp),%rsi
    9571:	31 c0                	xor    %eax,%eax
    9573:	eb 22                	jmp    9597 <main_graph_mobilenet+0x72c7>
    9575:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    957c:	00 00 00 
    957f:	90                   	nop
    9580:	48 ff c0             	inc    %rax
    9583:	48 01 ce             	add    %rcx,%rsi
    9586:	48 01 cf             	add    %rcx,%rdi
    9589:	49 01 cb             	add    %rcx,%r11
    958c:	49 01 cc             	add    %rcx,%r12
    958f:	48 3d 80 00 00 00    	cmp    $0x80,%rax
    9595:	74 89                	je     9520 <main_graph_mobilenet+0x7250>
    9597:	48 83 fd 08          	cmp    $0x8,%rbp
    959b:	f3 41 0f 10 04 81    	movss  (%r9,%rax,4),%xmm0
    95a1:	73 0d                	jae    95b0 <main_graph_mobilenet+0x72e0>
    95a3:	31 d2                	xor    %edx,%edx
    95a5:	eb 69                	jmp    9610 <main_graph_mobilenet+0x7340>
    95a7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    95ae:	00 00 
    95b0:	48 89 ca             	mov    %rcx,%rdx
    95b3:	48 0f af d0          	imul   %rax,%rdx
    95b7:	4d 8d 14 16          	lea    (%r14,%rdx,1),%r10
    95bb:	4c 01 fa             	add    %r15,%rdx
    95be:	4c 29 d2             	sub    %r10,%rdx
    95c1:	48 83 fa 20          	cmp    $0x20,%rdx
    95c5:	73 09                	jae    95d0 <main_graph_mobilenet+0x7300>
    95c7:	31 d2                	xor    %edx,%edx
    95c9:	eb 45                	jmp    9610 <main_graph_mobilenet+0x7340>
    95cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    95d0:	0f 28 c8             	movaps %xmm0,%xmm1
    95d3:	0f c6 c8 00          	shufps $0x0,%xmm0,%xmm1
    95d7:	31 d2                	xor    %edx,%edx
    95d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    95e0:	0f 10 54 97 f0       	movups -0x10(%rdi,%rdx,4),%xmm2
    95e5:	0f 10 1c 97          	movups (%rdi,%rdx,4),%xmm3
    95e9:	0f 58 d1             	addps  %xmm1,%xmm2
    95ec:	0f 58 d9             	addps  %xmm1,%xmm3
    95ef:	0f 11 54 96 f0       	movups %xmm2,-0x10(%rsi,%rdx,4)
    95f4:	0f 11 1c 96          	movups %xmm3,(%rsi,%rdx,4)
    95f8:	48 83 c2 08          	add    $0x8,%rdx
    95fc:	48 39 d3             	cmp    %rdx,%rbx
    95ff:	75 df                	jne    95e0 <main_graph_mobilenet+0x7310>
    9601:	48 39 dd             	cmp    %rbx,%rbp
    9604:	48 89 da             	mov    %rbx,%rdx
    9607:	0f 84 73 ff ff ff    	je     9580 <main_graph_mobilenet+0x72b0>
    960d:	0f 1f 00             	nopl   (%rax)
    9610:	f3 41 0f 10 0c 94    	movss  (%r12,%rdx,4),%xmm1
    9616:	f3 0f 58 c8          	addss  %xmm0,%xmm1
    961a:	f3 41 0f 11 0c 93    	movss  %xmm1,(%r11,%rdx,4)
    9620:	48 ff c2             	inc    %rdx
    9623:	48 39 d5             	cmp    %rdx,%rbp
    9626:	75 e8                	jne    9610 <main_graph_mobilenet+0x7340>
    9628:	e9 53 ff ff ff       	jmp    9580 <main_graph_mobilenet+0x72b0>
    962d:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    9632:	e8 d9 8a ff ff       	call   2110 <free@plt>
    9637:	bf 30 00 00 00       	mov    $0x30,%edi
    963c:	e8 8f 8a ff ff       	call   20d0 <malloc@plt>
    9641:	49 89 c7             	mov    %rax,%r15
    9644:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    9648:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    964c:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    9651:	48 8b 00             	mov    (%rax),%rax
    9654:	49 89 04 24          	mov    %rax,(%r12)
    9658:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    965d:	e8 ae 8a ff ff       	call   2110 <free@plt>
    9662:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    9667:	48 0f af e8          	imul   %rax,%rbp
    966b:	48 c1 e5 07          	shl    $0x7,%rbp
    966f:	49 8b 0c 24          	mov    (%r12),%rcx
    9673:	48 85 c9             	test   %rcx,%rcx
    9676:	48 0f 44 c8          	cmove  %rax,%rcx
    967a:	48 89 c8             	mov    %rcx,%rax
    967d:	48 c1 e0 09          	shl    $0x9,%rax
    9681:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    9685:	48 8d 34 c0          	lea    (%rax,%rax,8),%rsi
    9689:	b8 00 12 00 00       	mov    $0x1200,%eax
    968e:	48 0f 44 f0          	cmove  %rax,%rsi
    9692:	48 89 f0             	mov    %rsi,%rax
    9695:	48 f7 d0             	not    %rax
    9698:	48 c1 f8 3f          	sar    $0x3f,%rax
    969c:	48 83 c8 01          	or     $0x1,%rax
    96a0:	48 29 e8             	sub    %rbp,%rax
    96a3:	48 99                	cqto   
    96a5:	48 f7 fe             	idiv   %rsi
    96a8:	49 89 c6             	mov    %rax,%r14
    96ab:	49 f7 d6             	not    %r14
    96ae:	48 89 e8             	mov    %rbp,%rax
    96b1:	48 99                	cqto   
    96b3:	48 f7 fe             	idiv   %rsi
    96b6:	48 85 f6             	test   %rsi,%rsi
    96b9:	48 89 c2             	mov    %rax,%rdx
    96bc:	49 0f 48 d6          	cmovs  %r14,%rdx
    96c0:	48 85 ed             	test   %rbp,%rbp
    96c3:	48 0f 4e d0          	cmovle %rax,%rdx
    96c7:	48 85 f6             	test   %rsi,%rsi
    96ca:	4c 0f 4e f2          	cmovle %rdx,%r14
    96ce:	48 85 ed             	test   %rbp,%rbp
    96d1:	4c 0f 49 f2          	cmovns %rdx,%r14
    96d5:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    96d9:	4c 0f 45 f1          	cmovne %rcx,%r14
    96dd:	4c 89 f0             	mov    %r14,%rax
    96e0:	48 c1 e0 09          	shl    $0x9,%rax
    96e4:	48 8d 0c c0          	lea    (%rax,%rax,8),%rcx
    96e8:	4c 89 f6             	mov    %r14,%rsi
    96eb:	48 c1 e6 07          	shl    $0x7,%rsi
    96ef:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    96f3:	b8 80 00 00 00       	mov    $0x80,%eax
    96f8:	48 0f 44 f0          	cmove  %rax,%rsi
    96fc:	48 89 f0             	mov    %rsi,%rax
    96ff:	48 f7 d0             	not    %rax
    9702:	48 c1 f8 3f          	sar    $0x3f,%rax
    9706:	48 83 c8 01          	or     $0x1,%rax
    970a:	48 29 c8             	sub    %rcx,%rax
    970d:	48 99                	cqto   
    970f:	48 f7 fe             	idiv   %rsi
    9712:	48 89 c5             	mov    %rax,%rbp
    9715:	48 f7 d5             	not    %rbp
    9718:	48 89 c8             	mov    %rcx,%rax
    971b:	48 99                	cqto   
    971d:	48 f7 fe             	idiv   %rsi
    9720:	48 85 f6             	test   %rsi,%rsi
    9723:	48 89 c2             	mov    %rax,%rdx
    9726:	48 0f 48 d5          	cmovs  %rbp,%rdx
    972a:	48 85 c9             	test   %rcx,%rcx
    972d:	48 0f 4e d0          	cmovle %rax,%rdx
    9731:	48 85 f6             	test   %rsi,%rsi
    9734:	48 0f 4e ea          	cmovle %rdx,%rbp
    9738:	48 85 c9             	test   %rcx,%rcx
    973b:	48 0f 49 ea          	cmovns %rdx,%rbp
    973f:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    9743:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    974a:	00 00 
    974c:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    9753:	00 00 
    9755:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    975c:	00 00 
    975e:	4c 89 f0             	mov    %r14,%rax
    9761:	48 0f 44 c5          	cmove  %rbp,%rax
    9765:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    976a:	4c 89 ff             	mov    %r15,%rdi
    976d:	e8 9e 89 ff ff       	call   2110 <free@plt>
    9772:	4c 89 f0             	mov    %r14,%rax
    9775:	48 c1 e0 0b          	shl    $0xb,%rax
    9779:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    977e:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    9783:	e8 48 89 ff ff       	call   20d0 <malloc@plt>
    9788:	49 89 c7             	mov    %rax,%r15
    978b:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    978f:	48 89 04 24          	mov    %rax,(%rsp)
    9793:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    9797:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    979c:	bf 04 00 00 00       	mov    $0x4,%edi
    97a1:	e8 fa 89 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    97a6:	48 89 c3             	mov    %rax,%rbx
    97a9:	48 89 c7             	mov    %rax,%rdi
    97ac:	31 f6                	xor    %esi,%esi
    97ae:	4c 89 fa             	mov    %r15,%rdx
    97b1:	4c 89 e1             	mov    %r12,%rcx
    97b4:	e8 37 89 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    97b9:	be 01 00 00 00       	mov    $0x1,%esi
    97be:	48 89 df             	mov    %rbx,%rdi
    97c1:	e8 da 88 ff ff       	call   20a0 <omTensorSetDataType@plt>
    97c6:	48 89 df             	mov    %rbx,%rdi
    97c9:	e8 02 8a ff ff       	call   21d0 <omTensorGetShape@plt>
    97ce:	49 89 c4             	mov    %rax,%r12
    97d1:	48 89 df             	mov    %rbx,%rdi
    97d4:	e8 97 88 ff ff       	call   2070 <omTensorGetStrides@plt>
    97d9:	4d 89 34 24          	mov    %r14,(%r12)
    97dd:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    97e4:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    97eb:	00 00 
    97ed:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    97f4:	00 
    97f5:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    97fc:	00 00 
    97fe:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    9805:	00 
    9806:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    980d:	00 00 
    980f:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    9816:	00 
    9817:	bf 04 00 00 00       	mov    $0x4,%edi
    981c:	e8 7f 89 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    9821:	49 89 c4             	mov    %rax,%r12
    9824:	48 89 c7             	mov    %rax,%rdi
    9827:	31 f6                	xor    %esi,%esi
    9829:	4c 8b 7c 24 50       	mov    0x50(%rsp),%r15
    982e:	4c 89 fa             	mov    %r15,%rdx
    9831:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    9836:	e8 b5 88 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    983b:	be 01 00 00 00       	mov    $0x1,%esi
    9840:	4c 89 e7             	mov    %r12,%rdi
    9843:	e8 58 88 ff ff       	call   20a0 <omTensorSetDataType@plt>
    9848:	4c 89 e7             	mov    %r12,%rdi
    984b:	e8 80 89 ff ff       	call   21d0 <omTensorGetShape@plt>
    9850:	49 89 c5             	mov    %rax,%r13
    9853:	4c 89 e7             	mov    %r12,%rdi
    9856:	e8 15 88 ff ff       	call   2070 <omTensorGetStrides@plt>
    985b:	4d 89 75 00          	mov    %r14,0x0(%r13)
    985f:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    9866:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    986d:	00 
    986e:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    9875:	00 
    9876:	49 c7 45 10 06 00 00 	movq   $0x6,0x10(%r13)
    987d:	00 
    987e:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    9885:	00 
    9886:	49 c7 45 18 06 00 00 	movq   $0x6,0x18(%r13)
    988d:	00 
    988e:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    9895:	00 
    9896:	48 89 df             	mov    %rbx,%rdi
    9899:	4c 89 e6             	mov    %r12,%rsi
    989c:	e8 4f 89 ff ff       	call   21f0 <Relu@plt>
    98a1:	48 89 df             	mov    %rbx,%rdi
    98a4:	e8 07 89 ff ff       	call   21b0 <omTensorDestroy@plt>
    98a9:	4c 89 e7             	mov    %r12,%rdi
    98ac:	e8 ff 88 ff ff       	call   21b0 <omTensorDestroy@plt>
    98b1:	4c 89 ff             	mov    %r15,%rdi
    98b4:	e8 57 88 ff ff       	call   2110 <free@plt>
    98b9:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    98be:	e8 0d 88 ff ff       	call   20d0 <malloc@plt>
    98c3:	48 89 c3             	mov    %rax,%rbx
    98c6:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    98ca:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    98cf:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    98d3:	4c 89 7c 24 20       	mov    %r15,0x20(%rsp)
    98d8:	bf 04 00 00 00       	mov    $0x4,%edi
    98dd:	e8 be 88 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    98e2:	49 89 c4             	mov    %rax,%r12
    98e5:	48 89 c7             	mov    %rax,%rdi
    98e8:	31 f6                	xor    %esi,%esi
    98ea:	48 89 da             	mov    %rbx,%rdx
    98ed:	4c 89 f9             	mov    %r15,%rcx
    98f0:	e8 fb 87 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    98f5:	be 01 00 00 00       	mov    $0x1,%esi
    98fa:	4c 89 e3             	mov    %r12,%rbx
    98fd:	4c 89 64 24 40       	mov    %r12,0x40(%rsp)
    9902:	4c 89 e7             	mov    %r12,%rdi
    9905:	e8 96 87 ff ff       	call   20a0 <omTensorSetDataType@plt>
    990a:	4c 89 e7             	mov    %r12,%rdi
    990d:	e8 be 88 ff ff       	call   21d0 <omTensorGetShape@plt>
    9912:	49 89 c4             	mov    %rax,%r12
    9915:	48 89 df             	mov    %rbx,%rdi
    9918:	e8 53 87 ff ff       	call   2070 <omTensorGetStrides@plt>
    991d:	4d 89 34 24          	mov    %r14,(%r12)
    9921:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    9928:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    992f:	00 00 
    9931:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    9938:	00 
    9939:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    9940:	00 00 
    9942:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    9949:	00 
    994a:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    9951:	00 00 
    9953:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    995a:	00 
    995b:	bf 04 00 00 00       	mov    $0x4,%edi
    9960:	e8 3b 88 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    9965:	48 89 c3             	mov    %rax,%rbx
    9968:	48 89 c7             	mov    %rax,%rdi
    996b:	31 f6                	xor    %esi,%esi
    996d:	48 8b 14 24          	mov    (%rsp),%rdx
    9971:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    9976:	e8 75 87 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    997b:	be 01 00 00 00       	mov    $0x1,%esi
    9980:	48 89 df             	mov    %rbx,%rdi
    9983:	e8 18 87 ff ff       	call   20a0 <omTensorSetDataType@plt>
    9988:	48 89 df             	mov    %rbx,%rdi
    998b:	e8 40 88 ff ff       	call   21d0 <omTensorGetShape@plt>
    9990:	49 89 c7             	mov    %rax,%r15
    9993:	48 89 df             	mov    %rbx,%rdi
    9996:	e8 d5 86 ff ff       	call   2070 <omTensorGetStrides@plt>
    999b:	4d 89 37             	mov    %r14,(%r15)
    999e:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    99a5:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    99ac:	00 
    99ad:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    99b4:	00 
    99b5:	49 c7 47 10 06 00 00 	movq   $0x6,0x10(%r15)
    99bc:	00 
    99bd:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    99c4:	00 
    99c5:	49 c7 47 18 06 00 00 	movq   $0x6,0x18(%r15)
    99cc:	00 
    99cd:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    99d4:	00 
    99d5:	4c 8d 25 74 79 00 00 	lea    0x7974(%rip),%r12        # 11350 <constant_62_mobilenet>
    99dc:	bf 04 00 00 00       	mov    $0x4,%edi
    99e1:	e8 ba 87 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    99e6:	49 89 c7             	mov    %rax,%r15
    99e9:	48 89 c7             	mov    %rax,%rdi
    99ec:	31 f6                	xor    %esi,%esi
    99ee:	4c 89 e2             	mov    %r12,%rdx
    99f1:	4c 89 e1             	mov    %r12,%rcx
    99f4:	e8 f7 86 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    99f9:	be 01 00 00 00       	mov    $0x1,%esi
    99fe:	4c 89 ff             	mov    %r15,%rdi
    9a01:	e8 9a 86 ff ff       	call   20a0 <omTensorSetDataType@plt>
    9a06:	4c 89 ff             	mov    %r15,%rdi
    9a09:	e8 c2 87 ff ff       	call   21d0 <omTensorGetShape@plt>
    9a0e:	49 89 c4             	mov    %rax,%r12
    9a11:	4c 89 ff             	mov    %r15,%rdi
    9a14:	e8 57 86 ff ff       	call   2070 <omTensorGetStrides@plt>
    9a19:	49 c7 04 24 80 00 00 	movq   $0x80,(%r12)
    9a20:	00 
    9a21:	48 c7 00 09 00 00 00 	movq   $0x9,(%rax)
    9a28:	49 c7 44 24 08 01 00 	movq   $0x1,0x8(%r12)
    9a2f:	00 00 
    9a31:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    9a38:	00 
    9a39:	49 c7 44 24 10 03 00 	movq   $0x3,0x10(%r12)
    9a40:	00 00 
    9a42:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    9a49:	00 
    9a4a:	49 c7 44 24 18 03 00 	movq   $0x3,0x18(%r12)
    9a51:	00 00 
    9a53:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    9a5a:	00 
    9a5b:	4c 8d 2d 6e 0b 01 00 	lea    0x10b6e(%rip),%r13        # 1a5d0 <constant_45_mobilenet>
    9a62:	bf 01 00 00 00       	mov    $0x1,%edi
    9a67:	e8 34 87 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    9a6c:	49 89 c4             	mov    %rax,%r12
    9a6f:	48 89 c7             	mov    %rax,%rdi
    9a72:	31 f6                	xor    %esi,%esi
    9a74:	4c 89 ea             	mov    %r13,%rdx
    9a77:	4c 89 e9             	mov    %r13,%rcx
    9a7a:	e8 71 86 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    9a7f:	be 01 00 00 00       	mov    $0x1,%esi
    9a84:	4c 89 e7             	mov    %r12,%rdi
    9a87:	e8 14 86 ff ff       	call   20a0 <omTensorSetDataType@plt>
    9a8c:	4c 89 e7             	mov    %r12,%rdi
    9a8f:	e8 3c 87 ff ff       	call   21d0 <omTensorGetShape@plt>
    9a94:	49 89 c5             	mov    %rax,%r13
    9a97:	4c 89 e7             	mov    %r12,%rdi
    9a9a:	e8 d1 85 ff ff       	call   2070 <omTensorGetStrides@plt>
    9a9f:	49 c7 45 00 80 00 00 	movq   $0x80,0x0(%r13)
    9aa6:	00 
    9aa7:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    9aae:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    9ab3:	4c 89 ef             	mov    %r13,%rdi
    9ab6:	48 89 de             	mov    %rbx,%rsi
    9ab9:	4c 89 fa             	mov    %r15,%rdx
    9abc:	4c 89 e1             	mov    %r12,%rcx
    9abf:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    9ac5:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    9acb:	6a 01                	push   $0x1
    9acd:	6a 01                	push   $0x1
    9acf:	6a 01                	push   $0x1
    9ad1:	6a 01                	push   $0x1
    9ad3:	e8 88 86 ff ff       	call   2160 <Conv2D_bias@plt>
    9ad8:	48 83 c4 20          	add    $0x20,%rsp
    9adc:	4c 89 ef             	mov    %r13,%rdi
    9adf:	e8 cc 86 ff ff       	call   21b0 <omTensorDestroy@plt>
    9ae4:	48 89 df             	mov    %rbx,%rdi
    9ae7:	e8 c4 86 ff ff       	call   21b0 <omTensorDestroy@plt>
    9aec:	4c 89 ff             	mov    %r15,%rdi
    9aef:	e8 bc 86 ff ff       	call   21b0 <omTensorDestroy@plt>
    9af4:	4c 89 e7             	mov    %r12,%rdi
    9af7:	e8 b4 86 ff ff       	call   21b0 <omTensorDestroy@plt>
    9afc:	48 8b 3c 24          	mov    (%rsp),%rdi
    9b00:	e8 0b 86 ff ff       	call   2110 <free@plt>
    9b05:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    9b0a:	e8 c1 85 ff ff       	call   20d0 <malloc@plt>
    9b0f:	49 89 c7             	mov    %rax,%r15
    9b12:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    9b16:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    9b1b:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    9b1f:	4c 89 24 24          	mov    %r12,(%rsp)
    9b23:	bf 04 00 00 00       	mov    $0x4,%edi
    9b28:	e8 73 86 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    9b2d:	48 89 c3             	mov    %rax,%rbx
    9b30:	48 89 c7             	mov    %rax,%rdi
    9b33:	31 f6                	xor    %esi,%esi
    9b35:	4c 89 fa             	mov    %r15,%rdx
    9b38:	4c 89 e1             	mov    %r12,%rcx
    9b3b:	e8 b0 85 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    9b40:	be 01 00 00 00       	mov    $0x1,%esi
    9b45:	48 89 df             	mov    %rbx,%rdi
    9b48:	e8 53 85 ff ff       	call   20a0 <omTensorSetDataType@plt>
    9b4d:	48 89 df             	mov    %rbx,%rdi
    9b50:	e8 7b 86 ff ff       	call   21d0 <omTensorGetShape@plt>
    9b55:	49 89 c4             	mov    %rax,%r12
    9b58:	48 89 df             	mov    %rbx,%rdi
    9b5b:	e8 10 85 ff ff       	call   2070 <omTensorGetStrides@plt>
    9b60:	4d 89 34 24          	mov    %r14,(%r12)
    9b64:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    9b6b:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    9b72:	00 00 
    9b74:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    9b7b:	00 
    9b7c:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    9b83:	00 00 
    9b85:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    9b8c:	00 
    9b8d:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    9b94:	00 00 
    9b96:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    9b9d:	00 
    9b9e:	bf 04 00 00 00       	mov    $0x4,%edi
    9ba3:	e8 f8 85 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    9ba8:	49 89 c4             	mov    %rax,%r12
    9bab:	48 89 c7             	mov    %rax,%rdi
    9bae:	31 f6                	xor    %esi,%esi
    9bb0:	4c 8b 7c 24 30       	mov    0x30(%rsp),%r15
    9bb5:	4c 89 fa             	mov    %r15,%rdx
    9bb8:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    9bbd:	e8 2e 85 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    9bc2:	be 01 00 00 00       	mov    $0x1,%esi
    9bc7:	4c 89 e7             	mov    %r12,%rdi
    9bca:	e8 d1 84 ff ff       	call   20a0 <omTensorSetDataType@plt>
    9bcf:	4c 89 e7             	mov    %r12,%rdi
    9bd2:	e8 f9 85 ff ff       	call   21d0 <omTensorGetShape@plt>
    9bd7:	49 89 c5             	mov    %rax,%r13
    9bda:	4c 89 e7             	mov    %r12,%rdi
    9bdd:	e8 8e 84 ff ff       	call   2070 <omTensorGetStrides@plt>
    9be2:	4d 89 75 00          	mov    %r14,0x0(%r13)
    9be6:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    9bed:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    9bf4:	00 
    9bf5:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    9bfc:	00 
    9bfd:	49 c7 45 10 06 00 00 	movq   $0x6,0x10(%r13)
    9c04:	00 
    9c05:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    9c0c:	00 
    9c0d:	49 c7 45 18 06 00 00 	movq   $0x6,0x18(%r13)
    9c14:	00 
    9c15:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    9c1c:	00 
    9c1d:	48 89 df             	mov    %rbx,%rdi
    9c20:	4c 89 e6             	mov    %r12,%rsi
    9c23:	e8 c8 85 ff ff       	call   21f0 <Relu@plt>
    9c28:	48 89 df             	mov    %rbx,%rdi
    9c2b:	e8 80 85 ff ff       	call   21b0 <omTensorDestroy@plt>
    9c30:	4c 89 e7             	mov    %r12,%rdi
    9c33:	e8 78 85 ff ff       	call   21b0 <omTensorDestroy@plt>
    9c38:	4c 89 ff             	mov    %r15,%rdi
    9c3b:	e8 d0 84 ff ff       	call   2110 <free@plt>
    9c40:	bf 18 00 00 00       	mov    $0x18,%edi
    9c45:	e8 86 84 ff ff       	call   20d0 <malloc@plt>
    9c4a:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    9c4f:	48 83 c0 0f          	add    $0xf,%rax
    9c53:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    9c57:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    9c5c:	4c 89 30             	mov    %r14,(%rax)
    9c5f:	bf 28 00 00 00       	mov    $0x28,%edi
    9c64:	e8 67 84 ff ff       	call   20d0 <malloc@plt>
    9c69:	48 8d 48 0f          	lea    0xf(%rax),%rcx
    9c6d:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
    9c71:	4c 89 31             	mov    %r14,(%rcx)
    9c74:	48 c7 41 08 80 00 00 	movq   $0x80,0x8(%rcx)
    9c7b:	00 
    9c7c:	48 c7 41 10 ff ff ff 	movq   $0xffffffffffffffff,0x10(%rcx)
    9c83:	ff 
    9c84:	48 89 c7             	mov    %rax,%rdi
    9c87:	e8 84 84 ff ff       	call   2110 <free@plt>
    9c8c:	48 89 e9             	mov    %rbp,%rcx
    9c8f:	48 c1 e1 07          	shl    $0x7,%rcx
    9c93:	48 8b 5c 24 60       	mov    0x60(%rsp),%rbx
    9c98:	48 89 d8             	mov    %rbx,%rax
    9c9b:	48 0f af c1          	imul   %rcx,%rax
    9c9f:	49 89 cf             	mov    %rcx,%r15
    9ca2:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
    9ca7:	48 8d 3c 85 10 00 00 	lea    0x10(,%rax,4),%rdi
    9cae:	00 
    9caf:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
    9cb4:	e8 17 84 ff ff       	call   20d0 <malloc@plt>
    9cb9:	49 89 c4             	mov    %rax,%r12
    9cbc:	4c 8d 68 0f          	lea    0xf(%rax),%r13
    9cc0:	49 83 e5 f0          	and    $0xfffffffffffffff0,%r13
    9cc4:	bf 03 00 00 00       	mov    $0x3,%edi
    9cc9:	e8 d2 84 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    9cce:	49 89 c6             	mov    %rax,%r14
    9cd1:	48 89 c7             	mov    %rax,%rdi
    9cd4:	31 f6                	xor    %esi,%esi
    9cd6:	4c 89 64 24 70       	mov    %r12,0x70(%rsp)
    9cdb:	4c 89 e2             	mov    %r12,%rdx
    9cde:	4c 89 6c 24 08       	mov    %r13,0x8(%rsp)
    9ce3:	4c 89 e9             	mov    %r13,%rcx
    9ce6:	e8 05 84 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    9ceb:	be 01 00 00 00       	mov    $0x1,%esi
    9cf0:	4c 89 f7             	mov    %r14,%rdi
    9cf3:	e8 a8 83 ff ff       	call   20a0 <omTensorSetDataType@plt>
    9cf8:	4c 89 f7             	mov    %r14,%rdi
    9cfb:	e8 d0 84 ff ff       	call   21d0 <omTensorGetShape@plt>
    9d00:	49 89 c4             	mov    %rax,%r12
    9d03:	4c 89 f7             	mov    %r14,%rdi
    9d06:	e8 65 83 ff ff       	call   2070 <omTensorGetStrides@plt>
    9d0b:	49 89 1c 24          	mov    %rbx,(%r12)
    9d0f:	4c 89 38             	mov    %r15,(%rax)
    9d12:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    9d19:	00 00 
    9d1b:	48 89 68 08          	mov    %rbp,0x8(%rax)
    9d1f:	49 89 6c 24 10       	mov    %rbp,0x10(%r12)
    9d24:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    9d2b:	00 
    9d2c:	bf 02 00 00 00       	mov    $0x2,%edi
    9d31:	e8 6a 84 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    9d36:	49 89 c5             	mov    %rax,%r13
    9d39:	48 8d 0d d0 29 07 00 	lea    0x729d0(%rip),%rcx        # 7c710 <constant_21_mobilenet>
    9d40:	48 89 c7             	mov    %rax,%rdi
    9d43:	31 f6                	xor    %esi,%esi
    9d45:	48 89 ca             	mov    %rcx,%rdx
    9d48:	e8 a3 83 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    9d4d:	be 01 00 00 00       	mov    $0x1,%esi
    9d52:	4c 89 ef             	mov    %r13,%rdi
    9d55:	e8 46 83 ff ff       	call   20a0 <omTensorSetDataType@plt>
    9d5a:	4c 89 ef             	mov    %r13,%rdi
    9d5d:	e8 6e 84 ff ff       	call   21d0 <omTensorGetShape@plt>
    9d62:	49 89 c4             	mov    %rax,%r12
    9d65:	4c 89 ef             	mov    %r13,%rdi
    9d68:	e8 03 83 ff ff       	call   2070 <omTensorGetStrides@plt>
    9d6d:	49 c7 04 24 80 00 00 	movq   $0x80,(%r12)
    9d74:	00 
    9d75:	48 c7 00 80 00 00 00 	movq   $0x80,(%rax)
    9d7c:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    9d83:	00 00 
    9d85:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    9d8c:	00 
    9d8d:	bf 03 00 00 00       	mov    $0x3,%edi
    9d92:	e8 09 84 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    9d97:	49 89 c4             	mov    %rax,%r12
    9d9a:	48 89 c7             	mov    %rax,%rdi
    9d9d:	31 f6                	xor    %esi,%esi
    9d9f:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    9da4:	48 8b 0c 24          	mov    (%rsp),%rcx
    9da8:	e8 43 83 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    9dad:	be 01 00 00 00       	mov    $0x1,%esi
    9db2:	4c 89 e7             	mov    %r12,%rdi
    9db5:	e8 e6 82 ff ff       	call   20a0 <omTensorSetDataType@plt>
    9dba:	4c 89 e7             	mov    %r12,%rdi
    9dbd:	e8 0e 84 ff ff       	call   21d0 <omTensorGetShape@plt>
    9dc2:	49 89 c7             	mov    %rax,%r15
    9dc5:	4c 89 e7             	mov    %r12,%rdi
    9dc8:	e8 a3 82 ff ff       	call   2070 <omTensorGetStrides@plt>
    9dcd:	49 89 1f             	mov    %rbx,(%r15)
    9dd0:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    9dd5:	48 89 08             	mov    %rcx,(%rax)
    9dd8:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    9ddf:	00 
    9de0:	48 89 68 08          	mov    %rbp,0x8(%rax)
    9de4:	49 89 6f 10          	mov    %rbp,0x10(%r15)
    9de8:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    9def:	00 
    9df0:	4c 89 f7             	mov    %r14,%rdi
    9df3:	4c 89 ee             	mov    %r13,%rsi
    9df6:	4c 89 e2             	mov    %r12,%rdx
    9df9:	e8 52 82 ff ff       	call   2050 <MatMul@plt>
    9dfe:	4c 89 f7             	mov    %r14,%rdi
    9e01:	e8 aa 83 ff ff       	call   21b0 <omTensorDestroy@plt>
    9e06:	4c 89 ef             	mov    %r13,%rdi
    9e09:	e8 a2 83 ff ff       	call   21b0 <omTensorDestroy@plt>
    9e0e:	4c 89 e7             	mov    %r12,%rdi
    9e11:	e8 9a 83 ff ff       	call   21b0 <omTensorDestroy@plt>
    9e16:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    9e1b:	e8 f0 82 ff ff       	call   2110 <free@plt>
    9e20:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    9e25:	e8 a6 82 ff ff       	call   20d0 <malloc@plt>
    9e2a:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    9e2f:	48 83 c0 0f          	add    $0xf,%rax
    9e33:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    9e37:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    9e3c:	48 85 db             	test   %rbx,%rbx
    9e3f:	0f 8e 58 01 00 00    	jle    9f9d <main_graph_mobilenet+0x7ccd>
    9e45:	48 89 e8             	mov    %rbp,%rax
    9e48:	48 c1 e0 09          	shl    $0x9,%rax
    9e4c:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    9e51:	48 8d 0c ad 00 00 00 	lea    0x0(,%rbp,4),%rcx
    9e58:	00 
    9e59:	48 89 eb             	mov    %rbp,%rbx
    9e5c:	48 83 e3 f8          	and    $0xfffffffffffffff8,%rbx
    9e60:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    9e65:	49 8d 45 10          	lea    0x10(%r13),%rax
    9e69:	48 89 04 24          	mov    %rax,(%rsp)
    9e6d:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    9e72:	48 8d 50 10          	lea    0x10(%rax),%rdx
    9e76:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
    9e7b:	45 31 c0             	xor    %r8d,%r8d
    9e7e:	4c 8d 0d 8b 19 01 00 	lea    0x1198b(%rip),%r9        # 1b810 <constant_34_mobilenet>
    9e85:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    9e8a:	eb 28                	jmp    9eb4 <main_graph_mobilenet+0x7be4>
    9e8c:	0f 1f 40 00          	nopl   0x0(%rax)
    9e90:	49 ff c0             	inc    %r8
    9e93:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    9e98:	48 01 04 24          	add    %rax,(%rsp)
    9e9c:	48 01 44 24 20       	add    %rax,0x20(%rsp)
    9ea1:	49 01 c5             	add    %rax,%r13
    9ea4:	48 01 44 24 30       	add    %rax,0x30(%rsp)
    9ea9:	4c 3b 44 24 60       	cmp    0x60(%rsp),%r8
    9eae:	0f 84 e9 00 00 00    	je     9f9d <main_graph_mobilenet+0x7ccd>
    9eb4:	48 85 ed             	test   %rbp,%rbp
    9eb7:	7e d7                	jle    9e90 <main_graph_mobilenet+0x7bc0>
    9eb9:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
    9ebe:	4d 0f af f0          	imul   %r8,%r14
    9ec2:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    9ec7:	4e 8d 3c 30          	lea    (%rax,%r14,1),%r15
    9ecb:	4c 03 74 24 08       	add    0x8(%rsp),%r14
    9ed0:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
    9ed5:	4d 89 eb             	mov    %r13,%r11
    9ed8:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    9edd:	48 8b 34 24          	mov    (%rsp),%rsi
    9ee1:	31 c0                	xor    %eax,%eax
    9ee3:	eb 22                	jmp    9f07 <main_graph_mobilenet+0x7c37>
    9ee5:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    9eec:	00 00 00 
    9eef:	90                   	nop
    9ef0:	48 ff c0             	inc    %rax
    9ef3:	48 01 ce             	add    %rcx,%rsi
    9ef6:	48 01 cf             	add    %rcx,%rdi
    9ef9:	49 01 cb             	add    %rcx,%r11
    9efc:	49 01 cc             	add    %rcx,%r12
    9eff:	48 3d 80 00 00 00    	cmp    $0x80,%rax
    9f05:	74 89                	je     9e90 <main_graph_mobilenet+0x7bc0>
    9f07:	48 83 fd 08          	cmp    $0x8,%rbp
    9f0b:	f3 41 0f 10 04 81    	movss  (%r9,%rax,4),%xmm0
    9f11:	73 0d                	jae    9f20 <main_graph_mobilenet+0x7c50>
    9f13:	31 d2                	xor    %edx,%edx
    9f15:	eb 69                	jmp    9f80 <main_graph_mobilenet+0x7cb0>
    9f17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    9f1e:	00 00 
    9f20:	48 89 ca             	mov    %rcx,%rdx
    9f23:	48 0f af d0          	imul   %rax,%rdx
    9f27:	4d 8d 14 16          	lea    (%r14,%rdx,1),%r10
    9f2b:	4c 01 fa             	add    %r15,%rdx
    9f2e:	4c 29 d2             	sub    %r10,%rdx
    9f31:	48 83 fa 20          	cmp    $0x20,%rdx
    9f35:	73 09                	jae    9f40 <main_graph_mobilenet+0x7c70>
    9f37:	31 d2                	xor    %edx,%edx
    9f39:	eb 45                	jmp    9f80 <main_graph_mobilenet+0x7cb0>
    9f3b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    9f40:	0f 28 c8             	movaps %xmm0,%xmm1
    9f43:	0f c6 c8 00          	shufps $0x0,%xmm0,%xmm1
    9f47:	31 d2                	xor    %edx,%edx
    9f49:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    9f50:	0f 10 54 97 f0       	movups -0x10(%rdi,%rdx,4),%xmm2
    9f55:	0f 10 1c 97          	movups (%rdi,%rdx,4),%xmm3
    9f59:	0f 58 d1             	addps  %xmm1,%xmm2
    9f5c:	0f 58 d9             	addps  %xmm1,%xmm3
    9f5f:	0f 11 54 96 f0       	movups %xmm2,-0x10(%rsi,%rdx,4)
    9f64:	0f 11 1c 96          	movups %xmm3,(%rsi,%rdx,4)
    9f68:	48 83 c2 08          	add    $0x8,%rdx
    9f6c:	48 39 d3             	cmp    %rdx,%rbx
    9f6f:	75 df                	jne    9f50 <main_graph_mobilenet+0x7c80>
    9f71:	48 39 dd             	cmp    %rbx,%rbp
    9f74:	48 89 da             	mov    %rbx,%rdx
    9f77:	0f 84 73 ff ff ff    	je     9ef0 <main_graph_mobilenet+0x7c20>
    9f7d:	0f 1f 00             	nopl   (%rax)
    9f80:	f3 41 0f 10 0c 94    	movss  (%r12,%rdx,4),%xmm1
    9f86:	f3 0f 58 c8          	addss  %xmm0,%xmm1
    9f8a:	f3 41 0f 11 0c 93    	movss  %xmm1,(%r11,%rdx,4)
    9f90:	48 ff c2             	inc    %rdx
    9f93:	48 39 d5             	cmp    %rdx,%rbp
    9f96:	75 e8                	jne    9f80 <main_graph_mobilenet+0x7cb0>
    9f98:	e9 53 ff ff ff       	jmp    9ef0 <main_graph_mobilenet+0x7c20>
    9f9d:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    9fa2:	e8 69 81 ff ff       	call   2110 <free@plt>
    9fa7:	bf 30 00 00 00       	mov    $0x30,%edi
    9fac:	e8 1f 81 ff ff       	call   20d0 <malloc@plt>
    9fb1:	49 89 c4             	mov    %rax,%r12
    9fb4:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    9fb8:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    9fbc:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    9fc1:	48 8b 00             	mov    (%rax),%rax
    9fc4:	49 89 07             	mov    %rax,(%r15)
    9fc7:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    9fcc:	e8 3f 81 ff ff       	call   2110 <free@plt>
    9fd1:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    9fd6:	48 0f af e8          	imul   %rax,%rbp
    9fda:	48 c1 e5 07          	shl    $0x7,%rbp
    9fde:	49 8b 0f             	mov    (%r15),%rcx
    9fe1:	48 85 c9             	test   %rcx,%rcx
    9fe4:	48 0f 44 c8          	cmove  %rax,%rcx
    9fe8:	48 89 c8             	mov    %rcx,%rax
    9feb:	48 c1 e0 09          	shl    $0x9,%rax
    9fef:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    9ff3:	48 8d 34 c0          	lea    (%rax,%rax,8),%rsi
    9ff7:	b8 00 12 00 00       	mov    $0x1200,%eax
    9ffc:	48 0f 44 f0          	cmove  %rax,%rsi
    a000:	48 89 f0             	mov    %rsi,%rax
    a003:	48 f7 d0             	not    %rax
    a006:	48 c1 f8 3f          	sar    $0x3f,%rax
    a00a:	48 83 c8 01          	or     $0x1,%rax
    a00e:	48 29 e8             	sub    %rbp,%rax
    a011:	48 99                	cqto   
    a013:	48 f7 fe             	idiv   %rsi
    a016:	49 89 c6             	mov    %rax,%r14
    a019:	49 f7 d6             	not    %r14
    a01c:	48 89 e8             	mov    %rbp,%rax
    a01f:	48 99                	cqto   
    a021:	48 f7 fe             	idiv   %rsi
    a024:	48 85 f6             	test   %rsi,%rsi
    a027:	48 89 c2             	mov    %rax,%rdx
    a02a:	49 0f 48 d6          	cmovs  %r14,%rdx
    a02e:	48 85 ed             	test   %rbp,%rbp
    a031:	48 0f 4e d0          	cmovle %rax,%rdx
    a035:	48 85 f6             	test   %rsi,%rsi
    a038:	4c 0f 4e f2          	cmovle %rdx,%r14
    a03c:	48 85 ed             	test   %rbp,%rbp
    a03f:	4c 0f 49 f2          	cmovns %rdx,%r14
    a043:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    a047:	4c 0f 45 f1          	cmovne %rcx,%r14
    a04b:	4c 89 f6             	mov    %r14,%rsi
    a04e:	48 c1 e6 07          	shl    $0x7,%rsi
    a052:	48 8d 0c f6          	lea    (%rsi,%rsi,8),%rcx
    a056:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    a05a:	b8 80 00 00 00       	mov    $0x80,%eax
    a05f:	48 0f 44 f0          	cmove  %rax,%rsi
    a063:	48 89 f0             	mov    %rsi,%rax
    a066:	48 f7 d0             	not    %rax
    a069:	48 c1 f8 3f          	sar    $0x3f,%rax
    a06d:	48 83 c8 01          	or     $0x1,%rax
    a071:	48 29 c8             	sub    %rcx,%rax
    a074:	48 99                	cqto   
    a076:	48 f7 fe             	idiv   %rsi
    a079:	48 89 c5             	mov    %rax,%rbp
    a07c:	48 f7 d5             	not    %rbp
    a07f:	48 89 c8             	mov    %rcx,%rax
    a082:	48 99                	cqto   
    a084:	48 f7 fe             	idiv   %rsi
    a087:	48 85 f6             	test   %rsi,%rsi
    a08a:	48 89 c2             	mov    %rax,%rdx
    a08d:	48 0f 48 d5          	cmovs  %rbp,%rdx
    a091:	48 85 c9             	test   %rcx,%rcx
    a094:	48 0f 4e d0          	cmovle %rax,%rdx
    a098:	48 85 f6             	test   %rsi,%rsi
    a09b:	48 0f 4e ea          	cmovle %rdx,%rbp
    a09f:	48 85 c9             	test   %rcx,%rcx
    a0a2:	48 0f 49 ea          	cmovns %rdx,%rbp
    a0a6:	49 83 fe ff          	cmp    $0xffffffffffffffff,%r14
    a0aa:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    a0b1:	00 
    a0b2:	49 c7 47 10 06 00 00 	movq   $0x6,0x10(%r15)
    a0b9:	00 
    a0ba:	49 c7 47 18 06 00 00 	movq   $0x6,0x18(%r15)
    a0c1:	00 
    a0c2:	4c 89 f0             	mov    %r14,%rax
    a0c5:	48 0f 44 c5          	cmove  %rbp,%rax
    a0c9:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    a0ce:	4c 89 e7             	mov    %r12,%rdi
    a0d1:	e8 3a 80 ff ff       	call   2110 <free@plt>
    a0d6:	4c 89 f0             	mov    %r14,%rax
    a0d9:	48 c1 e0 0b          	shl    $0xb,%rax
    a0dd:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    a0e2:	e8 e9 7f ff ff       	call   20d0 <malloc@plt>
    a0e7:	49 89 c7             	mov    %rax,%r15
    a0ea:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    a0ee:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    a0f3:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    a0f7:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    a0fc:	bf 04 00 00 00       	mov    $0x4,%edi
    a101:	e8 9a 80 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    a106:	48 89 c3             	mov    %rax,%rbx
    a109:	48 89 c7             	mov    %rax,%rdi
    a10c:	31 f6                	xor    %esi,%esi
    a10e:	4c 89 fa             	mov    %r15,%rdx
    a111:	4c 89 e1             	mov    %r12,%rcx
    a114:	e8 d7 7f ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    a119:	be 01 00 00 00       	mov    $0x1,%esi
    a11e:	48 89 df             	mov    %rbx,%rdi
    a121:	e8 7a 7f ff ff       	call   20a0 <omTensorSetDataType@plt>
    a126:	48 89 df             	mov    %rbx,%rdi
    a129:	e8 a2 80 ff ff       	call   21d0 <omTensorGetShape@plt>
    a12e:	49 89 c7             	mov    %rax,%r15
    a131:	48 89 df             	mov    %rbx,%rdi
    a134:	e8 37 7f ff ff       	call   2070 <omTensorGetStrides@plt>
    a139:	4d 89 37             	mov    %r14,(%r15)
    a13c:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    a143:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    a14a:	00 
    a14b:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    a152:	00 
    a153:	49 c7 47 10 06 00 00 	movq   $0x6,0x10(%r15)
    a15a:	00 
    a15b:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    a162:	00 
    a163:	49 c7 47 18 06 00 00 	movq   $0x6,0x18(%r15)
    a16a:	00 
    a16b:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    a172:	00 
    a173:	bf 04 00 00 00       	mov    $0x4,%edi
    a178:	e8 23 80 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    a17d:	49 89 c7             	mov    %rax,%r15
    a180:	48 89 c7             	mov    %rax,%rdi
    a183:	31 f6                	xor    %esi,%esi
    a185:	4c 8b 64 24 50       	mov    0x50(%rsp),%r12
    a18a:	4c 89 e2             	mov    %r12,%rdx
    a18d:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    a192:	e8 59 7f ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    a197:	be 01 00 00 00       	mov    $0x1,%esi
    a19c:	4c 89 ff             	mov    %r15,%rdi
    a19f:	e8 fc 7e ff ff       	call   20a0 <omTensorSetDataType@plt>
    a1a4:	4c 89 ff             	mov    %r15,%rdi
    a1a7:	e8 24 80 ff ff       	call   21d0 <omTensorGetShape@plt>
    a1ac:	49 89 c5             	mov    %rax,%r13
    a1af:	4c 89 ff             	mov    %r15,%rdi
    a1b2:	e8 b9 7e ff ff       	call   2070 <omTensorGetStrides@plt>
    a1b7:	4d 89 75 00          	mov    %r14,0x0(%r13)
    a1bb:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    a1c2:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    a1c9:	00 
    a1ca:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    a1d1:	00 
    a1d2:	49 c7 45 10 06 00 00 	movq   $0x6,0x10(%r13)
    a1d9:	00 
    a1da:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    a1e1:	00 
    a1e2:	49 c7 45 18 06 00 00 	movq   $0x6,0x18(%r13)
    a1e9:	00 
    a1ea:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    a1f1:	00 
    a1f2:	48 89 df             	mov    %rbx,%rdi
    a1f5:	4c 89 fe             	mov    %r15,%rsi
    a1f8:	e8 f3 7f ff ff       	call   21f0 <Relu@plt>
    a1fd:	48 89 df             	mov    %rbx,%rdi
    a200:	e8 ab 7f ff ff       	call   21b0 <omTensorDestroy@plt>
    a205:	4c 89 ff             	mov    %r15,%rdi
    a208:	e8 a3 7f ff ff       	call   21b0 <omTensorDestroy@plt>
    a20d:	4c 89 e7             	mov    %r12,%rdi
    a210:	e8 fb 7e ff ff       	call   2110 <free@plt>
    a215:	4c 89 f0             	mov    %r14,%rax
    a218:	48 c1 e0 09          	shl    $0x9,%rax
    a21c:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    a221:	48 89 7c 24 30       	mov    %rdi,0x30(%rsp)
    a226:	e8 a5 7e ff ff       	call   20d0 <malloc@plt>
    a22b:	48 89 c3             	mov    %rax,%rbx
    a22e:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    a232:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    a237:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    a23b:	4c 89 3c 24          	mov    %r15,(%rsp)
    a23f:	bf 04 00 00 00       	mov    $0x4,%edi
    a244:	e8 57 7f ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    a249:	49 89 c4             	mov    %rax,%r12
    a24c:	48 89 c7             	mov    %rax,%rdi
    a24f:	31 f6                	xor    %esi,%esi
    a251:	48 89 da             	mov    %rbx,%rdx
    a254:	4c 89 f9             	mov    %r15,%rcx
    a257:	e8 94 7e ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    a25c:	be 01 00 00 00       	mov    $0x1,%esi
    a261:	4c 89 64 24 40       	mov    %r12,0x40(%rsp)
    a266:	4c 89 e7             	mov    %r12,%rdi
    a269:	e8 32 7e ff ff       	call   20a0 <omTensorSetDataType@plt>
    a26e:	4c 89 e7             	mov    %r12,%rdi
    a271:	e8 5a 7f ff ff       	call   21d0 <omTensorGetShape@plt>
    a276:	49 89 c5             	mov    %rax,%r13
    a279:	4c 89 e7             	mov    %r12,%rdi
    a27c:	e8 ef 7d ff ff       	call   2070 <omTensorGetStrides@plt>
    a281:	4d 89 75 00          	mov    %r14,0x0(%r13)
    a285:	48 c7 00 80 04 00 00 	movq   $0x480,(%rax)
    a28c:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    a293:	00 
    a294:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    a29b:	00 
    a29c:	49 c7 45 10 03 00 00 	movq   $0x3,0x10(%r13)
    a2a3:	00 
    a2a4:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    a2ab:	00 
    a2ac:	49 c7 45 18 03 00 00 	movq   $0x3,0x18(%r13)
    a2b3:	00 
    a2b4:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    a2bb:	00 
    a2bc:	bf 04 00 00 00       	mov    $0x4,%edi
    a2c1:	e8 da 7e ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    a2c6:	48 89 c3             	mov    %rax,%rbx
    a2c9:	48 89 c7             	mov    %rax,%rdi
    a2cc:	31 f6                	xor    %esi,%esi
    a2ce:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    a2d3:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    a2d8:	e8 13 7e ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    a2dd:	be 01 00 00 00       	mov    $0x1,%esi
    a2e2:	48 89 df             	mov    %rbx,%rdi
    a2e5:	e8 b6 7d ff ff       	call   20a0 <omTensorSetDataType@plt>
    a2ea:	48 89 df             	mov    %rbx,%rdi
    a2ed:	e8 de 7e ff ff       	call   21d0 <omTensorGetShape@plt>
    a2f2:	49 89 c4             	mov    %rax,%r12
    a2f5:	48 89 df             	mov    %rbx,%rdi
    a2f8:	e8 73 7d ff ff       	call   2070 <omTensorGetStrides@plt>
    a2fd:	4d 89 34 24          	mov    %r14,(%r12)
    a301:	48 c7 00 00 12 00 00 	movq   $0x1200,(%rax)
    a308:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    a30f:	00 00 
    a311:	48 c7 40 08 24 00 00 	movq   $0x24,0x8(%rax)
    a318:	00 
    a319:	49 c7 44 24 10 06 00 	movq   $0x6,0x10(%r12)
    a320:	00 00 
    a322:	48 c7 40 10 06 00 00 	movq   $0x6,0x10(%rax)
    a329:	00 
    a32a:	49 c7 44 24 18 06 00 	movq   $0x6,0x18(%r12)
    a331:	00 00 
    a333:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    a33a:	00 
    a33b:	4c 8d 2d 0e 5e 00 00 	lea    0x5e0e(%rip),%r13        # 10150 <constant_63_mobilenet>
    a342:	bf 04 00 00 00       	mov    $0x4,%edi
    a347:	e8 54 7e ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    a34c:	49 89 c4             	mov    %rax,%r12
    a34f:	48 89 c7             	mov    %rax,%rdi
    a352:	31 f6                	xor    %esi,%esi
    a354:	4c 89 ea             	mov    %r13,%rdx
    a357:	4c 89 e9             	mov    %r13,%rcx
    a35a:	e8 91 7d ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    a35f:	be 01 00 00 00       	mov    $0x1,%esi
    a364:	4c 89 e7             	mov    %r12,%rdi
    a367:	e8 34 7d ff ff       	call   20a0 <omTensorSetDataType@plt>
    a36c:	4c 89 e7             	mov    %r12,%rdi
    a36f:	e8 5c 7e ff ff       	call   21d0 <omTensorGetShape@plt>
    a374:	49 89 c5             	mov    %rax,%r13
    a377:	4c 89 e7             	mov    %r12,%rdi
    a37a:	e8 f1 7c ff ff       	call   2070 <omTensorGetStrides@plt>
    a37f:	49 c7 45 00 80 00 00 	movq   $0x80,0x0(%r13)
    a386:	00 
    a387:	48 c7 00 09 00 00 00 	movq   $0x9,(%rax)
    a38e:	49 c7 45 08 01 00 00 	movq   $0x1,0x8(%r13)
    a395:	00 
    a396:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    a39d:	00 
    a39e:	49 c7 45 10 03 00 00 	movq   $0x3,0x10(%r13)
    a3a5:	00 
    a3a6:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    a3ad:	00 
    a3ae:	49 c7 45 18 03 00 00 	movq   $0x3,0x18(%r13)
    a3b5:	00 
    a3b6:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    a3bd:	00 
    a3be:	4c 8d 3d 0b 04 01 00 	lea    0x1040b(%rip),%r15        # 1a7d0 <constant_44_mobilenet>
    a3c5:	bf 01 00 00 00       	mov    $0x1,%edi
    a3ca:	e8 d1 7d ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    a3cf:	49 89 c5             	mov    %rax,%r13
    a3d2:	48 89 c7             	mov    %rax,%rdi
    a3d5:	31 f6                	xor    %esi,%esi
    a3d7:	4c 89 fa             	mov    %r15,%rdx
    a3da:	4c 89 f9             	mov    %r15,%rcx
    a3dd:	e8 0e 7d ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    a3e2:	be 01 00 00 00       	mov    $0x1,%esi
    a3e7:	4c 89 ef             	mov    %r13,%rdi
    a3ea:	e8 b1 7c ff ff       	call   20a0 <omTensorSetDataType@plt>
    a3ef:	4c 89 ef             	mov    %r13,%rdi
    a3f2:	e8 d9 7d ff ff       	call   21d0 <omTensorGetShape@plt>
    a3f7:	49 89 c7             	mov    %rax,%r15
    a3fa:	4c 89 ef             	mov    %r13,%rdi
    a3fd:	e8 6e 7c ff ff       	call   2070 <omTensorGetStrides@plt>
    a402:	49 c7 07 80 00 00 00 	movq   $0x80,(%r15)
    a409:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    a410:	4c 8b 7c 24 40       	mov    0x40(%rsp),%r15
    a415:	4c 89 ff             	mov    %r15,%rdi
    a418:	48 89 de             	mov    %rbx,%rsi
    a41b:	4c 89 e2             	mov    %r12,%rdx
    a41e:	4c 89 e9             	mov    %r13,%rcx
    a421:	45 31 c0             	xor    %r8d,%r8d
    a424:	45 31 c9             	xor    %r9d,%r9d
    a427:	6a 02                	push   $0x2
    a429:	6a 02                	push   $0x2
    a42b:	6a 01                	push   $0x1
    a42d:	6a 01                	push   $0x1
    a42f:	e8 2c 7d ff ff       	call   2160 <Conv2D_bias@plt>
    a434:	48 83 c4 20          	add    $0x20,%rsp
    a438:	4c 89 ff             	mov    %r15,%rdi
    a43b:	e8 70 7d ff ff       	call   21b0 <omTensorDestroy@plt>
    a440:	48 89 df             	mov    %rbx,%rdi
    a443:	e8 68 7d ff ff       	call   21b0 <omTensorDestroy@plt>
    a448:	4c 89 e7             	mov    %r12,%rdi
    a44b:	e8 60 7d ff ff       	call   21b0 <omTensorDestroy@plt>
    a450:	4c 89 ef             	mov    %r13,%rdi
    a453:	e8 58 7d ff ff       	call   21b0 <omTensorDestroy@plt>
    a458:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    a45d:	e8 ae 7c ff ff       	call   2110 <free@plt>
    a462:	48 8b 7c 24 30       	mov    0x30(%rsp),%rdi
    a467:	e8 64 7c ff ff       	call   20d0 <malloc@plt>
    a46c:	49 89 c7             	mov    %rax,%r15
    a46f:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    a473:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    a478:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    a47c:	4c 89 64 24 30       	mov    %r12,0x30(%rsp)
    a481:	bf 04 00 00 00       	mov    $0x4,%edi
    a486:	e8 15 7d ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    a48b:	48 89 c3             	mov    %rax,%rbx
    a48e:	48 89 c7             	mov    %rax,%rdi
    a491:	31 f6                	xor    %esi,%esi
    a493:	4c 89 fa             	mov    %r15,%rdx
    a496:	4c 89 e1             	mov    %r12,%rcx
    a499:	e8 52 7c ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    a49e:	be 01 00 00 00       	mov    $0x1,%esi
    a4a3:	48 89 df             	mov    %rbx,%rdi
    a4a6:	e8 f5 7b ff ff       	call   20a0 <omTensorSetDataType@plt>
    a4ab:	48 89 df             	mov    %rbx,%rdi
    a4ae:	e8 1d 7d ff ff       	call   21d0 <omTensorGetShape@plt>
    a4b3:	49 89 c4             	mov    %rax,%r12
    a4b6:	48 89 df             	mov    %rbx,%rdi
    a4b9:	e8 b2 7b ff ff       	call   2070 <omTensorGetStrides@plt>
    a4be:	4d 89 34 24          	mov    %r14,(%r12)
    a4c2:	48 c7 00 80 04 00 00 	movq   $0x480,(%rax)
    a4c9:	49 c7 44 24 08 80 00 	movq   $0x80,0x8(%r12)
    a4d0:	00 00 
    a4d2:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    a4d9:	00 
    a4da:	49 c7 44 24 10 03 00 	movq   $0x3,0x10(%r12)
    a4e1:	00 00 
    a4e3:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    a4ea:	00 
    a4eb:	49 c7 44 24 18 03 00 	movq   $0x3,0x18(%r12)
    a4f2:	00 00 
    a4f4:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    a4fb:	00 
    a4fc:	bf 04 00 00 00       	mov    $0x4,%edi
    a501:	e8 9a 7c ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    a506:	49 89 c4             	mov    %rax,%r12
    a509:	48 89 c7             	mov    %rax,%rdi
    a50c:	31 f6                	xor    %esi,%esi
    a50e:	4c 8b 7c 24 20       	mov    0x20(%rsp),%r15
    a513:	4c 89 fa             	mov    %r15,%rdx
    a516:	48 8b 0c 24          	mov    (%rsp),%rcx
    a51a:	e8 d1 7b ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    a51f:	be 01 00 00 00       	mov    $0x1,%esi
    a524:	4c 89 e7             	mov    %r12,%rdi
    a527:	e8 74 7b ff ff       	call   20a0 <omTensorSetDataType@plt>
    a52c:	4c 89 e7             	mov    %r12,%rdi
    a52f:	e8 9c 7c ff ff       	call   21d0 <omTensorGetShape@plt>
    a534:	49 89 c5             	mov    %rax,%r13
    a537:	4c 89 e7             	mov    %r12,%rdi
    a53a:	e8 31 7b ff ff       	call   2070 <omTensorGetStrides@plt>
    a53f:	4d 89 75 00          	mov    %r14,0x0(%r13)
    a543:	48 c7 00 80 04 00 00 	movq   $0x480,(%rax)
    a54a:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    a551:	00 
    a552:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    a559:	00 
    a55a:	49 c7 45 10 03 00 00 	movq   $0x3,0x10(%r13)
    a561:	00 
    a562:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    a569:	00 
    a56a:	49 c7 45 18 03 00 00 	movq   $0x3,0x18(%r13)
    a571:	00 
    a572:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    a579:	00 
    a57a:	48 89 df             	mov    %rbx,%rdi
    a57d:	4c 89 e6             	mov    %r12,%rsi
    a580:	e8 6b 7c ff ff       	call   21f0 <Relu@plt>
    a585:	48 89 df             	mov    %rbx,%rdi
    a588:	e8 23 7c ff ff       	call   21b0 <omTensorDestroy@plt>
    a58d:	4c 89 e7             	mov    %r12,%rdi
    a590:	e8 1b 7c ff ff       	call   21b0 <omTensorDestroy@plt>
    a595:	4c 89 ff             	mov    %r15,%rdi
    a598:	e8 73 7b ff ff       	call   2110 <free@plt>
    a59d:	bf 18 00 00 00       	mov    $0x18,%edi
    a5a2:	e8 29 7b ff ff       	call   20d0 <malloc@plt>
    a5a7:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    a5ac:	48 83 c0 0f          	add    $0xf,%rax
    a5b0:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    a5b4:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    a5b9:	4c 89 30             	mov    %r14,(%rax)
    a5bc:	bf 28 00 00 00       	mov    $0x28,%edi
    a5c1:	e8 0a 7b ff ff       	call   20d0 <malloc@plt>
    a5c6:	48 8d 48 0f          	lea    0xf(%rax),%rcx
    a5ca:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
    a5ce:	4c 89 31             	mov    %r14,(%rcx)
    a5d1:	48 c7 41 08 80 00 00 	movq   $0x80,0x8(%rcx)
    a5d8:	00 
    a5d9:	48 c7 41 10 ff ff ff 	movq   $0xffffffffffffffff,0x10(%rcx)
    a5e0:	ff 
    a5e1:	48 89 c7             	mov    %rax,%rdi
    a5e4:	e8 27 7b ff ff       	call   2110 <free@plt>
    a5e9:	48 89 e8             	mov    %rbp,%rax
    a5ec:	48 c1 e0 07          	shl    $0x7,%rax
    a5f0:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    a5f5:	49 89 ed             	mov    %rbp,%r13
    a5f8:	49 c1 e5 08          	shl    $0x8,%r13
    a5fc:	48 8b 5c 24 60       	mov    0x60(%rsp),%rbx
    a601:	48 89 d8             	mov    %rbx,%rax
    a604:	49 0f af c5          	imul   %r13,%rax
    a608:	48 8d 3c 85 10 00 00 	lea    0x10(,%rax,4),%rdi
    a60f:	00 
    a610:	48 89 3c 24          	mov    %rdi,(%rsp)
    a614:	e8 b7 7a ff ff       	call   20d0 <malloc@plt>
    a619:	49 89 c4             	mov    %rax,%r12
    a61c:	4c 8d 78 0f          	lea    0xf(%rax),%r15
    a620:	49 83 e7 f0          	and    $0xfffffffffffffff0,%r15
    a624:	bf 03 00 00 00       	mov    $0x3,%edi
    a629:	e8 72 7b ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    a62e:	49 89 c6             	mov    %rax,%r14
    a631:	48 89 c7             	mov    %rax,%rdi
    a634:	31 f6                	xor    %esi,%esi
    a636:	4c 89 64 24 70       	mov    %r12,0x70(%rsp)
    a63b:	4c 89 e2             	mov    %r12,%rdx
    a63e:	4c 89 7c 24 08       	mov    %r15,0x8(%rsp)
    a643:	4c 89 f9             	mov    %r15,%rcx
    a646:	e8 a5 7a ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    a64b:	be 01 00 00 00       	mov    $0x1,%esi
    a650:	4c 89 f7             	mov    %r14,%rdi
    a653:	e8 48 7a ff ff       	call   20a0 <omTensorSetDataType@plt>
    a658:	4c 89 f7             	mov    %r14,%rdi
    a65b:	e8 70 7b ff ff       	call   21d0 <omTensorGetShape@plt>
    a660:	49 89 c4             	mov    %rax,%r12
    a663:	4c 89 f7             	mov    %r14,%rdi
    a666:	e8 05 7a ff ff       	call   2070 <omTensorGetStrides@plt>
    a66b:	49 89 1c 24          	mov    %rbx,(%r12)
    a66f:	4c 89 28             	mov    %r13,(%rax)
    a672:	49 c7 44 24 08 00 01 	movq   $0x100,0x8(%r12)
    a679:	00 00 
    a67b:	48 89 68 08          	mov    %rbp,0x8(%rax)
    a67f:	49 89 6c 24 10       	mov    %rbp,0x10(%r12)
    a684:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    a68b:	00 
    a68c:	bf 02 00 00 00       	mov    $0x2,%edi
    a691:	e8 0a 7b ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    a696:	49 89 c4             	mov    %rax,%r12
    a699:	48 8d 0d 70 20 05 00 	lea    0x52070(%rip),%rcx        # 5c710 <constant_22_mobilenet>
    a6a0:	48 89 c7             	mov    %rax,%rdi
    a6a3:	31 f6                	xor    %esi,%esi
    a6a5:	48 89 ca             	mov    %rcx,%rdx
    a6a8:	e8 43 7a ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    a6ad:	be 01 00 00 00       	mov    $0x1,%esi
    a6b2:	4c 89 e7             	mov    %r12,%rdi
    a6b5:	e8 e6 79 ff ff       	call   20a0 <omTensorSetDataType@plt>
    a6ba:	4c 89 e7             	mov    %r12,%rdi
    a6bd:	e8 0e 7b ff ff       	call   21d0 <omTensorGetShape@plt>
    a6c2:	49 89 c5             	mov    %rax,%r13
    a6c5:	4c 89 e7             	mov    %r12,%rdi
    a6c8:	e8 a3 79 ff ff       	call   2070 <omTensorGetStrides@plt>
    a6cd:	49 c7 45 00 00 01 00 	movq   $0x100,0x0(%r13)
    a6d4:	00 
    a6d5:	48 c7 00 80 00 00 00 	movq   $0x80,(%rax)
    a6dc:	49 c7 45 08 80 00 00 	movq   $0x80,0x8(%r13)
    a6e3:	00 
    a6e4:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    a6eb:	00 
    a6ec:	bf 03 00 00 00       	mov    $0x3,%edi
    a6f1:	e8 aa 7a ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    a6f6:	49 89 c5             	mov    %rax,%r13
    a6f9:	48 89 c7             	mov    %rax,%rdi
    a6fc:	31 f6                	xor    %esi,%esi
    a6fe:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    a703:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    a708:	e8 e3 79 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    a70d:	be 01 00 00 00       	mov    $0x1,%esi
    a712:	4c 89 ef             	mov    %r13,%rdi
    a715:	e8 86 79 ff ff       	call   20a0 <omTensorSetDataType@plt>
    a71a:	4c 89 ef             	mov    %r13,%rdi
    a71d:	e8 ae 7a ff ff       	call   21d0 <omTensorGetShape@plt>
    a722:	49 89 c7             	mov    %rax,%r15
    a725:	4c 89 ef             	mov    %r13,%rdi
    a728:	e8 43 79 ff ff       	call   2070 <omTensorGetStrides@plt>
    a72d:	49 89 1f             	mov    %rbx,(%r15)
    a730:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    a735:	48 89 08             	mov    %rcx,(%rax)
    a738:	49 c7 47 08 80 00 00 	movq   $0x80,0x8(%r15)
    a73f:	00 
    a740:	48 89 68 08          	mov    %rbp,0x8(%rax)
    a744:	49 89 6f 10          	mov    %rbp,0x10(%r15)
    a748:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    a74f:	00 
    a750:	4c 89 f7             	mov    %r14,%rdi
    a753:	4c 89 e6             	mov    %r12,%rsi
    a756:	4c 89 ea             	mov    %r13,%rdx
    a759:	e8 f2 78 ff ff       	call   2050 <MatMul@plt>
    a75e:	4c 89 f7             	mov    %r14,%rdi
    a761:	e8 4a 7a ff ff       	call   21b0 <omTensorDestroy@plt>
    a766:	4c 89 e7             	mov    %r12,%rdi
    a769:	e8 42 7a ff ff       	call   21b0 <omTensorDestroy@plt>
    a76e:	4c 89 ef             	mov    %r13,%rdi
    a771:	e8 3a 7a ff ff       	call   21b0 <omTensorDestroy@plt>
    a776:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    a77b:	e8 90 79 ff ff       	call   2110 <free@plt>
    a780:	48 8b 3c 24          	mov    (%rsp),%rdi
    a784:	e8 47 79 ff ff       	call   20d0 <malloc@plt>
    a789:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    a78e:	48 83 c0 0f          	add    $0xf,%rax
    a792:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    a796:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    a79b:	48 85 db             	test   %rbx,%rbx
    a79e:	0f 8e 59 01 00 00    	jle    a8fd <main_graph_mobilenet+0x862d>
    a7a4:	48 89 e8             	mov    %rbp,%rax
    a7a7:	48 c1 e0 0a          	shl    $0xa,%rax
    a7ab:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    a7b0:	48 8d 0c ad 00 00 00 	lea    0x0(,%rbp,4),%rcx
    a7b7:	00 
    a7b8:	48 89 eb             	mov    %rbp,%rbx
    a7bb:	48 83 e3 f8          	and    $0xfffffffffffffff8,%rbx
    a7bf:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    a7c4:	49 8d 45 10          	lea    0x10(%r13),%rax
    a7c8:	48 89 04 24          	mov    %rax,(%rsp)
    a7cc:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    a7d1:	48 8d 50 10          	lea    0x10(%rax),%rdx
    a7d5:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
    a7da:	45 31 c0             	xor    %r8d,%r8d
    a7dd:	4c 8d 0d 2c 0c 01 00 	lea    0x10c2c(%rip),%r9        # 1b410 <constant_35_mobilenet>
    a7e4:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    a7e9:	eb 29                	jmp    a814 <main_graph_mobilenet+0x8544>
    a7eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    a7f0:	49 ff c0             	inc    %r8
    a7f3:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    a7f8:	48 01 04 24          	add    %rax,(%rsp)
    a7fc:	48 01 44 24 20       	add    %rax,0x20(%rsp)
    a801:	49 01 c5             	add    %rax,%r13
    a804:	48 01 44 24 30       	add    %rax,0x30(%rsp)
    a809:	4c 3b 44 24 60       	cmp    0x60(%rsp),%r8
    a80e:	0f 84 e9 00 00 00    	je     a8fd <main_graph_mobilenet+0x862d>
    a814:	48 85 ed             	test   %rbp,%rbp
    a817:	7e d7                	jle    a7f0 <main_graph_mobilenet+0x8520>
    a819:	4c 8b 74 24 10       	mov    0x10(%rsp),%r14
    a81e:	4d 0f af f0          	imul   %r8,%r14
    a822:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    a827:	4e 8d 3c 30          	lea    (%rax,%r14,1),%r15
    a82b:	4c 03 74 24 08       	add    0x8(%rsp),%r14
    a830:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
    a835:	4d 89 eb             	mov    %r13,%r11
    a838:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    a83d:	48 8b 34 24          	mov    (%rsp),%rsi
    a841:	31 c0                	xor    %eax,%eax
    a843:	eb 22                	jmp    a867 <main_graph_mobilenet+0x8597>
    a845:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    a84c:	00 00 00 
    a84f:	90                   	nop
    a850:	48 ff c0             	inc    %rax
    a853:	48 01 ce             	add    %rcx,%rsi
    a856:	48 01 cf             	add    %rcx,%rdi
    a859:	49 01 cb             	add    %rcx,%r11
    a85c:	49 01 cc             	add    %rcx,%r12
    a85f:	48 3d 00 01 00 00    	cmp    $0x100,%rax
    a865:	74 89                	je     a7f0 <main_graph_mobilenet+0x8520>
    a867:	48 83 fd 08          	cmp    $0x8,%rbp
    a86b:	f3 41 0f 10 04 81    	movss  (%r9,%rax,4),%xmm0
    a871:	73 0d                	jae    a880 <main_graph_mobilenet+0x85b0>
    a873:	31 d2                	xor    %edx,%edx
    a875:	eb 69                	jmp    a8e0 <main_graph_mobilenet+0x8610>
    a877:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    a87e:	00 00 
    a880:	48 89 ca             	mov    %rcx,%rdx
    a883:	48 0f af d0          	imul   %rax,%rdx
    a887:	4d 8d 14 16          	lea    (%r14,%rdx,1),%r10
    a88b:	4c 01 fa             	add    %r15,%rdx
    a88e:	4c 29 d2             	sub    %r10,%rdx
    a891:	48 83 fa 20          	cmp    $0x20,%rdx
    a895:	73 09                	jae    a8a0 <main_graph_mobilenet+0x85d0>
    a897:	31 d2                	xor    %edx,%edx
    a899:	eb 45                	jmp    a8e0 <main_graph_mobilenet+0x8610>
    a89b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    a8a0:	0f 28 c8             	movaps %xmm0,%xmm1
    a8a3:	0f c6 c8 00          	shufps $0x0,%xmm0,%xmm1
    a8a7:	31 d2                	xor    %edx,%edx
    a8a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    a8b0:	0f 10 54 97 f0       	movups -0x10(%rdi,%rdx,4),%xmm2
    a8b5:	0f 10 1c 97          	movups (%rdi,%rdx,4),%xmm3
    a8b9:	0f 58 d1             	addps  %xmm1,%xmm2
    a8bc:	0f 58 d9             	addps  %xmm1,%xmm3
    a8bf:	0f 11 54 96 f0       	movups %xmm2,-0x10(%rsi,%rdx,4)
    a8c4:	0f 11 1c 96          	movups %xmm3,(%rsi,%rdx,4)
    a8c8:	48 83 c2 08          	add    $0x8,%rdx
    a8cc:	48 39 d3             	cmp    %rdx,%rbx
    a8cf:	75 df                	jne    a8b0 <main_graph_mobilenet+0x85e0>
    a8d1:	48 39 dd             	cmp    %rbx,%rbp
    a8d4:	48 89 da             	mov    %rbx,%rdx
    a8d7:	0f 84 73 ff ff ff    	je     a850 <main_graph_mobilenet+0x8580>
    a8dd:	0f 1f 00             	nopl   (%rax)
    a8e0:	f3 41 0f 10 0c 94    	movss  (%r12,%rdx,4),%xmm1
    a8e6:	f3 0f 58 c8          	addss  %xmm0,%xmm1
    a8ea:	f3 41 0f 11 0c 93    	movss  %xmm1,(%r11,%rdx,4)
    a8f0:	48 ff c2             	inc    %rdx
    a8f3:	48 39 d5             	cmp    %rdx,%rbp
    a8f6:	75 e8                	jne    a8e0 <main_graph_mobilenet+0x8610>
    a8f8:	e9 53 ff ff ff       	jmp    a850 <main_graph_mobilenet+0x8580>
    a8fd:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    a902:	e8 09 78 ff ff       	call   2110 <free@plt>
    a907:	bf 30 00 00 00       	mov    $0x30,%edi
    a90c:	e8 bf 77 ff ff       	call   20d0 <malloc@plt>
    a911:	49 89 c4             	mov    %rax,%r12
    a914:	48 8d 58 0f          	lea    0xf(%rax),%rbx
    a918:	48 83 e3 f0          	and    $0xfffffffffffffff0,%rbx
    a91c:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
    a921:	48 8b 00             	mov    (%rax),%rax
    a924:	48 89 03             	mov    %rax,(%rbx)
    a927:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
    a92c:	e8 df 77 ff ff       	call   2110 <free@plt>
    a931:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    a936:	48 0f af e8          	imul   %rax,%rbp
    a93a:	48 c1 e5 08          	shl    $0x8,%rbp
    a93e:	48 8b 0b             	mov    (%rbx),%rcx
    a941:	48 85 c9             	test   %rcx,%rcx
    a944:	48 0f 44 c8          	cmove  %rax,%rcx
    a948:	48 89 c8             	mov    %rcx,%rax
    a94b:	48 c1 e0 08          	shl    $0x8,%rax
    a94f:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    a953:	48 8d 34 c0          	lea    (%rax,%rax,8),%rsi
    a957:	b8 00 09 00 00       	mov    $0x900,%eax
    a95c:	48 0f 44 f0          	cmove  %rax,%rsi
    a960:	48 89 f0             	mov    %rsi,%rax
    a963:	48 f7 d0             	not    %rax
    a966:	48 c1 f8 3f          	sar    $0x3f,%rax
    a96a:	48 83 c8 01          	or     $0x1,%rax
    a96e:	48 29 e8             	sub    %rbp,%rax
    a971:	48 99                	cqto   
    a973:	48 f7 fe             	idiv   %rsi
    a976:	49 89 c7             	mov    %rax,%r15
    a979:	49 f7 d7             	not    %r15
    a97c:	48 89 e8             	mov    %rbp,%rax
    a97f:	48 99                	cqto   
    a981:	48 f7 fe             	idiv   %rsi
    a984:	48 85 f6             	test   %rsi,%rsi
    a987:	48 89 c2             	mov    %rax,%rdx
    a98a:	49 0f 48 d7          	cmovs  %r15,%rdx
    a98e:	48 85 ed             	test   %rbp,%rbp
    a991:	48 0f 4e d0          	cmovle %rax,%rdx
    a995:	48 85 f6             	test   %rsi,%rsi
    a998:	4c 0f 4e fa          	cmovle %rdx,%r15
    a99c:	48 85 ed             	test   %rbp,%rbp
    a99f:	4c 0f 49 fa          	cmovns %rdx,%r15
    a9a3:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    a9a7:	4c 0f 45 f9          	cmovne %rcx,%r15
    a9ab:	4c 89 f8             	mov    %r15,%rax
    a9ae:	48 c1 e0 08          	shl    $0x8,%rax
    a9b2:	48 8d 0c c0          	lea    (%rax,%rax,8),%rcx
    a9b6:	49 83 ff ff          	cmp    $0xffffffffffffffff,%r15
    a9ba:	be 00 01 00 00       	mov    $0x100,%esi
    a9bf:	48 0f 45 f0          	cmovne %rax,%rsi
    a9c3:	48 89 f0             	mov    %rsi,%rax
    a9c6:	48 f7 d0             	not    %rax
    a9c9:	48 c1 f8 3f          	sar    $0x3f,%rax
    a9cd:	48 83 c8 01          	or     $0x1,%rax
    a9d1:	48 29 c8             	sub    %rcx,%rax
    a9d4:	48 99                	cqto   
    a9d6:	48 f7 fe             	idiv   %rsi
    a9d9:	48 89 c5             	mov    %rax,%rbp
    a9dc:	48 f7 d5             	not    %rbp
    a9df:	48 89 c8             	mov    %rcx,%rax
    a9e2:	48 99                	cqto   
    a9e4:	48 f7 fe             	idiv   %rsi
    a9e7:	48 85 f6             	test   %rsi,%rsi
    a9ea:	48 89 c2             	mov    %rax,%rdx
    a9ed:	48 0f 48 d5          	cmovs  %rbp,%rdx
    a9f1:	48 85 c9             	test   %rcx,%rcx
    a9f4:	48 0f 4e d0          	cmovle %rax,%rdx
    a9f8:	48 85 f6             	test   %rsi,%rsi
    a9fb:	48 0f 4e ea          	cmovle %rdx,%rbp
    a9ff:	48 85 c9             	test   %rcx,%rcx
    aa02:	48 0f 49 ea          	cmovns %rdx,%rbp
    aa06:	49 83 ff ff          	cmp    $0xffffffffffffffff,%r15
    aa0a:	48 c7 43 08 00 01 00 	movq   $0x100,0x8(%rbx)
    aa11:	00 
    aa12:	48 c7 43 10 03 00 00 	movq   $0x3,0x10(%rbx)
    aa19:	00 
    aa1a:	48 c7 43 18 03 00 00 	movq   $0x3,0x18(%rbx)
    aa21:	00 
    aa22:	4c 89 f8             	mov    %r15,%rax
    aa25:	48 0f 44 c5          	cmove  %rbp,%rax
    aa29:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    aa2e:	4c 89 e7             	mov    %r12,%rdi
    aa31:	e8 da 76 ff ff       	call   2110 <free@plt>
    aa36:	4c 89 f8             	mov    %r15,%rax
    aa39:	48 c1 e0 0a          	shl    $0xa,%rax
    aa3d:	48 8d 7c c0 10       	lea    0x10(%rax,%rax,8),%rdi
    aa42:	48 89 7c 24 10       	mov    %rdi,0x10(%rsp)
    aa47:	e8 84 76 ff ff       	call   20d0 <malloc@plt>
    aa4c:	49 89 c6             	mov    %rax,%r14
    aa4f:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    aa53:	48 89 04 24          	mov    %rax,(%rsp)
    aa57:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    aa5b:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    aa60:	bf 04 00 00 00       	mov    $0x4,%edi
    aa65:	e8 36 77 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    aa6a:	48 89 c3             	mov    %rax,%rbx
    aa6d:	48 89 c7             	mov    %rax,%rdi
    aa70:	31 f6                	xor    %esi,%esi
    aa72:	4c 89 f2             	mov    %r14,%rdx
    aa75:	4c 89 e1             	mov    %r12,%rcx
    aa78:	e8 73 76 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    aa7d:	be 01 00 00 00       	mov    $0x1,%esi
    aa82:	48 89 df             	mov    %rbx,%rdi
    aa85:	e8 16 76 ff ff       	call   20a0 <omTensorSetDataType@plt>
    aa8a:	48 89 df             	mov    %rbx,%rdi
    aa8d:	e8 3e 77 ff ff       	call   21d0 <omTensorGetShape@plt>
    aa92:	49 89 c6             	mov    %rax,%r14
    aa95:	48 89 df             	mov    %rbx,%rdi
    aa98:	e8 d3 75 ff ff       	call   2070 <omTensorGetStrides@plt>
    aa9d:	4d 89 3e             	mov    %r15,(%r14)
    aaa0:	48 c7 00 00 09 00 00 	movq   $0x900,(%rax)
    aaa7:	49 c7 46 08 00 01 00 	movq   $0x100,0x8(%r14)
    aaae:	00 
    aaaf:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    aab6:	00 
    aab7:	49 c7 46 10 03 00 00 	movq   $0x3,0x10(%r14)
    aabe:	00 
    aabf:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    aac6:	00 
    aac7:	49 c7 46 18 03 00 00 	movq   $0x3,0x18(%r14)
    aace:	00 
    aacf:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    aad6:	00 
    aad7:	bf 04 00 00 00       	mov    $0x4,%edi
    aadc:	e8 bf 76 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    aae1:	49 89 c6             	mov    %rax,%r14
    aae4:	48 89 c7             	mov    %rax,%rdi
    aae7:	31 f6                	xor    %esi,%esi
    aae9:	4c 8b 64 24 50       	mov    0x50(%rsp),%r12
    aaee:	4c 89 e2             	mov    %r12,%rdx
    aaf1:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    aaf6:	e8 f5 75 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    aafb:	be 01 00 00 00       	mov    $0x1,%esi
    ab00:	4c 89 f7             	mov    %r14,%rdi
    ab03:	e8 98 75 ff ff       	call   20a0 <omTensorSetDataType@plt>
    ab08:	4c 89 f7             	mov    %r14,%rdi
    ab0b:	e8 c0 76 ff ff       	call   21d0 <omTensorGetShape@plt>
    ab10:	49 89 c5             	mov    %rax,%r13
    ab13:	4c 89 f7             	mov    %r14,%rdi
    ab16:	e8 55 75 ff ff       	call   2070 <omTensorGetStrides@plt>
    ab1b:	4d 89 7d 00          	mov    %r15,0x0(%r13)
    ab1f:	48 c7 00 00 09 00 00 	movq   $0x900,(%rax)
    ab26:	49 c7 45 08 00 01 00 	movq   $0x100,0x8(%r13)
    ab2d:	00 
    ab2e:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    ab35:	00 
    ab36:	49 c7 45 10 03 00 00 	movq   $0x3,0x10(%r13)
    ab3d:	00 
    ab3e:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    ab45:	00 
    ab46:	49 c7 45 18 03 00 00 	movq   $0x3,0x18(%r13)
    ab4d:	00 
    ab4e:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    ab55:	00 
    ab56:	48 89 df             	mov    %rbx,%rdi
    ab59:	4c 89 f6             	mov    %r14,%rsi
    ab5c:	e8 8f 76 ff ff       	call   21f0 <Relu@plt>
    ab61:	48 89 df             	mov    %rbx,%rdi
    ab64:	e8 47 76 ff ff       	call   21b0 <omTensorDestroy@plt>
    ab69:	4c 89 f7             	mov    %r14,%rdi
    ab6c:	e8 3f 76 ff ff       	call   21b0 <omTensorDestroy@plt>
    ab71:	4c 89 e7             	mov    %r12,%rdi
    ab74:	e8 97 75 ff ff       	call   2110 <free@plt>
    ab79:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    ab7e:	e8 4d 75 ff ff       	call   20d0 <malloc@plt>
    ab83:	48 89 c3             	mov    %rax,%rbx
    ab86:	4c 8d 70 0f          	lea    0xf(%rax),%r14
    ab8a:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    ab8f:	49 83 e6 f0          	and    $0xfffffffffffffff0,%r14
    ab93:	4c 89 74 24 20       	mov    %r14,0x20(%rsp)
    ab98:	bf 04 00 00 00       	mov    $0x4,%edi
    ab9d:	e8 fe 75 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    aba2:	49 89 c4             	mov    %rax,%r12
    aba5:	48 89 c7             	mov    %rax,%rdi
    aba8:	31 f6                	xor    %esi,%esi
    abaa:	48 89 da             	mov    %rbx,%rdx
    abad:	4c 89 f1             	mov    %r14,%rcx
    abb0:	e8 3b 75 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    abb5:	be 01 00 00 00       	mov    $0x1,%esi
    abba:	4c 89 64 24 40       	mov    %r12,0x40(%rsp)
    abbf:	4c 89 e7             	mov    %r12,%rdi
    abc2:	e8 d9 74 ff ff       	call   20a0 <omTensorSetDataType@plt>
    abc7:	4c 89 e7             	mov    %r12,%rdi
    abca:	e8 01 76 ff ff       	call   21d0 <omTensorGetShape@plt>
    abcf:	48 89 c3             	mov    %rax,%rbx
    abd2:	4c 89 e7             	mov    %r12,%rdi
    abd5:	e8 96 74 ff ff       	call   2070 <omTensorGetStrides@plt>
    abda:	4c 89 3b             	mov    %r15,(%rbx)
    abdd:	48 c7 00 00 09 00 00 	movq   $0x900,(%rax)
    abe4:	48 c7 43 08 00 01 00 	movq   $0x100,0x8(%rbx)
    abeb:	00 
    abec:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    abf3:	00 
    abf4:	48 c7 43 10 03 00 00 	movq   $0x3,0x10(%rbx)
    abfb:	00 
    abfc:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    ac03:	00 
    ac04:	48 c7 43 18 03 00 00 	movq   $0x3,0x18(%rbx)
    ac0b:	00 
    ac0c:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    ac13:	00 
    ac14:	bf 04 00 00 00       	mov    $0x4,%edi
    ac19:	e8 82 75 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    ac1e:	49 89 c6             	mov    %rax,%r14
    ac21:	48 89 c7             	mov    %rax,%rdi
    ac24:	31 f6                	xor    %esi,%esi
    ac26:	48 8b 14 24          	mov    (%rsp),%rdx
    ac2a:	48 8b 4c 24 08       	mov    0x8(%rsp),%rcx
    ac2f:	e8 bc 74 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    ac34:	be 01 00 00 00       	mov    $0x1,%esi
    ac39:	4c 89 f7             	mov    %r14,%rdi
    ac3c:	e8 5f 74 ff ff       	call   20a0 <omTensorSetDataType@plt>
    ac41:	4c 89 f7             	mov    %r14,%rdi
    ac44:	e8 87 75 ff ff       	call   21d0 <omTensorGetShape@plt>
    ac49:	48 89 c3             	mov    %rax,%rbx
    ac4c:	4c 89 f7             	mov    %r14,%rdi
    ac4f:	e8 1c 74 ff ff       	call   2070 <omTensorGetStrides@plt>
    ac54:	4c 89 3b             	mov    %r15,(%rbx)
    ac57:	48 c7 00 00 09 00 00 	movq   $0x900,(%rax)
    ac5e:	48 c7 43 08 00 01 00 	movq   $0x100,0x8(%rbx)
    ac65:	00 
    ac66:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    ac6d:	00 
    ac6e:	48 c7 43 10 03 00 00 	movq   $0x3,0x10(%rbx)
    ac75:	00 
    ac76:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    ac7d:	00 
    ac7e:	48 c7 43 18 03 00 00 	movq   $0x3,0x18(%rbx)
    ac85:	00 
    ac86:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    ac8d:	00 
    ac8e:	48 8d 1d 1b c4 00 00 	lea    0xc41b(%rip),%rbx        # 170b0 <constant_56_mobilenet>
    ac95:	bf 04 00 00 00       	mov    $0x4,%edi
    ac9a:	e8 01 75 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    ac9f:	49 89 c4             	mov    %rax,%r12
    aca2:	48 89 c7             	mov    %rax,%rdi
    aca5:	31 f6                	xor    %esi,%esi
    aca7:	48 89 da             	mov    %rbx,%rdx
    acaa:	48 89 d9             	mov    %rbx,%rcx
    acad:	e8 3e 74 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    acb2:	be 01 00 00 00       	mov    $0x1,%esi
    acb7:	4c 89 e7             	mov    %r12,%rdi
    acba:	e8 e1 73 ff ff       	call   20a0 <omTensorSetDataType@plt>
    acbf:	4c 89 e7             	mov    %r12,%rdi
    acc2:	e8 09 75 ff ff       	call   21d0 <omTensorGetShape@plt>
    acc7:	48 89 c3             	mov    %rax,%rbx
    acca:	4c 89 e7             	mov    %r12,%rdi
    accd:	e8 9e 73 ff ff       	call   2070 <omTensorGetStrides@plt>
    acd2:	48 c7 03 00 01 00 00 	movq   $0x100,(%rbx)
    acd9:	48 c7 00 09 00 00 00 	movq   $0x9,(%rax)
    ace0:	48 c7 43 08 01 00 00 	movq   $0x1,0x8(%rbx)
    ace7:	00 
    ace8:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    acef:	00 
    acf0:	48 c7 43 10 03 00 00 	movq   $0x3,0x10(%rbx)
    acf7:	00 
    acf8:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    acff:	00 
    ad00:	48 c7 43 18 03 00 00 	movq   $0x3,0x18(%rbx)
    ad07:	00 
    ad08:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    ad0f:	00 
    ad10:	4c 8d 2d b9 fc 00 00 	lea    0xfcb9(%rip),%r13        # 1a9d0 <constant_43_mobilenet>
    ad17:	bf 01 00 00 00       	mov    $0x1,%edi
    ad1c:	e8 7f 74 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    ad21:	48 89 c3             	mov    %rax,%rbx
    ad24:	48 89 c7             	mov    %rax,%rdi
    ad27:	31 f6                	xor    %esi,%esi
    ad29:	4c 89 ea             	mov    %r13,%rdx
    ad2c:	4c 89 e9             	mov    %r13,%rcx
    ad2f:	e8 bc 73 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    ad34:	be 01 00 00 00       	mov    $0x1,%esi
    ad39:	48 89 df             	mov    %rbx,%rdi
    ad3c:	e8 5f 73 ff ff       	call   20a0 <omTensorSetDataType@plt>
    ad41:	48 89 df             	mov    %rbx,%rdi
    ad44:	e8 87 74 ff ff       	call   21d0 <omTensorGetShape@plt>
    ad49:	49 89 c5             	mov    %rax,%r13
    ad4c:	48 89 df             	mov    %rbx,%rdi
    ad4f:	e8 1c 73 ff ff       	call   2070 <omTensorGetStrides@plt>
    ad54:	49 c7 45 00 00 01 00 	movq   $0x100,0x0(%r13)
    ad5b:	00 
    ad5c:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    ad63:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    ad68:	4c 89 ef             	mov    %r13,%rdi
    ad6b:	4c 89 f6             	mov    %r14,%rsi
    ad6e:	4c 89 e2             	mov    %r12,%rdx
    ad71:	48 89 d9             	mov    %rbx,%rcx
    ad74:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    ad7a:	41 b9 01 00 00 00    	mov    $0x1,%r9d
    ad80:	6a 01                	push   $0x1
    ad82:	6a 01                	push   $0x1
    ad84:	6a 01                	push   $0x1
    ad86:	6a 01                	push   $0x1
    ad88:	e8 d3 73 ff ff       	call   2160 <Conv2D_bias@plt>
    ad8d:	48 83 c4 20          	add    $0x20,%rsp
    ad91:	4c 89 ef             	mov    %r13,%rdi
    ad94:	e8 17 74 ff ff       	call   21b0 <omTensorDestroy@plt>
    ad99:	4c 89 f7             	mov    %r14,%rdi
    ad9c:	e8 0f 74 ff ff       	call   21b0 <omTensorDestroy@plt>
    ada1:	4c 89 e7             	mov    %r12,%rdi
    ada4:	e8 07 74 ff ff       	call   21b0 <omTensorDestroy@plt>
    ada9:	48 89 df             	mov    %rbx,%rdi
    adac:	e8 ff 73 ff ff       	call   21b0 <omTensorDestroy@plt>
    adb1:	48 8b 3c 24          	mov    (%rsp),%rdi
    adb5:	e8 56 73 ff ff       	call   2110 <free@plt>
    adba:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    adbf:	e8 0c 73 ff ff       	call   20d0 <malloc@plt>
    adc4:	48 89 c3             	mov    %rax,%rbx
    adc7:	4c 8d 60 0f          	lea    0xf(%rax),%r12
    adcb:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    add0:	49 83 e4 f0          	and    $0xfffffffffffffff0,%r12
    add4:	4c 89 24 24          	mov    %r12,(%rsp)
    add8:	bf 04 00 00 00       	mov    $0x4,%edi
    addd:	e8 be 73 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    ade2:	49 89 c6             	mov    %rax,%r14
    ade5:	48 89 c7             	mov    %rax,%rdi
    ade8:	31 f6                	xor    %esi,%esi
    adea:	48 89 da             	mov    %rbx,%rdx
    aded:	4c 89 e1             	mov    %r12,%rcx
    adf0:	e8 fb 72 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    adf5:	be 01 00 00 00       	mov    $0x1,%esi
    adfa:	4c 89 f7             	mov    %r14,%rdi
    adfd:	e8 9e 72 ff ff       	call   20a0 <omTensorSetDataType@plt>
    ae02:	4c 89 f7             	mov    %r14,%rdi
    ae05:	e8 c6 73 ff ff       	call   21d0 <omTensorGetShape@plt>
    ae0a:	48 89 c3             	mov    %rax,%rbx
    ae0d:	4c 89 f7             	mov    %r14,%rdi
    ae10:	e8 5b 72 ff ff       	call   2070 <omTensorGetStrides@plt>
    ae15:	4c 89 3b             	mov    %r15,(%rbx)
    ae18:	48 c7 00 00 09 00 00 	movq   $0x900,(%rax)
    ae1f:	48 c7 43 08 00 01 00 	movq   $0x100,0x8(%rbx)
    ae26:	00 
    ae27:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    ae2e:	00 
    ae2f:	48 c7 43 10 03 00 00 	movq   $0x3,0x10(%rbx)
    ae36:	00 
    ae37:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    ae3e:	00 
    ae3f:	48 c7 43 18 03 00 00 	movq   $0x3,0x18(%rbx)
    ae46:	00 
    ae47:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    ae4e:	00 
    ae4f:	bf 04 00 00 00       	mov    $0x4,%edi
    ae54:	e8 47 73 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    ae59:	48 89 c3             	mov    %rax,%rbx
    ae5c:	48 89 c7             	mov    %rax,%rdi
    ae5f:	31 f6                	xor    %esi,%esi
    ae61:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
    ae66:	4c 89 e2             	mov    %r12,%rdx
    ae69:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
    ae6e:	e8 7d 72 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    ae73:	be 01 00 00 00       	mov    $0x1,%esi
    ae78:	48 89 df             	mov    %rbx,%rdi
    ae7b:	e8 20 72 ff ff       	call   20a0 <omTensorSetDataType@plt>
    ae80:	48 89 df             	mov    %rbx,%rdi
    ae83:	e8 48 73 ff ff       	call   21d0 <omTensorGetShape@plt>
    ae88:	49 89 c5             	mov    %rax,%r13
    ae8b:	48 89 df             	mov    %rbx,%rdi
    ae8e:	e8 dd 71 ff ff       	call   2070 <omTensorGetStrides@plt>
    ae93:	4d 89 7d 00          	mov    %r15,0x0(%r13)
    ae97:	48 c7 00 00 09 00 00 	movq   $0x900,(%rax)
    ae9e:	49 c7 45 08 00 01 00 	movq   $0x100,0x8(%r13)
    aea5:	00 
    aea6:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    aead:	00 
    aeae:	49 c7 45 10 03 00 00 	movq   $0x3,0x10(%r13)
    aeb5:	00 
    aeb6:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    aebd:	00 
    aebe:	49 c7 45 18 03 00 00 	movq   $0x3,0x18(%r13)
    aec5:	00 
    aec6:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    aecd:	00 
    aece:	4c 89 f7             	mov    %r14,%rdi
    aed1:	48 89 de             	mov    %rbx,%rsi
    aed4:	e8 17 73 ff ff       	call   21f0 <Relu@plt>
    aed9:	4c 89 f7             	mov    %r14,%rdi
    aedc:	e8 cf 72 ff ff       	call   21b0 <omTensorDestroy@plt>
    aee1:	48 89 df             	mov    %rbx,%rdi
    aee4:	e8 c7 72 ff ff       	call   21b0 <omTensorDestroy@plt>
    aee9:	4c 89 e7             	mov    %r12,%rdi
    aeec:	e8 1f 72 ff ff       	call   2110 <free@plt>
    aef1:	bf 18 00 00 00       	mov    $0x18,%edi
    aef6:	e8 d5 71 ff ff       	call   20d0 <malloc@plt>
    aefb:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    af00:	48 83 c0 0f          	add    $0xf,%rax
    af04:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    af08:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    af0d:	4c 89 38             	mov    %r15,(%rax)
    af10:	bf 28 00 00 00       	mov    $0x28,%edi
    af15:	e8 b6 71 ff ff       	call   20d0 <malloc@plt>
    af1a:	48 8d 48 0f          	lea    0xf(%rax),%rcx
    af1e:	48 83 e1 f0          	and    $0xfffffffffffffff0,%rcx
    af22:	4c 89 39             	mov    %r15,(%rcx)
    af25:	48 c7 41 08 00 01 00 	movq   $0x100,0x8(%rcx)
    af2c:	00 
    af2d:	48 c7 41 10 ff ff ff 	movq   $0xffffffffffffffff,0x10(%rcx)
    af34:	ff 
    af35:	48 89 c7             	mov    %rax,%rdi
    af38:	e8 d3 71 ff ff       	call   2110 <free@plt>
    af3d:	48 89 e9             	mov    %rbp,%rcx
    af40:	48 c1 e1 08          	shl    $0x8,%rcx
    af44:	4c 8b 74 24 60       	mov    0x60(%rsp),%r14
    af49:	4c 89 f0             	mov    %r14,%rax
    af4c:	48 0f af c1          	imul   %rcx,%rax
    af50:	49 89 cc             	mov    %rcx,%r12
    af53:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
    af58:	48 8d 3c 85 10 00 00 	lea    0x10(,%rax,4),%rdi
    af5f:	00 
    af60:	48 89 7c 24 20       	mov    %rdi,0x20(%rsp)
    af65:	e8 66 71 ff ff       	call   20d0 <malloc@plt>
    af6a:	49 89 c7             	mov    %rax,%r15
    af6d:	4c 8d 68 0f          	lea    0xf(%rax),%r13
    af71:	49 83 e5 f0          	and    $0xfffffffffffffff0,%r13
    af75:	bf 03 00 00 00       	mov    $0x3,%edi
    af7a:	e8 21 72 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    af7f:	48 89 c3             	mov    %rax,%rbx
    af82:	48 89 c7             	mov    %rax,%rdi
    af85:	31 f6                	xor    %esi,%esi
    af87:	4c 89 7c 24 70       	mov    %r15,0x70(%rsp)
    af8c:	4c 89 fa             	mov    %r15,%rdx
    af8f:	4c 89 6c 24 08       	mov    %r13,0x8(%rsp)
    af94:	4c 89 e9             	mov    %r13,%rcx
    af97:	e8 54 71 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    af9c:	be 01 00 00 00       	mov    $0x1,%esi
    afa1:	48 89 df             	mov    %rbx,%rdi
    afa4:	e8 f7 70 ff ff       	call   20a0 <omTensorSetDataType@plt>
    afa9:	48 89 df             	mov    %rbx,%rdi
    afac:	e8 1f 72 ff ff       	call   21d0 <omTensorGetShape@plt>
    afb1:	49 89 c7             	mov    %rax,%r15
    afb4:	48 89 df             	mov    %rbx,%rdi
    afb7:	e8 b4 70 ff ff       	call   2070 <omTensorGetStrides@plt>
    afbc:	4d 89 37             	mov    %r14,(%r15)
    afbf:	4c 89 20             	mov    %r12,(%rax)
    afc2:	49 c7 47 08 00 01 00 	movq   $0x100,0x8(%r15)
    afc9:	00 
    afca:	48 89 68 08          	mov    %rbp,0x8(%rax)
    afce:	49 89 6f 10          	mov    %rbp,0x10(%r15)
    afd2:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    afd9:	00 
    afda:	bf 02 00 00 00       	mov    $0x2,%edi
    afdf:	e8 bc 71 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    afe4:	49 89 c5             	mov    %rax,%r13
    afe7:	48 8d 0d 22 17 01 00 	lea    0x11722(%rip),%rcx        # 1c710 <constant_23_mobilenet>
    afee:	48 89 c7             	mov    %rax,%rdi
    aff1:	31 f6                	xor    %esi,%esi
    aff3:	48 89 ca             	mov    %rcx,%rdx
    aff6:	e8 f5 70 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    affb:	be 01 00 00 00       	mov    $0x1,%esi
    b000:	4c 89 ef             	mov    %r13,%rdi
    b003:	e8 98 70 ff ff       	call   20a0 <omTensorSetDataType@plt>
    b008:	4c 89 ef             	mov    %r13,%rdi
    b00b:	e8 c0 71 ff ff       	call   21d0 <omTensorGetShape@plt>
    b010:	49 89 c7             	mov    %rax,%r15
    b013:	4c 89 ef             	mov    %r13,%rdi
    b016:	e8 55 70 ff ff       	call   2070 <omTensorGetStrides@plt>
    b01b:	49 c7 07 00 01 00 00 	movq   $0x100,(%r15)
    b022:	48 c7 00 00 01 00 00 	movq   $0x100,(%rax)
    b029:	49 c7 47 08 00 01 00 	movq   $0x100,0x8(%r15)
    b030:	00 
    b031:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    b038:	00 
    b039:	bf 03 00 00 00       	mov    $0x3,%edi
    b03e:	e8 5d 71 ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    b043:	49 89 c7             	mov    %rax,%r15
    b046:	48 89 c7             	mov    %rax,%rdi
    b049:	31 f6                	xor    %esi,%esi
    b04b:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    b050:	48 8b 0c 24          	mov    (%rsp),%rcx
    b054:	e8 97 70 ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    b059:	be 01 00 00 00       	mov    $0x1,%esi
    b05e:	4c 89 ff             	mov    %r15,%rdi
    b061:	e8 3a 70 ff ff       	call   20a0 <omTensorSetDataType@plt>
    b066:	4c 89 ff             	mov    %r15,%rdi
    b069:	e8 62 71 ff ff       	call   21d0 <omTensorGetShape@plt>
    b06e:	49 89 c4             	mov    %rax,%r12
    b071:	4c 89 ff             	mov    %r15,%rdi
    b074:	e8 f7 6f ff ff       	call   2070 <omTensorGetStrides@plt>
    b079:	4d 89 34 24          	mov    %r14,(%r12)
    b07d:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    b082:	48 89 08             	mov    %rcx,(%rax)
    b085:	49 c7 44 24 08 00 01 	movq   $0x100,0x8(%r12)
    b08c:	00 00 
    b08e:	48 89 68 08          	mov    %rbp,0x8(%rax)
    b092:	49 89 6c 24 10       	mov    %rbp,0x10(%r12)
    b097:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    b09e:	00 
    b09f:	48 89 df             	mov    %rbx,%rdi
    b0a2:	4c 89 ee             	mov    %r13,%rsi
    b0a5:	4c 89 fa             	mov    %r15,%rdx
    b0a8:	e8 a3 6f ff ff       	call   2050 <MatMul@plt>
    b0ad:	48 89 df             	mov    %rbx,%rdi
    b0b0:	e8 fb 70 ff ff       	call   21b0 <omTensorDestroy@plt>
    b0b5:	4c 89 ef             	mov    %r13,%rdi
    b0b8:	e8 f3 70 ff ff       	call   21b0 <omTensorDestroy@plt>
    b0bd:	4c 89 ff             	mov    %r15,%rdi
    b0c0:	e8 eb 70 ff ff       	call   21b0 <omTensorDestroy@plt>
    b0c5:	48 8b 7c 24 10       	mov    0x10(%rsp),%rdi
    b0ca:	e8 41 70 ff ff       	call   2110 <free@plt>
    b0cf:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    b0d4:	e8 f7 6f ff ff       	call   20d0 <malloc@plt>
    b0d9:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    b0de:	48 83 c0 0f          	add    $0xf,%rax
    b0e2:	48 83 e0 f0          	and    $0xfffffffffffffff0,%rax
    b0e6:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    b0eb:	4d 85 f6             	test   %r14,%r14
    b0ee:	0f 8e 59 01 00 00    	jle    b24d <main_graph_mobilenet+0x8f7d>
    b0f4:	48 89 e8             	mov    %rbp,%rax
    b0f7:	48 c1 e0 0a          	shl    $0xa,%rax
    b0fb:	48 89 44 24 10       	mov    %rax,0x10(%rsp)
    b100:	48 8d 0c ad 00 00 00 	lea    0x0(,%rbp,4),%rcx
    b107:	00 
    b108:	49 89 ee             	mov    %rbp,%r14
    b10b:	49 83 e6 f8          	and    $0xfffffffffffffff8,%r14
    b10f:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    b114:	49 8d 45 10          	lea    0x10(%r13),%rax
    b118:	48 89 04 24          	mov    %rax,(%rsp)
    b11c:	48 8b 44 24 08       	mov    0x8(%rsp),%rax
    b121:	48 8d 50 10          	lea    0x10(%rax),%rdx
    b125:	48 89 54 24 20       	mov    %rdx,0x20(%rsp)
    b12a:	45 31 c0             	xor    %r8d,%r8d
    b12d:	4c 8d 0d dc fe 00 00 	lea    0xfedc(%rip),%r9        # 1b010 <constant_36_mobilenet>
    b134:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    b139:	eb 29                	jmp    b164 <main_graph_mobilenet+0x8e94>
    b13b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    b140:	49 ff c0             	inc    %r8
    b143:	48 8b 44 24 10       	mov    0x10(%rsp),%rax
    b148:	48 01 04 24          	add    %rax,(%rsp)
    b14c:	48 01 44 24 20       	add    %rax,0x20(%rsp)
    b151:	49 01 c5             	add    %rax,%r13
    b154:	48 01 44 24 30       	add    %rax,0x30(%rsp)
    b159:	4c 3b 44 24 60       	cmp    0x60(%rsp),%r8
    b15e:	0f 84 e9 00 00 00    	je     b24d <main_graph_mobilenet+0x8f7d>
    b164:	48 85 ed             	test   %rbp,%rbp
    b167:	7e d7                	jle    b140 <main_graph_mobilenet+0x8e70>
    b169:	48 8b 5c 24 10       	mov    0x10(%rsp),%rbx
    b16e:	49 0f af d8          	imul   %r8,%rbx
    b172:	48 8b 44 24 40       	mov    0x40(%rsp),%rax
    b177:	4c 8d 3c 18          	lea    (%rax,%rbx,1),%r15
    b17b:	48 03 5c 24 08       	add    0x8(%rsp),%rbx
    b180:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
    b185:	4d 89 eb             	mov    %r13,%r11
    b188:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
    b18d:	48 8b 34 24          	mov    (%rsp),%rsi
    b191:	31 c0                	xor    %eax,%eax
    b193:	eb 22                	jmp    b1b7 <main_graph_mobilenet+0x8ee7>
    b195:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    b19c:	00 00 00 
    b19f:	90                   	nop
    b1a0:	48 ff c0             	inc    %rax
    b1a3:	48 01 ce             	add    %rcx,%rsi
    b1a6:	48 01 cf             	add    %rcx,%rdi
    b1a9:	49 01 cb             	add    %rcx,%r11
    b1ac:	49 01 cc             	add    %rcx,%r12
    b1af:	48 3d 00 01 00 00    	cmp    $0x100,%rax
    b1b5:	74 89                	je     b140 <main_graph_mobilenet+0x8e70>
    b1b7:	48 83 fd 08          	cmp    $0x8,%rbp
    b1bb:	f3 41 0f 10 04 81    	movss  (%r9,%rax,4),%xmm0
    b1c1:	73 0d                	jae    b1d0 <main_graph_mobilenet+0x8f00>
    b1c3:	31 d2                	xor    %edx,%edx
    b1c5:	eb 69                	jmp    b230 <main_graph_mobilenet+0x8f60>
    b1c7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    b1ce:	00 00 
    b1d0:	48 89 ca             	mov    %rcx,%rdx
    b1d3:	48 0f af d0          	imul   %rax,%rdx
    b1d7:	4c 8d 14 13          	lea    (%rbx,%rdx,1),%r10
    b1db:	4c 01 fa             	add    %r15,%rdx
    b1de:	4c 29 d2             	sub    %r10,%rdx
    b1e1:	48 83 fa 20          	cmp    $0x20,%rdx
    b1e5:	73 09                	jae    b1f0 <main_graph_mobilenet+0x8f20>
    b1e7:	31 d2                	xor    %edx,%edx
    b1e9:	eb 45                	jmp    b230 <main_graph_mobilenet+0x8f60>
    b1eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    b1f0:	0f 28 c8             	movaps %xmm0,%xmm1
    b1f3:	0f c6 c8 00          	shufps $0x0,%xmm0,%xmm1
    b1f7:	31 d2                	xor    %edx,%edx
    b1f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    b200:	0f 10 54 97 f0       	movups -0x10(%rdi,%rdx,4),%xmm2
    b205:	0f 10 1c 97          	movups (%rdi,%rdx,4),%xmm3
    b209:	0f 58 d1             	addps  %xmm1,%xmm2
    b20c:	0f 58 d9             	addps  %xmm1,%xmm3
    b20f:	0f 11 54 96 f0       	movups %xmm2,-0x10(%rsi,%rdx,4)
    b214:	0f 11 1c 96          	movups %xmm3,(%rsi,%rdx,4)
    b218:	48 83 c2 08          	add    $0x8,%rdx
    b21c:	49 39 d6             	cmp    %rdx,%r14
    b21f:	75 df                	jne    b200 <main_graph_mobilenet+0x8f30>
    b221:	4c 39 f5             	cmp    %r14,%rbp
    b224:	4c 89 f2             	mov    %r14,%rdx
    b227:	0f 84 73 ff ff ff    	je     b1a0 <main_graph_mobilenet+0x8ed0>
    b22d:	0f 1f 00             	nopl   (%rax)
    b230:	f3 41 0f 10 0c 94    	movss  (%r12,%rdx,4),%xmm1
    b236:	f3 0f 58 c8          	addss  %xmm0,%xmm1
    b23a:	f3 41 0f 11 0c 93    	movss  %xmm1,(%r11,%rdx,4)
    b240:	48 ff c2             	inc    %rdx
    b243:	48 39 d5             	cmp    %rdx,%rbp
    b246:	75 e8                	jne    b230 <main_graph_mobilenet+0x8f60>
    b248:	e9 53 ff ff ff       	jmp    b1a0 <main_graph_mobilenet+0x8ed0>
    b24d:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    b252:	e8 b9 6e ff ff       	call   2110 <free@plt>
    b257:	bf 30 00 00 00       	mov    $0x30,%edi
    b25c:	e8 6f 6e ff ff       	call   20d0 <malloc@plt>
    b261:	49 89 c7             	mov    %rax,%r15
    b264:	48 8d 58 0f          	lea    0xf(%rax),%rbx
    b268:	48 83 e3 f0          	and    $0xfffffffffffffff0,%rbx
    b26c:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
    b271:	48 8b 00             	mov    (%rax),%rax
    b274:	48 89 03             	mov    %rax,(%rbx)
    b277:	48 8b 7c 24 50       	mov    0x50(%rsp),%rdi
    b27c:	e8 8f 6e ff ff       	call   2110 <free@plt>
    b281:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
    b286:	48 0f af e8          	imul   %rax,%rbp
    b28a:	48 c1 e5 08          	shl    $0x8,%rbp
    b28e:	48 8b 0b             	mov    (%rbx),%rcx
    b291:	48 85 c9             	test   %rcx,%rcx
    b294:	48 0f 44 c8          	cmove  %rax,%rcx
    b298:	48 89 c8             	mov    %rcx,%rax
    b29b:	48 c1 e0 08          	shl    $0x8,%rax
    b29f:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    b2a3:	48 8d 34 c0          	lea    (%rax,%rax,8),%rsi
    b2a7:	b8 00 09 00 00       	mov    $0x900,%eax
    b2ac:	48 0f 44 f0          	cmove  %rax,%rsi
    b2b0:	48 89 f0             	mov    %rsi,%rax
    b2b3:	48 f7 d0             	not    %rax
    b2b6:	48 c1 f8 3f          	sar    $0x3f,%rax
    b2ba:	48 83 c8 01          	or     $0x1,%rax
    b2be:	48 29 e8             	sub    %rbp,%rax
    b2c1:	48 99                	cqto   
    b2c3:	48 f7 fe             	idiv   %rsi
    b2c6:	49 89 c4             	mov    %rax,%r12
    b2c9:	49 f7 d4             	not    %r12
    b2cc:	48 89 e8             	mov    %rbp,%rax
    b2cf:	48 99                	cqto   
    b2d1:	48 f7 fe             	idiv   %rsi
    b2d4:	48 85 f6             	test   %rsi,%rsi
    b2d7:	48 89 c2             	mov    %rax,%rdx
    b2da:	49 0f 48 d4          	cmovs  %r12,%rdx
    b2de:	48 85 ed             	test   %rbp,%rbp
    b2e1:	48 0f 4e d0          	cmovle %rax,%rdx
    b2e5:	48 85 f6             	test   %rsi,%rsi
    b2e8:	4c 0f 4e e2          	cmovle %rdx,%r12
    b2ec:	48 85 ed             	test   %rbp,%rbp
    b2ef:	4c 0f 49 e2          	cmovns %rdx,%r12
    b2f3:	48 83 f9 ff          	cmp    $0xffffffffffffffff,%rcx
    b2f7:	48 c7 43 08 00 01 00 	movq   $0x100,0x8(%rbx)
    b2fe:	00 
    b2ff:	48 c7 43 10 03 00 00 	movq   $0x3,0x10(%rbx)
    b306:	00 
    b307:	48 c7 43 18 03 00 00 	movq   $0x3,0x18(%rbx)
    b30e:	00 
    b30f:	4c 0f 45 e1          	cmovne %rcx,%r12
    b313:	4c 89 ff             	mov    %r15,%rdi
    b316:	e8 f5 6d ff ff       	call   2110 <free@plt>
    b31b:	4c 89 e3             	mov    %r12,%rbx
    b31e:	48 c1 e3 0a          	shl    $0xa,%rbx
    b322:	48 8d 7c db 10       	lea    0x10(%rbx,%rbx,8),%rdi
    b327:	e8 a4 6d ff ff       	call   20d0 <malloc@plt>
    b32c:	49 89 c6             	mov    %rax,%r14
    b32f:	4c 8d 68 0f          	lea    0xf(%rax),%r13
    b333:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    b338:	49 83 e5 f0          	and    $0xfffffffffffffff0,%r13
    b33c:	4c 89 6c 24 10       	mov    %r13,0x10(%rsp)
    b341:	bf 04 00 00 00       	mov    $0x4,%edi
    b346:	e8 55 6e ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    b34b:	49 89 c7             	mov    %rax,%r15
    b34e:	48 89 c7             	mov    %rax,%rdi
    b351:	31 f6                	xor    %esi,%esi
    b353:	4c 89 f2             	mov    %r14,%rdx
    b356:	4c 89 e9             	mov    %r13,%rcx
    b359:	e8 92 6d ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    b35e:	be 01 00 00 00       	mov    $0x1,%esi
    b363:	4c 89 ff             	mov    %r15,%rdi
    b366:	e8 35 6d ff ff       	call   20a0 <omTensorSetDataType@plt>
    b36b:	4c 89 ff             	mov    %r15,%rdi
    b36e:	e8 5d 6e ff ff       	call   21d0 <omTensorGetShape@plt>
    b373:	49 89 c6             	mov    %rax,%r14
    b376:	4c 89 ff             	mov    %r15,%rdi
    b379:	e8 f2 6c ff ff       	call   2070 <omTensorGetStrides@plt>
    b37e:	4d 89 26             	mov    %r12,(%r14)
    b381:	48 c7 00 00 09 00 00 	movq   $0x900,(%rax)
    b388:	49 c7 46 08 00 01 00 	movq   $0x100,0x8(%r14)
    b38f:	00 
    b390:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    b397:	00 
    b398:	49 c7 46 10 03 00 00 	movq   $0x3,0x10(%r14)
    b39f:	00 
    b3a0:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    b3a7:	00 
    b3a8:	49 c7 46 18 03 00 00 	movq   $0x3,0x18(%r14)
    b3af:	00 
    b3b0:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    b3b7:	00 
    b3b8:	bf 04 00 00 00       	mov    $0x4,%edi
    b3bd:	e8 de 6d ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    b3c2:	49 89 c6             	mov    %rax,%r14
    b3c5:	48 89 c7             	mov    %rax,%rdi
    b3c8:	31 f6                	xor    %esi,%esi
    b3ca:	48 8b 6c 24 78       	mov    0x78(%rsp),%rbp
    b3cf:	48 89 ea             	mov    %rbp,%rdx
    b3d2:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    b3d7:	e8 14 6d ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    b3dc:	be 01 00 00 00       	mov    $0x1,%esi
    b3e1:	4c 89 f7             	mov    %r14,%rdi
    b3e4:	e8 b7 6c ff ff       	call   20a0 <omTensorSetDataType@plt>
    b3e9:	4c 89 f7             	mov    %r14,%rdi
    b3ec:	e8 df 6d ff ff       	call   21d0 <omTensorGetShape@plt>
    b3f1:	49 89 c5             	mov    %rax,%r13
    b3f4:	4c 89 f7             	mov    %r14,%rdi
    b3f7:	e8 74 6c ff ff       	call   2070 <omTensorGetStrides@plt>
    b3fc:	4d 89 65 00          	mov    %r12,0x0(%r13)
    b400:	48 c7 00 00 09 00 00 	movq   $0x900,(%rax)
    b407:	49 c7 45 08 00 01 00 	movq   $0x100,0x8(%r13)
    b40e:	00 
    b40f:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    b416:	00 
    b417:	49 c7 45 10 03 00 00 	movq   $0x3,0x10(%r13)
    b41e:	00 
    b41f:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    b426:	00 
    b427:	49 c7 45 18 03 00 00 	movq   $0x3,0x18(%r13)
    b42e:	00 
    b42f:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    b436:	00 
    b437:	4c 89 ff             	mov    %r15,%rdi
    b43a:	4c 89 f6             	mov    %r14,%rsi
    b43d:	e8 ae 6d ff ff       	call   21f0 <Relu@plt>
    b442:	4c 89 ff             	mov    %r15,%rdi
    b445:	e8 66 6d ff ff       	call   21b0 <omTensorDestroy@plt>
    b44a:	4c 89 f7             	mov    %r14,%rdi
    b44d:	e8 5e 6d ff ff       	call   21b0 <omTensorDestroy@plt>
    b452:	48 89 ef             	mov    %rbp,%rdi
    b455:	e8 b6 6c ff ff       	call   2110 <free@plt>
    b45a:	48 83 cb 10          	or     $0x10,%rbx
    b45e:	48 89 df             	mov    %rbx,%rdi
    b461:	e8 6a 6c ff ff       	call   20d0 <malloc@plt>
    b466:	48 89 c3             	mov    %rax,%rbx
    b469:	48 8d 68 0f          	lea    0xf(%rax),%rbp
    b46d:	48 83 e5 f0          	and    $0xfffffffffffffff0,%rbp
    b471:	bf 04 00 00 00       	mov    $0x4,%edi
    b476:	e8 25 6d ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    b47b:	49 89 c7             	mov    %rax,%r15
    b47e:	48 89 c7             	mov    %rax,%rdi
    b481:	31 f6                	xor    %esi,%esi
    b483:	48 89 5c 24 70       	mov    %rbx,0x70(%rsp)
    b488:	48 89 da             	mov    %rbx,%rdx
    b48b:	48 89 e9             	mov    %rbp,%rcx
    b48e:	e8 5d 6c ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    b493:	be 01 00 00 00       	mov    $0x1,%esi
    b498:	4c 89 ff             	mov    %r15,%rdi
    b49b:	e8 00 6c ff ff       	call   20a0 <omTensorSetDataType@plt>
    b4a0:	4c 89 ff             	mov    %r15,%rdi
    b4a3:	e8 28 6d ff ff       	call   21d0 <omTensorGetShape@plt>
    b4a8:	49 89 c6             	mov    %rax,%r14
    b4ab:	4c 89 ff             	mov    %r15,%rdi
    b4ae:	e8 bd 6b ff ff       	call   2070 <omTensorGetStrides@plt>
    b4b3:	4d 89 26             	mov    %r12,(%r14)
    b4b6:	48 c7 00 00 01 00 00 	movq   $0x100,(%rax)
    b4bd:	49 c7 46 08 00 01 00 	movq   $0x100,0x8(%r14)
    b4c4:	00 
    b4c5:	48 c7 40 08 01 00 00 	movq   $0x1,0x8(%rax)
    b4cc:	00 
    b4cd:	49 c7 46 10 01 00 00 	movq   $0x1,0x10(%r14)
    b4d4:	00 
    b4d5:	48 c7 40 10 01 00 00 	movq   $0x1,0x10(%rax)
    b4dc:	00 
    b4dd:	49 c7 46 18 01 00 00 	movq   $0x1,0x18(%r14)
    b4e4:	00 
    b4e5:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    b4ec:	00 
    b4ed:	bf 04 00 00 00       	mov    $0x4,%edi
    b4f2:	e8 a9 6c ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    b4f7:	49 89 c5             	mov    %rax,%r13
    b4fa:	48 89 c7             	mov    %rax,%rdi
    b4fd:	31 f6                	xor    %esi,%esi
    b4ff:	48 8b 5c 24 20       	mov    0x20(%rsp),%rbx
    b504:	48 89 da             	mov    %rbx,%rdx
    b507:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
    b50c:	e8 df 6b ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    b511:	be 01 00 00 00       	mov    $0x1,%esi
    b516:	4c 89 ef             	mov    %r13,%rdi
    b519:	e8 82 6b ff ff       	call   20a0 <omTensorSetDataType@plt>
    b51e:	4c 89 ef             	mov    %r13,%rdi
    b521:	e8 aa 6c ff ff       	call   21d0 <omTensorGetShape@plt>
    b526:	49 89 c6             	mov    %rax,%r14
    b529:	4c 89 ef             	mov    %r13,%rdi
    b52c:	e8 3f 6b ff ff       	call   2070 <omTensorGetStrides@plt>
    b531:	4d 89 26             	mov    %r12,(%r14)
    b534:	48 c7 00 00 09 00 00 	movq   $0x900,(%rax)
    b53b:	49 c7 46 08 00 01 00 	movq   $0x100,0x8(%r14)
    b542:	00 
    b543:	48 c7 40 08 09 00 00 	movq   $0x9,0x8(%rax)
    b54a:	00 
    b54b:	49 c7 46 10 03 00 00 	movq   $0x3,0x10(%r14)
    b552:	00 
    b553:	48 c7 40 10 03 00 00 	movq   $0x3,0x10(%rax)
    b55a:	00 
    b55b:	49 c7 46 18 03 00 00 	movq   $0x3,0x18(%r14)
    b562:	00 
    b563:	48 c7 40 18 01 00 00 	movq   $0x1,0x18(%rax)
    b56a:	00 
    b56b:	4c 89 ff             	mov    %r15,%rdi
    b56e:	4c 89 ee             	mov    %r13,%rsi
    b571:	ba 03 00 00 00       	mov    $0x3,%edx
    b576:	b9 03 00 00 00       	mov    $0x3,%ecx
    b57b:	41 b8 03 00 00 00    	mov    $0x3,%r8d
    b581:	41 b9 03 00 00 00    	mov    $0x3,%r9d
    b587:	e8 04 6c ff ff       	call   2190 <AveragePool@plt>
    b58c:	4c 89 ff             	mov    %r15,%rdi
    b58f:	e8 1c 6c ff ff       	call   21b0 <omTensorDestroy@plt>
    b594:	4c 89 ef             	mov    %r13,%rdi
    b597:	4d 89 e5             	mov    %r12,%r13
    b59a:	e8 11 6c ff ff       	call   21b0 <omTensorDestroy@plt>
    b59f:	48 89 df             	mov    %rbx,%rdi
    b5a2:	e8 69 6b ff ff       	call   2110 <free@plt>
    b5a7:	4a 8d 1c e5 00 00 00 	lea    0x0(,%r12,8),%rbx
    b5ae:	00 
    b5af:	4a 8d 3c e5 80 00 00 	lea    0x80(,%r12,8),%rdi
    b5b6:	00 
    b5b7:	e8 14 6b ff ff       	call   20d0 <malloc@plt>
    b5bc:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
    b5c1:	4c 8d 78 7f          	lea    0x7f(%rax),%r15
    b5c5:	49 83 e7 80          	and    $0xffffffffffffff80,%r15
    b5c9:	4d 85 e4             	test   %r12,%r12
    b5cc:	7e 0d                	jle    b5db <main_graph_mobilenet+0x930b>
    b5ce:	4c 89 ff             	mov    %r15,%rdi
    b5d1:	31 f6                	xor    %esi,%esi
    b5d3:	48 89 da             	mov    %rbx,%rdx
    b5d6:	e8 65 6a ff ff       	call   2040 <memset@plt>
    b5db:	bf 80 80 00 00       	mov    $0x8080,%edi
    b5e0:	e8 eb 6a ff ff       	call   20d0 <malloc@plt>
    b5e5:	49 89 c6             	mov    %rax,%r14
    b5e8:	bf 80 00 01 00       	mov    $0x10080,%edi
    b5ed:	e8 de 6a ff ff       	call   20d0 <malloc@plt>
    b5f2:	48 89 84 24 88 00 00 	mov    %rax,0x88(%rsp)
    b5f9:	00 
    b5fa:	4c 8d 58 7f          	lea    0x7f(%rax),%r11
    b5fe:	49 83 e3 80          	and    $0xffffffffffffff80,%r11
    b602:	4d 85 ed             	test   %r13,%r13
    b605:	48 89 9c 24 98 00 00 	mov    %rbx,0x98(%rsp)
    b60c:	00 
    b60d:	4c 89 b4 24 90 00 00 	mov    %r14,0x90(%rsp)
    b614:	00 
    b615:	0f 8e e7 07 00 00    	jle    be02 <main_graph_mobilenet+0x9b32>
    b61b:	49 83 c6 7f          	add    $0x7f,%r14
    b61f:	49 83 e6 80          	and    $0xffffffffffffff80,%r14
    b623:	49 8d 43 04          	lea    0x4(%r11),%rax
    b627:	48 c7 c1 00 ff ff ff 	mov    $0xffffffffffffff00,%rcx
    b62e:	48 8d 15 9b 35 00 00 	lea    0x359b(%rip),%rdx        # ebd0 <constant_66_mobilenet>
    b635:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    b63c:	00 00 00 
    b63f:	90                   	nop
    b640:	48 8b b4 ca 00 08 00 	mov    0x800(%rdx,%rcx,8),%rsi
    b647:	00 
    b648:	48 89 70 fc          	mov    %rsi,-0x4(%rax)
    b64c:	48 05 00 01 00 00    	add    $0x100,%rax
    b652:	48 ff c1             	inc    %rcx
    b655:	75 e9                	jne    b640 <main_graph_mobilenet+0x9370>
    b657:	49 8d 43 04          	lea    0x4(%r11),%rax
    b65b:	48 89 44 24 60       	mov    %rax,0x60(%rsp)
    b660:	45 31 e4             	xor    %r12d,%r12d
    b663:	48 89 6c 24 58       	mov    %rbp,0x58(%rsp)
    b668:	4c 89 e9             	mov    %r13,%rcx
    b66b:	31 db                	xor    %ebx,%ebx
    b66d:	4c 89 2c 24          	mov    %r13,(%rsp)
    b671:	4c 89 5c 24 30       	mov    %r11,0x30(%rsp)
    b676:	eb 34                	jmp    b6ac <main_graph_mobilenet+0x93dc>
    b678:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    b67f:	00 
    b680:	48 8b 4c 24 40       	mov    0x40(%rsp),%rcx
    b685:	48 83 c1 e0          	add    $0xffffffffffffffe0,%rcx
    b689:	4c 8b 64 24 08       	mov    0x8(%rsp),%r12
    b68e:	49 ff c4             	inc    %r12
    b691:	48 81 44 24 58 00 80 	addq   $0x8000,0x58(%rsp)
    b698:	00 00 
    b69a:	4c 8b 2c 24          	mov    (%rsp),%r13
    b69e:	48 8b 5c 24 50       	mov    0x50(%rsp),%rbx
    b6a3:	4c 39 eb             	cmp    %r13,%rbx
    b6a6:	0f 8d 8c 07 00 00    	jge    be38 <main_graph_mobilenet+0x9b68>
    b6ac:	48 83 f9 20          	cmp    $0x20,%rcx
    b6b0:	b8 20 00 00 00       	mov    $0x20,%eax
    b6b5:	48 89 4c 24 40       	mov    %rcx,0x40(%rsp)
    b6ba:	48 0f 4c c1          	cmovl  %rcx,%rax
    b6be:	48 83 f8 02          	cmp    $0x2,%rax
    b6c2:	b9 01 00 00 00       	mov    $0x1,%ecx
    b6c7:	48 0f 4c c1          	cmovl  %rcx,%rax
    b6cb:	4c 89 e9             	mov    %r13,%rcx
    b6ce:	48 29 d9             	sub    %rbx,%rcx
    b6d1:	48 85 c9             	test   %rcx,%rcx
    b6d4:	0f 8e d6 05 00 00    	jle    bcb0 <main_graph_mobilenet+0x99e0>
    b6da:	4c 89 e1             	mov    %r12,%rcx
    b6dd:	48 c1 e1 0f          	shl    $0xf,%rcx
    b6e1:	48 01 e9             	add    %rbp,%rcx
    b6e4:	48 8b 54 24 58       	mov    0x58(%rsp),%rdx
    b6e9:	4c 89 f6             	mov    %r14,%rsi
    b6ec:	31 ff                	xor    %edi,%edi
    b6ee:	e9 76 05 00 00       	jmp    bc69 <main_graph_mobilenet+0x9999>
    b6f3:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    b6fa:	00 00 00 
    b6fd:	0f 1f 00             	nopl   (%rax)
    b700:	4c 8d 04 1f          	lea    (%rdi,%rbx,1),%r8
    b704:	49 c1 e0 08          	shl    $0x8,%r8
    b708:	49 89 fa             	mov    %rdi,%r10
    b70b:	49 c1 e2 08          	shl    $0x8,%r10
    b70f:	4e 8d 0c 85 00 00 00 	lea    0x0(,%r8,4),%r9
    b716:	00 
    b717:	42 0f 28 44 85 00    	movaps 0x0(%rbp,%r8,4),%xmm0
    b71d:	42 0f 28 4c 85 10    	movaps 0x10(%rbp,%r8,4),%xmm1
    b723:	4e 8d 04 95 00 00 00 	lea    0x0(,%r10,4),%r8
    b72a:	00 
    b72b:	43 0f 29 04 96       	movaps %xmm0,(%r14,%r10,4)
    b730:	43 0f 29 4c 96 10    	movaps %xmm1,0x10(%r14,%r10,4)
    b736:	4d 89 ca             	mov    %r9,%r10
    b739:	49 83 ca 20          	or     $0x20,%r10
    b73d:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b743:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b749:	4d 89 c2             	mov    %r8,%r10
    b74c:	49 83 ca 20          	or     $0x20,%r10
    b750:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b755:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b75b:	4d 89 ca             	mov    %r9,%r10
    b75e:	49 83 ca 40          	or     $0x40,%r10
    b762:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b768:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b76e:	4d 89 c2             	mov    %r8,%r10
    b771:	49 83 ca 40          	or     $0x40,%r10
    b775:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b77a:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b780:	4d 89 ca             	mov    %r9,%r10
    b783:	49 83 ca 60          	or     $0x60,%r10
    b787:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b78d:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b793:	4d 89 c2             	mov    %r8,%r10
    b796:	49 83 ca 60          	or     $0x60,%r10
    b79a:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b79f:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b7a5:	4d 89 ca             	mov    %r9,%r10
    b7a8:	49 81 ca 80 00 00 00 	or     $0x80,%r10
    b7af:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b7b5:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b7bb:	4d 89 c2             	mov    %r8,%r10
    b7be:	49 81 ca 80 00 00 00 	or     $0x80,%r10
    b7c5:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b7ca:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b7d0:	4d 89 ca             	mov    %r9,%r10
    b7d3:	49 81 ca a0 00 00 00 	or     $0xa0,%r10
    b7da:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b7e0:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b7e6:	4d 89 c2             	mov    %r8,%r10
    b7e9:	49 81 ca a0 00 00 00 	or     $0xa0,%r10
    b7f0:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b7f5:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b7fb:	4d 89 ca             	mov    %r9,%r10
    b7fe:	49 81 ca c0 00 00 00 	or     $0xc0,%r10
    b805:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b80b:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b811:	4d 89 c2             	mov    %r8,%r10
    b814:	49 81 ca c0 00 00 00 	or     $0xc0,%r10
    b81b:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b820:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b826:	4d 89 ca             	mov    %r9,%r10
    b829:	49 81 ca e0 00 00 00 	or     $0xe0,%r10
    b830:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b836:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b83c:	4d 89 c2             	mov    %r8,%r10
    b83f:	49 81 ca e0 00 00 00 	or     $0xe0,%r10
    b846:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b84b:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b851:	4d 89 ca             	mov    %r9,%r10
    b854:	49 81 ca 00 01 00 00 	or     $0x100,%r10
    b85b:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b861:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b867:	4d 89 c2             	mov    %r8,%r10
    b86a:	49 81 ca 00 01 00 00 	or     $0x100,%r10
    b871:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b876:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b87c:	4d 89 ca             	mov    %r9,%r10
    b87f:	49 81 ca 20 01 00 00 	or     $0x120,%r10
    b886:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b88c:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b892:	4d 89 c2             	mov    %r8,%r10
    b895:	49 81 ca 20 01 00 00 	or     $0x120,%r10
    b89c:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b8a1:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b8a7:	4d 89 ca             	mov    %r9,%r10
    b8aa:	49 81 ca 40 01 00 00 	or     $0x140,%r10
    b8b1:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b8b7:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b8bd:	4d 89 c2             	mov    %r8,%r10
    b8c0:	49 81 ca 40 01 00 00 	or     $0x140,%r10
    b8c7:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b8cc:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b8d2:	4d 89 ca             	mov    %r9,%r10
    b8d5:	49 81 ca 60 01 00 00 	or     $0x160,%r10
    b8dc:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b8e2:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b8e8:	4d 89 c2             	mov    %r8,%r10
    b8eb:	49 81 ca 60 01 00 00 	or     $0x160,%r10
    b8f2:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b8f7:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b8fd:	4d 89 ca             	mov    %r9,%r10
    b900:	49 81 ca 80 01 00 00 	or     $0x180,%r10
    b907:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b90d:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b913:	4d 89 c2             	mov    %r8,%r10
    b916:	49 81 ca 80 01 00 00 	or     $0x180,%r10
    b91d:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b922:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b928:	4d 89 ca             	mov    %r9,%r10
    b92b:	49 81 ca a0 01 00 00 	or     $0x1a0,%r10
    b932:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b938:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b93e:	4d 89 c2             	mov    %r8,%r10
    b941:	49 81 ca a0 01 00 00 	or     $0x1a0,%r10
    b948:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b94d:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b953:	4d 89 ca             	mov    %r9,%r10
    b956:	49 81 ca c0 01 00 00 	or     $0x1c0,%r10
    b95d:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b963:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b969:	4d 89 c2             	mov    %r8,%r10
    b96c:	49 81 ca c0 01 00 00 	or     $0x1c0,%r10
    b973:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b978:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b97e:	4d 89 ca             	mov    %r9,%r10
    b981:	49 81 ca e0 01 00 00 	or     $0x1e0,%r10
    b988:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b98e:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b994:	4d 89 c2             	mov    %r8,%r10
    b997:	49 81 ca e0 01 00 00 	or     $0x1e0,%r10
    b99e:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b9a3:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b9a9:	4d 89 ca             	mov    %r9,%r10
    b9ac:	49 81 ca 00 02 00 00 	or     $0x200,%r10
    b9b3:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b9b9:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b9bf:	4d 89 c2             	mov    %r8,%r10
    b9c2:	49 81 ca 00 02 00 00 	or     $0x200,%r10
    b9c9:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b9ce:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b9d4:	4d 89 ca             	mov    %r9,%r10
    b9d7:	49 81 ca 20 02 00 00 	or     $0x220,%r10
    b9de:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    b9e4:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    b9ea:	4d 89 c2             	mov    %r8,%r10
    b9ed:	49 81 ca 20 02 00 00 	or     $0x220,%r10
    b9f4:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    b9f9:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    b9ff:	4d 89 ca             	mov    %r9,%r10
    ba02:	49 81 ca 40 02 00 00 	or     $0x240,%r10
    ba09:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    ba0f:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    ba15:	4d 89 c2             	mov    %r8,%r10
    ba18:	49 81 ca 40 02 00 00 	or     $0x240,%r10
    ba1f:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    ba24:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    ba2a:	4d 89 ca             	mov    %r9,%r10
    ba2d:	49 81 ca 60 02 00 00 	or     $0x260,%r10
    ba34:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    ba3a:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    ba40:	4d 89 c2             	mov    %r8,%r10
    ba43:	49 81 ca 60 02 00 00 	or     $0x260,%r10
    ba4a:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    ba4f:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    ba55:	4d 89 ca             	mov    %r9,%r10
    ba58:	49 81 ca 80 02 00 00 	or     $0x280,%r10
    ba5f:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    ba65:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    ba6b:	4d 89 c2             	mov    %r8,%r10
    ba6e:	49 81 ca 80 02 00 00 	or     $0x280,%r10
    ba75:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    ba7a:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    ba80:	4d 89 ca             	mov    %r9,%r10
    ba83:	49 81 ca a0 02 00 00 	or     $0x2a0,%r10
    ba8a:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    ba90:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    ba96:	4d 89 c2             	mov    %r8,%r10
    ba99:	49 81 ca a0 02 00 00 	or     $0x2a0,%r10
    baa0:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    baa5:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    baab:	4d 89 ca             	mov    %r9,%r10
    baae:	49 81 ca c0 02 00 00 	or     $0x2c0,%r10
    bab5:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    babb:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    bac1:	4d 89 c2             	mov    %r8,%r10
    bac4:	49 81 ca c0 02 00 00 	or     $0x2c0,%r10
    bacb:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    bad0:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    bad6:	4d 89 ca             	mov    %r9,%r10
    bad9:	49 81 ca e0 02 00 00 	or     $0x2e0,%r10
    bae0:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    bae6:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    baec:	4d 89 c2             	mov    %r8,%r10
    baef:	49 81 ca e0 02 00 00 	or     $0x2e0,%r10
    baf6:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    bafb:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    bb01:	4d 89 ca             	mov    %r9,%r10
    bb04:	49 81 ca 00 03 00 00 	or     $0x300,%r10
    bb0b:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    bb11:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    bb17:	4d 89 c2             	mov    %r8,%r10
    bb1a:	49 81 ca 00 03 00 00 	or     $0x300,%r10
    bb21:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    bb26:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    bb2c:	4d 89 ca             	mov    %r9,%r10
    bb2f:	49 81 ca 20 03 00 00 	or     $0x320,%r10
    bb36:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    bb3c:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    bb42:	4d 89 c2             	mov    %r8,%r10
    bb45:	49 81 ca 20 03 00 00 	or     $0x320,%r10
    bb4c:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    bb51:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    bb57:	4d 89 ca             	mov    %r9,%r10
    bb5a:	49 81 ca 40 03 00 00 	or     $0x340,%r10
    bb61:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    bb67:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    bb6d:	4d 89 c2             	mov    %r8,%r10
    bb70:	49 81 ca 40 03 00 00 	or     $0x340,%r10
    bb77:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    bb7c:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    bb82:	4d 89 ca             	mov    %r9,%r10
    bb85:	49 81 ca 60 03 00 00 	or     $0x360,%r10
    bb8c:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    bb92:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    bb98:	4d 89 c2             	mov    %r8,%r10
    bb9b:	49 81 ca 60 03 00 00 	or     $0x360,%r10
    bba2:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    bba7:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    bbad:	4d 89 ca             	mov    %r9,%r10
    bbb0:	49 81 ca 80 03 00 00 	or     $0x380,%r10
    bbb7:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    bbbd:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    bbc3:	4d 89 c2             	mov    %r8,%r10
    bbc6:	49 81 ca 80 03 00 00 	or     $0x380,%r10
    bbcd:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    bbd2:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    bbd8:	4d 89 ca             	mov    %r9,%r10
    bbdb:	49 81 ca a0 03 00 00 	or     $0x3a0,%r10
    bbe2:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    bbe8:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    bbee:	4d 89 c2             	mov    %r8,%r10
    bbf1:	49 81 ca a0 03 00 00 	or     $0x3a0,%r10
    bbf8:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    bbfd:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    bc03:	4d 89 ca             	mov    %r9,%r10
    bc06:	49 81 ca c0 03 00 00 	or     $0x3c0,%r10
    bc0d:	42 0f 28 44 15 00    	movaps 0x0(%rbp,%r10,1),%xmm0
    bc13:	42 0f 28 4c 15 10    	movaps 0x10(%rbp,%r10,1),%xmm1
    bc19:	4d 89 c2             	mov    %r8,%r10
    bc1c:	49 81 ca c0 03 00 00 	or     $0x3c0,%r10
    bc23:	43 0f 29 04 16       	movaps %xmm0,(%r14,%r10,1)
    bc28:	43 0f 29 4c 16 10    	movaps %xmm1,0x10(%r14,%r10,1)
    bc2e:	49 81 c9 e0 03 00 00 	or     $0x3e0,%r9
    bc35:	42 0f 28 44 0d 00    	movaps 0x0(%rbp,%r9,1),%xmm0
    bc3b:	42 0f 28 4c 0d 10    	movaps 0x10(%rbp,%r9,1),%xmm1
    bc41:	49 81 c8 e0 03 00 00 	or     $0x3e0,%r8
    bc48:	43 0f 29 04 06       	movaps %xmm0,(%r14,%r8,1)
    bc4d:	43 0f 29 4c 06 10    	movaps %xmm1,0x10(%r14,%r8,1)
    bc53:	48 ff c7             	inc    %rdi
    bc56:	48 81 c6 00 04 00 00 	add    $0x400,%rsi
    bc5d:	48 81 c2 00 04 00 00 	add    $0x400,%rdx
    bc64:	48 39 c7             	cmp    %rax,%rdi
    bc67:	74 47                	je     bcb0 <main_graph_mobilenet+0x99e0>
    bc69:	49 89 f8             	mov    %rdi,%r8
    bc6c:	49 c1 e0 0a          	shl    $0xa,%r8
    bc70:	4e 8d 0c 01          	lea    (%rcx,%r8,1),%r9
    bc74:	4d 01 f0             	add    %r14,%r8
    bc77:	4d 29 c8             	sub    %r9,%r8
    bc7a:	49 83 f8 1f          	cmp    $0x1f,%r8
    bc7e:	0f 87 7c fa ff ff    	ja     b700 <main_graph_mobilenet+0x9430>
    bc84:	49 c7 c0 00 fc ff ff 	mov    $0xfffffffffffffc00,%r8
    bc8b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    bc90:	f3 42 0f 10 84 02 00 	movss  0x400(%rdx,%r8,1),%xmm0
    bc97:	04 00 00 
    bc9a:	f3 42 0f 11 84 06 00 	movss  %xmm0,0x400(%rsi,%r8,1)
    bca1:	04 00 00 
    bca4:	49 83 c0 04          	add    $0x4,%r8
    bca8:	75 e6                	jne    bc90 <main_graph_mobilenet+0x99c0>
    bcaa:	eb a7                	jmp    bc53 <main_graph_mobilenet+0x9983>
    bcac:	0f 1f 40 00          	nopl   0x0(%rax)
    bcb0:	4c 89 64 24 08       	mov    %r12,0x8(%rsp)
    bcb5:	48 8d 43 20          	lea    0x20(%rbx),%rax
    bcb9:	4c 39 e8             	cmp    %r13,%rax
    bcbc:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
    bcc1:	4c 0f 4c e8          	cmovl  %rax,%r13
    bcc5:	4c 89 6c 24 20       	mov    %r13,0x20(%rsp)
    bcca:	4c 39 eb             	cmp    %r13,%rbx
    bccd:	0f 8d ad f9 ff ff    	jge    b680 <main_graph_mobilenet+0x93b0>
    bcd3:	4c 89 74 24 10       	mov    %r14,0x10(%rsp)
    bcd8:	4c 8b 6c 24 40       	mov    0x40(%rsp),%r13
    bcdd:	eb 25                	jmp    bd04 <main_graph_mobilenet+0x9a34>
    bcdf:	90                   	nop
    bce0:	48 89 c7             	mov    %rax,%rdi
    bce3:	e8 28 64 ff ff       	call   2110 <free@plt>
    bce8:	48 83 c3 04          	add    $0x4,%rbx
    bcec:	49 83 c5 fc          	add    $0xfffffffffffffffc,%r13
    bcf0:	48 81 44 24 10 00 10 	addq   $0x1000,0x10(%rsp)
    bcf7:	00 00 
    bcf9:	48 3b 5c 24 20       	cmp    0x20(%rsp),%rbx
    bcfe:	0f 8d 7c f9 ff ff    	jge    b680 <main_graph_mobilenet+0x93b0>
    bd04:	49 83 fd 04          	cmp    $0x4,%r13
    bd08:	41 bc 04 00 00 00    	mov    $0x4,%r12d
    bd0e:	4d 0f 4c e5          	cmovl  %r13,%r12
    bd12:	49 83 fc 02          	cmp    $0x2,%r12
    bd16:	b8 01 00 00 00       	mov    $0x1,%eax
    bd1b:	4c 0f 4c e0          	cmovl  %rax,%r12
    bd1f:	bf 44 00 00 00       	mov    $0x44,%edi
    bd24:	e8 a7 63 ff ff       	call   20d0 <malloc@plt>
    bd29:	48 8b 0c 24          	mov    (%rsp),%rcx
    bd2d:	48 29 d9             	sub    %rbx,%rcx
    bd30:	48 85 c9             	test   %rcx,%rcx
    bd33:	4c 8b 54 24 30       	mov    0x30(%rsp),%r10
    bd38:	4c 8b 5c 24 60       	mov    0x60(%rsp),%r11
    bd3d:	7e a1                	jle    bce0 <main_graph_mobilenet+0x9a10>
    bd3f:	48 8d 48 3f          	lea    0x3f(%rax),%rcx
    bd43:	48 83 e1 c0          	and    $0xffffffffffffffc0,%rcx
    bd47:	48 8b 54 24 10       	mov    0x10(%rsp),%rdx
    bd4c:	31 f6                	xor    %esi,%esi
    bd4e:	66 90                	xchg   %ax,%ax
    bd50:	48 8d 3c 1e          	lea    (%rsi,%rbx,1),%rdi
    bd54:	f3 41 0f 10 04 ff    	movss  (%r15,%rdi,8),%xmm0
    bd5a:	48 01 ff             	add    %rdi,%rdi
    bd5d:	f3 0f 11 01          	movss  %xmm0,(%rcx)
    bd61:	49 c7 c0 00 fc ff ff 	mov    $0xfffffffffffffc00,%r8
    bd68:	4d 89 d1             	mov    %r10,%r9
    bd6b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    bd70:	f3 42 0f 10 8c 02 00 	movss  0x400(%rdx,%r8,1),%xmm1
    bd77:	04 00 00 
    bd7a:	f3 41 0f 59 09       	mulss  (%r9),%xmm1
    bd7f:	f3 0f 58 c1          	addss  %xmm1,%xmm0
    bd83:	f3 0f 11 01          	movss  %xmm0,(%rcx)
    bd87:	49 81 c1 00 01 00 00 	add    $0x100,%r9
    bd8e:	49 83 c0 04          	add    $0x4,%r8
    bd92:	75 dc                	jne    bd70 <main_graph_mobilenet+0x9aa0>
    bd94:	f3 41 0f 11 04 bf    	movss  %xmm0,(%r15,%rdi,4)
    bd9a:	48 83 cf 01          	or     $0x1,%rdi
    bd9e:	f3 41 0f 10 04 bf    	movss  (%r15,%rdi,4),%xmm0
    bda4:	f3 0f 11 01          	movss  %xmm0,(%rcx)
    bda8:	49 c7 c0 00 fc ff ff 	mov    $0xfffffffffffffc00,%r8
    bdaf:	4d 89 d9             	mov    %r11,%r9
    bdb2:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    bdb9:	00 00 00 
    bdbc:	0f 1f 40 00          	nopl   0x0(%rax)
    bdc0:	f3 42 0f 10 8c 02 00 	movss  0x400(%rdx,%r8,1),%xmm1
    bdc7:	04 00 00 
    bdca:	f3 41 0f 59 09       	mulss  (%r9),%xmm1
    bdcf:	f3 0f 58 c1          	addss  %xmm1,%xmm0
    bdd3:	f3 0f 11 01          	movss  %xmm0,(%rcx)
    bdd7:	49 81 c1 00 01 00 00 	add    $0x100,%r9
    bdde:	49 83 c0 04          	add    $0x4,%r8
    bde2:	75 dc                	jne    bdc0 <main_graph_mobilenet+0x9af0>
    bde4:	f3 41 0f 11 04 bf    	movss  %xmm0,(%r15,%rdi,4)
    bdea:	48 ff c6             	inc    %rsi
    bded:	48 81 c2 00 04 00 00 	add    $0x400,%rdx
    bdf4:	4c 39 e6             	cmp    %r12,%rsi
    bdf7:	0f 85 53 ff ff ff    	jne    bd50 <main_graph_mobilenet+0x9a80>
    bdfd:	e9 de fe ff ff       	jmp    bce0 <main_graph_mobilenet+0x9a10>
    be02:	49 83 c3 04          	add    $0x4,%r11
    be06:	48 c7 c0 00 ff ff ff 	mov    $0xffffffffffffff00,%rax
    be0d:	48 8d 0d bc 2d 00 00 	lea    0x2dbc(%rip),%rcx        # ebd0 <constant_66_mobilenet>
    be14:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    be1b:	00 00 00 
    be1e:	66 90                	xchg   %ax,%ax
    be20:	48 8b 94 c1 00 08 00 	mov    0x800(%rcx,%rax,8),%rdx
    be27:	00 
    be28:	49 89 53 fc          	mov    %rdx,-0x4(%r11)
    be2c:	49 81 c3 00 01 00 00 	add    $0x100,%r11
    be33:	48 ff c0             	inc    %rax
    be36:	75 e8                	jne    be20 <main_graph_mobilenet+0x9b50>
    be38:	48 8b bc 24 88 00 00 	mov    0x88(%rsp),%rdi
    be3f:	00 
    be40:	e8 cb 62 ff ff       	call   2110 <free@plt>
    be45:	48 8b bc 24 90 00 00 	mov    0x90(%rsp),%rdi
    be4c:	00 
    be4d:	e8 be 62 ff ff       	call   2110 <free@plt>
    be52:	48 8b 7c 24 70       	mov    0x70(%rsp),%rdi
    be57:	e8 b4 62 ff ff       	call   2110 <free@plt>
    be5c:	4d 85 ed             	test   %r13,%r13
    be5f:	48 8b bc 24 98 00 00 	mov    0x98(%rsp),%rdi
    be66:	00 
    be67:	0f 8e 96 00 00 00    	jle    bf03 <main_graph_mobilenet+0x9c33>
    be6d:	49 83 fd 0a          	cmp    $0xa,%r13
    be71:	73 04                	jae    be77 <main_graph_mobilenet+0x9ba7>
    be73:	31 c0                	xor    %eax,%eax
    be75:	eb 61                	jmp    bed8 <main_graph_mobilenet+0x9c08>
    be77:	49 8d 4f 04          	lea    0x4(%r15),%rcx
    be7b:	4b 8d 54 ef fc       	lea    -0x4(%r15,%r13,8),%rdx
    be80:	31 c0                	xor    %eax,%eax
    be82:	48 39 ca             	cmp    %rcx,%rdx
    be85:	72 51                	jb     bed8 <main_graph_mobilenet+0x9c08>
    be87:	4b 8d 4c ef f8       	lea    -0x8(%r15,%r13,8),%rcx
    be8c:	4c 39 f9             	cmp    %r15,%rcx
    be8f:	72 47                	jb     bed8 <main_graph_mobilenet+0x9c08>
    be91:	49 8d 4d ff          	lea    -0x1(%r13),%rcx
    be95:	48 ba ff ff ff ff ff 	movabs $0x1fffffffffffffff,%rdx
    be9c:	ff ff 1f 
    be9f:	48 39 d1             	cmp    %rdx,%rcx
    bea2:	77 34                	ja     bed8 <main_graph_mobilenet+0x9c08>
    bea4:	4c 89 e8             	mov    %r13,%rax
    bea7:	48 83 e0 fe          	and    $0xfffffffffffffffe,%rax
    beab:	0f 28 05 4e 22 00 00 	movaps 0x224e(%rip),%xmm0        # e100 <_fini+0x184>
    beb2:	48 89 c1             	mov    %rax,%rcx
    beb5:	4c 89 fa             	mov    %r15,%rdx
    beb8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    bebf:	00 
    bec0:	0f 28 0a             	movaps (%rdx),%xmm1
    bec3:	0f 58 c8             	addps  %xmm0,%xmm1
    bec6:	0f 29 0a             	movaps %xmm1,(%rdx)
    bec9:	48 83 c2 10          	add    $0x10,%rdx
    becd:	48 83 c1 fe          	add    $0xfffffffffffffffe,%rcx
    bed1:	75 ed                	jne    bec0 <main_graph_mobilenet+0x9bf0>
    bed3:	49 39 c5             	cmp    %rax,%r13
    bed6:	74 2b                	je     bf03 <main_graph_mobilenet+0x9c33>
    bed8:	49 8d 0c c7          	lea    (%r15,%rax,8),%rcx
    bedc:	4c 89 ea             	mov    %r13,%rdx
    bedf:	48 29 c2             	sub    %rax,%rdx
    bee2:	0f 28 05 27 22 00 00 	movaps 0x2227(%rip),%xmm0        # e110 <_fini+0x194>
    bee9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    bef0:	f2 0f 10 09          	movsd  (%rcx),%xmm1
    bef4:	0f 58 c8             	addps  %xmm0,%xmm1
    bef7:	0f 13 09             	movlps %xmm1,(%rcx)
    befa:	48 83 c1 08          	add    $0x8,%rcx
    befe:	48 ff ca             	dec    %rdx
    bf01:	75 ed                	jne    bef0 <main_graph_mobilenet+0x9c20>
    bf03:	48 83 c7 10          	add    $0x10,%rdi
    bf07:	e8 c4 61 ff ff       	call   20d0 <malloc@plt>
    bf0c:	49 89 c6             	mov    %rax,%r14
    bf0f:	48 8d 58 0f          	lea    0xf(%rax),%rbx
    bf13:	48 83 e3 f0          	and    $0xfffffffffffffff0,%rbx
    bf17:	4d 85 ed             	test   %r13,%r13
    bf1a:	0f 8e 1e 02 00 00    	jle    c13e <main_graph_mobilenet+0x9e6e>
    bf20:	49 83 fd 04          	cmp    $0x4,%r13
    bf24:	73 08                	jae    bf2e <main_graph_mobilenet+0x9c5e>
    bf26:	45 31 e4             	xor    %r12d,%r12d
    bf29:	e9 b2 01 00 00       	jmp    c0e0 <main_graph_mobilenet+0x9e10>
    bf2e:	4a 8d 44 eb f8       	lea    -0x8(%rbx,%r13,8),%rax
    bf33:	45 31 e4             	xor    %r12d,%r12d
    bf36:	48 39 d8             	cmp    %rbx,%rax
    bf39:	0f 82 a1 01 00 00    	jb     c0e0 <main_graph_mobilenet+0x9e10>
    bf3f:	48 8d 43 04          	lea    0x4(%rbx),%rax
    bf43:	4a 8d 4c eb fc       	lea    -0x4(%rbx,%r13,8),%rcx
    bf48:	48 39 c1             	cmp    %rax,%rcx
    bf4b:	0f 82 8f 01 00 00    	jb     c0e0 <main_graph_mobilenet+0x9e10>
    bf51:	49 8d 45 ff          	lea    -0x1(%r13),%rax
    bf55:	48 b9 ff ff ff ff ff 	movabs $0x1fffffffffffffff,%rcx
    bf5c:	ff ff 1f 
    bf5f:	48 39 c8             	cmp    %rcx,%rax
    bf62:	0f 87 78 01 00 00    	ja     c0e0 <main_graph_mobilenet+0x9e10>
    bf68:	4b 8d 04 ef          	lea    (%r15,%r13,8),%rax
    bf6c:	48 39 d8             	cmp    %rbx,%rax
    bf6f:	76 11                	jbe    bf82 <main_graph_mobilenet+0x9cb2>
    bf71:	4a 8d 04 eb          	lea    (%rbx,%r13,8),%rax
    bf75:	4c 39 f8             	cmp    %r15,%rax
    bf78:	76 08                	jbe    bf82 <main_graph_mobilenet+0x9cb2>
    bf7a:	45 31 e4             	xor    %r12d,%r12d
    bf7d:	e9 5e 01 00 00       	jmp    c0e0 <main_graph_mobilenet+0x9e10>
    bf82:	4c 89 ed             	mov    %r13,%rbp
    bf85:	4d 89 ec             	mov    %r13,%r12
    bf88:	49 83 e4 fc          	and    $0xfffffffffffffffc,%r12
    bf8c:	45 31 ed             	xor    %r13d,%r13d
    bf8f:	90                   	nop
    bf90:	43 0f 28 44 ef 10    	movaps 0x10(%r15,%r13,8),%xmm0
    bf96:	43 0f 28 0c ef       	movaps (%r15,%r13,8),%xmm1
    bf9b:	0f 28 d9             	movaps %xmm1,%xmm3
    bf9e:	0f c6 d8 88          	shufps $0x88,%xmm0,%xmm3
    bfa2:	0f c6 c8 dd          	shufps $0xdd,%xmm0,%xmm1
    bfa6:	0f 29 4c 24 10       	movaps %xmm1,0x10(%rsp)
    bfab:	f3 43 0f 10 44 ef 1c 	movss  0x1c(%r15,%r13,8),%xmm0
    bfb2:	f3 43 0f 10 4c ef 14 	movss  0x14(%r15,%r13,8),%xmm1
    bfb9:	0f 14 c8             	unpcklps %xmm0,%xmm1
    bfbc:	f3 43 0f 10 44 ef 0c 	movss  0xc(%r15,%r13,8),%xmm0
    bfc3:	f3 43 0f 10 54 ef 04 	movss  0x4(%r15,%r13,8),%xmm2
    bfca:	0f 14 d0             	unpcklps %xmm0,%xmm2
    bfcd:	0f 16 d1             	movlhps %xmm1,%xmm2
    bfd0:	0f 28 c3             	movaps %xmm3,%xmm0
    bfd3:	0f 5f c2             	maxps  %xmm2,%xmm0
    bfd6:	0f 29 44 24 60       	movaps %xmm0,0x60(%rsp)
    bfdb:	0f 5c d8             	subps  %xmm0,%xmm3
    bfde:	0f 29 5c 24 30       	movaps %xmm3,0x30(%rsp)
    bfe3:	0f 28 c3             	movaps %xmm3,%xmm0
    bfe6:	0f c6 c3 ff          	shufps $0xff,%xmm3,%xmm0
    bfea:	e8 41 61 ff ff       	call   2130 <expf@plt>
    bfef:	0f 29 44 24 20       	movaps %xmm0,0x20(%rsp)
    bff4:	0f 28 44 24 30       	movaps 0x30(%rsp),%xmm0
    bff9:	0f 12 c0             	movhlps %xmm0,%xmm0
    bffc:	e8 2f 61 ff ff       	call   2130 <expf@plt>
    c001:	0f 14 44 24 20       	unpcklps 0x20(%rsp),%xmm0
    c006:	0f 29 44 24 40       	movaps %xmm0,0x40(%rsp)
    c00b:	0f 28 44 24 30       	movaps 0x30(%rsp),%xmm0
    c010:	e8 1b 61 ff ff       	call   2130 <expf@plt>
    c015:	0f 29 44 24 20       	movaps %xmm0,0x20(%rsp)
    c01a:	0f 28 44 24 30       	movaps 0x30(%rsp),%xmm0
    c01f:	0f c6 c0 55          	shufps $0x55,%xmm0,%xmm0
    c023:	e8 08 61 ff ff       	call   2130 <expf@plt>
    c028:	0f 28 4c 24 20       	movaps 0x20(%rsp),%xmm1
    c02d:	0f 14 c8             	unpcklps %xmm0,%xmm1
    c030:	66 0f 14 4c 24 40    	unpcklpd 0x40(%rsp),%xmm1
    c036:	0f 29 4c 24 20       	movaps %xmm1,0x20(%rsp)
    c03b:	0f 28 44 24 10       	movaps 0x10(%rsp),%xmm0
    c040:	0f 5c 44 24 60       	subps  0x60(%rsp),%xmm0
    c045:	0f 29 44 24 10       	movaps %xmm0,0x10(%rsp)
    c04a:	0f c6 c0 ff          	shufps $0xff,%xmm0,%xmm0
    c04e:	e8 dd 60 ff ff       	call   2130 <expf@plt>
    c053:	0f 29 44 24 30       	movaps %xmm0,0x30(%rsp)
    c058:	0f 28 44 24 10       	movaps 0x10(%rsp),%xmm0
    c05d:	0f 12 c0             	movhlps %xmm0,%xmm0
    c060:	e8 cb 60 ff ff       	call   2130 <expf@plt>
    c065:	0f 14 44 24 30       	unpcklps 0x30(%rsp),%xmm0
    c06a:	0f 29 44 24 30       	movaps %xmm0,0x30(%rsp)
    c06f:	0f 28 44 24 10       	movaps 0x10(%rsp),%xmm0
    c074:	e8 b7 60 ff ff       	call   2130 <expf@plt>
    c079:	0f 29 44 24 60       	movaps %xmm0,0x60(%rsp)
    c07e:	0f 28 44 24 10       	movaps 0x10(%rsp),%xmm0
    c083:	0f c6 c0 55          	shufps $0x55,%xmm0,%xmm0
    c087:	e8 a4 60 ff ff       	call   2130 <expf@plt>
    c08c:	0f 28 54 24 60       	movaps 0x60(%rsp),%xmm2
    c091:	0f 14 d0             	unpcklps %xmm0,%xmm2
    c094:	66 0f 14 54 24 30    	unpcklpd 0x30(%rsp),%xmm2
    c09a:	0f 28 4c 24 20       	movaps 0x20(%rsp),%xmm1
    c09f:	0f 28 c1             	movaps %xmm1,%xmm0
    c0a2:	0f 58 c2             	addps  %xmm2,%xmm0
    c0a5:	0f 5e c8             	divps  %xmm0,%xmm1
    c0a8:	0f 5e d0             	divps  %xmm0,%xmm2
    c0ab:	0f 28 c1             	movaps %xmm1,%xmm0
    c0ae:	0f 14 c2             	unpcklps %xmm2,%xmm0
    c0b1:	0f 15 ca             	unpckhps %xmm2,%xmm1
    c0b4:	42 0f 29 4c eb 10    	movaps %xmm1,0x10(%rbx,%r13,8)
    c0ba:	42 0f 29 04 eb       	movaps %xmm0,(%rbx,%r13,8)
    c0bf:	49 83 c5 04          	add    $0x4,%r13
    c0c3:	4d 39 ec             	cmp    %r13,%r12
    c0c6:	0f 85 c4 fe ff ff    	jne    bf90 <main_graph_mobilenet+0x9cc0>
    c0cc:	49 89 ed             	mov    %rbp,%r13
    c0cf:	4c 39 e5             	cmp    %r12,%rbp
    c0d2:	74 6a                	je     c13e <main_graph_mobilenet+0x9e6e>
    c0d4:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    c0db:	00 00 00 
    c0de:	66 90                	xchg   %ax,%ax
    c0e0:	f3 43 0f 10 04 e7    	movss  (%r15,%r12,8),%xmm0
    c0e6:	0f 28 c8             	movaps %xmm0,%xmm1
    c0e9:	f3 43 0f 5f 4c e7 04 	maxss  0x4(%r15,%r12,8),%xmm1
    c0f0:	f3 0f 11 4c 24 10    	movss  %xmm1,0x10(%rsp)
    c0f6:	f3 0f 5c c1          	subss  %xmm1,%xmm0
    c0fa:	e8 31 60 ff ff       	call   2130 <expf@plt>
    c0ff:	0f 29 44 24 20       	movaps %xmm0,0x20(%rsp)
    c104:	f3 42 0f 11 04 e3    	movss  %xmm0,(%rbx,%r12,8)
    c10a:	f3 43 0f 10 44 e7 04 	movss  0x4(%r15,%r12,8),%xmm0
    c111:	f3 0f 5c 44 24 10    	subss  0x10(%rsp),%xmm0
    c117:	e8 14 60 ff ff       	call   2130 <expf@plt>
    c11c:	0f 28 54 24 20       	movaps 0x20(%rsp),%xmm2
    c121:	0f 28 ca             	movaps %xmm2,%xmm1
    c124:	f3 0f 58 c8          	addss  %xmm0,%xmm1
    c128:	0f 14 d0             	unpcklps %xmm0,%xmm2
    c12b:	0f 14 c9             	unpcklps %xmm1,%xmm1
    c12e:	0f 5e d1             	divps  %xmm1,%xmm2
    c131:	42 0f 13 14 e3       	movlps %xmm2,(%rbx,%r12,8)
    c136:	49 ff c4             	inc    %r12
    c139:	4d 39 e5             	cmp    %r12,%r13
    c13c:	75 a2                	jne    c0e0 <main_graph_mobilenet+0x9e10>
    c13e:	48 8b 7c 24 78       	mov    0x78(%rsp),%rdi
    c143:	e8 c8 5f ff ff       	call   2110 <free@plt>
    c148:	48 8b 84 24 28 02 00 	mov    0x228(%rsp),%rax
    c14f:	00 
    c150:	4c 89 68 18          	mov    %r13,0x18(%rax)
    c154:	48 89 58 08          	mov    %rbx,0x8(%rax)
    c158:	4c 89 30             	mov    %r14,(%rax)
    c15b:	48 c7 40 30 01 00 00 	movq   $0x1,0x30(%rax)
    c162:	00 
    c163:	48 c7 40 28 02 00 00 	movq   $0x2,0x28(%rax)
    c16a:	00 
    c16b:	48 c7 40 20 02 00 00 	movq   $0x2,0x20(%rax)
    c172:	00 
    c173:	48 c7 40 10 00 00 00 	movq   $0x0,0x10(%rax)
    c17a:	00 
    c17b:	48 81 c4 b8 02 00 00 	add    $0x2b8,%rsp
    c182:	5b                   	pop    %rbx
    c183:	41 5c                	pop    %r12
    c185:	41 5d                	pop    %r13
    c187:	41 5e                	pop    %r14
    c189:	41 5f                	pop    %r15
    c18b:	5d                   	pop    %rbp
    c18c:	c3                   	ret    
    c18d:	0f 1f 00             	nopl   (%rax)

000000000000c190 <_mlir_ciface_main_graph_mobilenet>:
    c190:	53                   	push   %rbx
    c191:	48 83 ec 70          	sub    $0x70,%rsp
    c195:	48 89 fb             	mov    %rdi,%rbx
    c198:	48 8b 56 08          	mov    0x8(%rsi),%rdx
    c19c:	4c 8b 46 18          	mov    0x18(%rsi),%r8
    c1a0:	48 8d 7c 24 38       	lea    0x38(%rsp),%rdi
    c1a5:	e8 c6 5f ff ff       	call   2170 <main_graph_mobilenet@plt>
    c1aa:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
    c1af:	48 8b 4c 24 60       	mov    0x60(%rsp),%rcx
    c1b4:	48 8b 54 24 58       	mov    0x58(%rsp),%rdx
    c1b9:	48 8b 74 24 50       	mov    0x50(%rsp),%rsi
    c1be:	48 8b 7c 24 48       	mov    0x48(%rsp),%rdi
    c1c3:	4c 8b 44 24 38       	mov    0x38(%rsp),%r8
    c1c8:	4c 8b 4c 24 40       	mov    0x40(%rsp),%r9
    c1cd:	4c 89 03             	mov    %r8,(%rbx)
    c1d0:	4c 89 4b 08          	mov    %r9,0x8(%rbx)
    c1d4:	48 89 7b 10          	mov    %rdi,0x10(%rbx)
    c1d8:	48 89 73 18          	mov    %rsi,0x18(%rbx)
    c1dc:	48 89 53 20          	mov    %rdx,0x20(%rbx)
    c1e0:	48 89 4b 28          	mov    %rcx,0x28(%rbx)
    c1e4:	48 89 43 30          	mov    %rax,0x30(%rbx)
    c1e8:	48 83 c4 70          	add    $0x70,%rsp
    c1ec:	5b                   	pop    %rbx
    c1ed:	c3                   	ret    
    c1ee:	66 90                	xchg   %ax,%ax

000000000000c1f0 <run_main_graph_mobilenet>:
    c1f0:	55                   	push   %rbp
    c1f1:	41 57                	push   %r15
    c1f3:	41 56                	push   %r14
    c1f5:	41 55                	push   %r13
    c1f7:	41 54                	push   %r12
    c1f9:	53                   	push   %rbx
    c1fa:	48 83 ec 78          	sub    $0x78,%rsp
    c1fe:	e8 8d 5e ff ff       	call   2090 <omTensorListGetOmtArray@plt>
    c203:	48 8b 18             	mov    (%rax),%rbx
    c206:	48 89 df             	mov    %rbx,%rdi
    c209:	e8 f2 5e ff ff       	call   2100 <omTensorGetDataPtr@plt>
    c20e:	49 89 c6             	mov    %rax,%r14
    c211:	48 89 df             	mov    %rbx,%rdi
    c214:	e8 b7 5f ff ff       	call   21d0 <omTensorGetShape@plt>
    c219:	49 89 c7             	mov    %rax,%r15
    c21c:	48 89 df             	mov    %rbx,%rdi
    c21f:	e8 4c 5e ff ff       	call   2070 <omTensorGetStrides@plt>
    c224:	4d 8b 07             	mov    (%r15),%r8
    c227:	48 8d 7c 24 40       	lea    0x40(%rsp),%rdi
    c22c:	4c 89 f2             	mov    %r14,%rdx
    c22f:	e8 3c 5f ff ff       	call   2170 <main_graph_mobilenet@plt>
    c234:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
    c239:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
    c23e:	4c 8b 6c 24 60       	mov    0x60(%rsp),%r13
    c243:	48 8b 6c 24 68       	mov    0x68(%rsp),%rbp
    c248:	4c 8b 64 24 58       	mov    0x58(%rsp),%r12
    c24d:	48 8b 5c 24 40       	mov    0x40(%rsp),%rbx
    c252:	4c 8b 74 24 48       	mov    0x48(%rsp),%r14
    c257:	bf 02 00 00 00       	mov    $0x2,%edi
    c25c:	e8 3f 5f ff ff       	call   21a0 <omTensorCreateUntyped@plt>
    c261:	49 89 c7             	mov    %rax,%r15
    c264:	be 01 00 00 00       	mov    $0x1,%esi
    c269:	48 89 c7             	mov    %rax,%rdi
    c26c:	48 89 da             	mov    %rbx,%rdx
    c26f:	4c 89 f1             	mov    %r14,%rcx
    c272:	e8 79 5e ff ff       	call   20f0 <omTensorSetDataPtr@plt>
    c277:	be 01 00 00 00       	mov    $0x1,%esi
    c27c:	4c 89 ff             	mov    %r15,%rdi
    c27f:	e8 1c 5e ff ff       	call   20a0 <omTensorSetDataType@plt>
    c284:	4c 89 ff             	mov    %r15,%rdi
    c287:	e8 44 5f ff ff       	call   21d0 <omTensorGetShape@plt>
    c28c:	48 89 c3             	mov    %rax,%rbx
    c28f:	4c 89 ff             	mov    %r15,%rdi
    c292:	e8 d9 5d ff ff       	call   2070 <omTensorGetStrides@plt>
    c297:	4c 89 23             	mov    %r12,(%rbx)
    c29a:	48 89 28             	mov    %rbp,(%rax)
    c29d:	4c 89 6b 08          	mov    %r13,0x8(%rbx)
    c2a1:	48 8b 4c 24 30       	mov    0x30(%rsp),%rcx
    c2a6:	48 89 48 08          	mov    %rcx,0x8(%rax)
    c2aa:	4c 89 7c 24 38       	mov    %r15,0x38(%rsp)
    c2af:	48 8d 7c 24 38       	lea    0x38(%rsp),%rdi
    c2b4:	be 01 00 00 00       	mov    $0x1,%esi
    c2b9:	e8 22 5e ff ff       	call   20e0 <omTensorListCreate@plt>
    c2be:	48 83 c4 78          	add    $0x78,%rsp
    c2c2:	5b                   	pop    %rbx
    c2c3:	41 5c                	pop    %r12
    c2c5:	41 5d                	pop    %r13
    c2c7:	41 5e                	pop    %r14
    c2c9:	41 5f                	pop    %r15
    c2cb:	5d                   	pop    %rbp
    c2cc:	c3                   	ret    
    c2cd:	0f 1f 00             	nopl   (%rax)

000000000000c2d0 <run_main_graph>:
    c2d0:	e9 eb 5e ff ff       	jmp    21c0 <run_main_graph_mobilenet@plt>
    c2d5:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    c2dc:	00 00 00 
    c2df:	90                   	nop

000000000000c2e0 <omQueryEntryPoints_mobilenet>:
    c2e0:	48 85 ff             	test   %rdi,%rdi
    c2e3:	74 07                	je     c2ec <omQueryEntryPoints_mobilenet+0xc>
    c2e5:	48 c7 07 02 00 00 00 	movq   $0x2,(%rdi)
    c2ec:	48 8d 05 9d 2a 0d 00 	lea    0xd2a9d(%rip),%rax        # ded90 <_entry_point_0_mobilenet@@Base+0xd0b7c>
    c2f3:	c3                   	ret    
    c2f4:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    c2fb:	00 00 00 
    c2fe:	66 90                	xchg   %ax,%ax

000000000000c300 <omQueryEntryPoints>:
    c300:	48 85 ff             	test   %rdi,%rdi
    c303:	74 07                	je     c30c <omQueryEntryPoints+0xc>
    c305:	48 c7 07 02 00 00 00 	movq   $0x2,(%rdi)
    c30c:	48 8d 05 7d 2a 0d 00 	lea    0xd2a7d(%rip),%rax        # ded90 <_entry_point_0_mobilenet@@Base+0xd0b7c>
    c313:	c3                   	ret    
    c314:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
    c31b:	00 00 00 
    c31e:	66 90                	xchg   %ax,%ax

000000000000c320 <omInputSignature_mobilenet>:
    c320:	53                   	push   %rbx
    c321:	48 89 fb             	mov    %rdi,%rbx
    c324:	48 8b 35 95 2c 0d 00 	mov    0xd2c95(%rip),%rsi        # defc0 <_entry_point_0_mobilenet@@Base+0xd0dac>
    c32b:	ba 0f 00 00 00       	mov    $0xf,%edx
    c330:	e8 8b 5d ff ff       	call   20c0 <strncmp@plt>
    c335:	85 c0                	test   %eax,%eax
    c337:	74 21                	je     c35a <omInputSignature_mobilenet+0x3a>
    c339:	48 8b 35 b0 2c 0d 00 	mov    0xd2cb0(%rip),%rsi        # deff0 <_entry_point_1_mobilenet@@Base+0xd0e90>
    c340:	ba 19 00 00 00       	mov    $0x19,%edx
    c345:	48 89 df             	mov    %rbx,%rdi
    c348:	e8 73 5d ff ff       	call   20c0 <strncmp@plt>
    c34d:	85 c0                	test   %eax,%eax
    c34f:	75 12                	jne    c363 <omInputSignature_mobilenet+0x43>
    c351:	48 8b 05 88 2c 0d 00 	mov    0xd2c88(%rip),%rax        # defe0 <_entry_point_1_in_sig_mobilenet@@Base+0xd0e60>
    c358:	5b                   	pop    %rbx
    c359:	c3                   	ret    
    c35a:	48 8b 05 67 2c 0d 00 	mov    0xd2c67(%rip),%rax        # defc8 <_entry_point_0_in_sig_mobilenet@@Base+0xd0d98>
    c361:	5b                   	pop    %rbx
    c362:	c3                   	ret    
    c363:	31 c0                	xor    %eax,%eax
    c365:	5b                   	pop    %rbx
    c366:	c3                   	ret    
    c367:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    c36e:	00 00 

000000000000c370 <omInputSignature>:
    c370:	53                   	push   %rbx
    c371:	48 89 fb             	mov    %rdi,%rbx
    c374:	48 8b 35 45 2c 0d 00 	mov    0xd2c45(%rip),%rsi        # defc0 <_entry_point_0_mobilenet@@Base+0xd0dac>
    c37b:	ba 0f 00 00 00       	mov    $0xf,%edx
    c380:	e8 3b 5d ff ff       	call   20c0 <strncmp@plt>
    c385:	85 c0                	test   %eax,%eax
    c387:	74 21                	je     c3aa <omInputSignature+0x3a>
    c389:	48 8b 35 60 2c 0d 00 	mov    0xd2c60(%rip),%rsi        # deff0 <_entry_point_1_mobilenet@@Base+0xd0e90>
    c390:	ba 19 00 00 00       	mov    $0x19,%edx
    c395:	48 89 df             	mov    %rbx,%rdi
    c398:	e8 23 5d ff ff       	call   20c0 <strncmp@plt>
    c39d:	85 c0                	test   %eax,%eax
    c39f:	75 12                	jne    c3b3 <omInputSignature+0x43>
    c3a1:	48 8b 05 38 2c 0d 00 	mov    0xd2c38(%rip),%rax        # defe0 <_entry_point_1_in_sig_mobilenet@@Base+0xd0e60>
    c3a8:	5b                   	pop    %rbx
    c3a9:	c3                   	ret    
    c3aa:	48 8b 05 17 2c 0d 00 	mov    0xd2c17(%rip),%rax        # defc8 <_entry_point_0_in_sig_mobilenet@@Base+0xd0d98>
    c3b1:	5b                   	pop    %rbx
    c3b2:	c3                   	ret    
    c3b3:	31 c0                	xor    %eax,%eax
    c3b5:	5b                   	pop    %rbx
    c3b6:	c3                   	ret    
    c3b7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    c3be:	00 00 

000000000000c3c0 <omOutputSignature_mobilenet>:
    c3c0:	53                   	push   %rbx
    c3c1:	48 89 fb             	mov    %rdi,%rbx
    c3c4:	48 8b 35 f5 2b 0d 00 	mov    0xd2bf5(%rip),%rsi        # defc0 <_entry_point_0_mobilenet@@Base+0xd0dac>
    c3cb:	ba 0f 00 00 00       	mov    $0xf,%edx
    c3d0:	e8 eb 5c ff ff       	call   20c0 <strncmp@plt>
    c3d5:	85 c0                	test   %eax,%eax
    c3d7:	74 21                	je     c3fa <omOutputSignature_mobilenet+0x3a>
    c3d9:	48 8b 35 10 2c 0d 00 	mov    0xd2c10(%rip),%rsi        # deff0 <_entry_point_1_mobilenet@@Base+0xd0e90>
    c3e0:	ba 19 00 00 00       	mov    $0x19,%edx
    c3e5:	48 89 df             	mov    %rbx,%rdi
    c3e8:	e8 d3 5c ff ff       	call   20c0 <strncmp@plt>
    c3ed:	85 c0                	test   %eax,%eax
    c3ef:	75 12                	jne    c403 <omOutputSignature_mobilenet+0x43>
    c3f1:	48 8b 05 f0 2b 0d 00 	mov    0xd2bf0(%rip),%rax        # defe8 <_entry_point_1_out_sig_mobilenet@@Base+0xd0e18>
    c3f8:	5b                   	pop    %rbx
    c3f9:	c3                   	ret    
    c3fa:	48 8b 05 97 2b 0d 00 	mov    0xd2b97(%rip),%rax        # def98 <_entry_point_0_out_sig_mobilenet@@Base+0xd0d18>
    c401:	5b                   	pop    %rbx
    c402:	c3                   	ret    
    c403:	31 c0                	xor    %eax,%eax
    c405:	5b                   	pop    %rbx
    c406:	c3                   	ret    
    c407:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    c40e:	00 00 

000000000000c410 <omOutputSignature>:
    c410:	53                   	push   %rbx
    c411:	48 89 fb             	mov    %rdi,%rbx
    c414:	48 8b 35 a5 2b 0d 00 	mov    0xd2ba5(%rip),%rsi        # defc0 <_entry_point_0_mobilenet@@Base+0xd0dac>
    c41b:	ba 0f 00 00 00       	mov    $0xf,%edx
    c420:	e8 9b 5c ff ff       	call   20c0 <strncmp@plt>
    c425:	85 c0                	test   %eax,%eax
    c427:	74 21                	je     c44a <omOutputSignature+0x3a>
    c429:	48 8b 35 c0 2b 0d 00 	mov    0xd2bc0(%rip),%rsi        # deff0 <_entry_point_1_mobilenet@@Base+0xd0e90>
    c430:	ba 19 00 00 00       	mov    $0x19,%edx
    c435:	48 89 df             	mov    %rbx,%rdi
    c438:	e8 83 5c ff ff       	call   20c0 <strncmp@plt>
    c43d:	85 c0                	test   %eax,%eax
    c43f:	75 12                	jne    c453 <omOutputSignature+0x43>
    c441:	48 8b 05 a0 2b 0d 00 	mov    0xd2ba0(%rip),%rax        # defe8 <_entry_point_1_out_sig_mobilenet@@Base+0xd0e18>
    c448:	5b                   	pop    %rbx
    c449:	c3                   	ret    
    c44a:	48 8b 05 47 2b 0d 00 	mov    0xd2b47(%rip),%rax        # def98 <_entry_point_0_out_sig_mobilenet@@Base+0xd0d18>
    c451:	5b                   	pop    %rbx
    c452:	c3                   	ret    
    c453:	31 c0                	xor    %eax,%eax
    c455:	5b                   	pop    %rbx
    c456:	c3                   	ret    

000000000000c457 <getDataTypeSize>:
    c457:	55                   	push   %rbp
    c458:	48 89 e5             	mov    %rsp,%rbp
    c45b:	89 7d fc             	mov    %edi,-0x4(%rbp)
    c45e:	48 8b 05 73 2b 0d 00 	mov    0xd2b73(%rip),%rax        # defd8 <OM_DATA_TYPE_SIZE@@Base+0x2c18>
    c465:	8b 55 fc             	mov    -0x4(%rbp),%edx
    c468:	8b 04 90             	mov    (%rax,%rdx,4),%eax
    c46b:	48 98                	cltq   
    c46d:	5d                   	pop    %rbp
    c46e:	c3                   	ret    

000000000000c46f <__extendhfsf2>:
    c46f:	f3 0f 1e fa          	endbr64 
    c473:	55                   	push   %rbp
    c474:	48 89 e5             	mov    %rsp,%rbp
    c477:	48 83 ec 10          	sub    $0x10,%rsp
    c47b:	89 f8                	mov    %edi,%eax
    c47d:	66 89 45 fc          	mov    %ax,-0x4(%rbp)
    c481:	0f b7 45 fc          	movzwl -0x4(%rbp),%eax
    c485:	89 c7                	mov    %eax,%edi
    c487:	e8 b4 5c ff ff       	call   2140 <om_f16_to_f32@plt>
    c48c:	c9                   	leave  
    c48d:	c3                   	ret    

000000000000c48e <__gnu_h2f_ieee>:
    c48e:	f3 0f 1e fa          	endbr64 
    c492:	55                   	push   %rbp
    c493:	48 89 e5             	mov    %rsp,%rbp
    c496:	48 83 ec 10          	sub    $0x10,%rsp
    c49a:	89 f8                	mov    %edi,%eax
    c49c:	66 89 45 fc          	mov    %ax,-0x4(%rbp)
    c4a0:	0f b7 45 fc          	movzwl -0x4(%rbp),%eax
    c4a4:	89 c7                	mov    %eax,%edi
    c4a6:	e8 95 5c ff ff       	call   2140 <om_f16_to_f32@plt>
    c4ab:	c9                   	leave  
    c4ac:	c3                   	ret    

000000000000c4ad <__gnu_f2h_ieee>:
    c4ad:	f3 0f 1e fa          	endbr64 
    c4b1:	55                   	push   %rbp
    c4b2:	48 89 e5             	mov    %rsp,%rbp
    c4b5:	48 83 ec 10          	sub    $0x10,%rsp
    c4b9:	f3 0f 11 45 fc       	movss  %xmm0,-0x4(%rbp)
    c4be:	8b 45 fc             	mov    -0x4(%rbp),%eax
    c4c1:	66 0f 6e c0          	movd   %eax,%xmm0
    c4c5:	e8 86 5c ff ff       	call   2150 <om_f32_to_f16@plt>
    c4ca:	c9                   	leave  
    c4cb:	c3                   	ret    

000000000000c4cc <__truncsfhf2>:
    c4cc:	f3 0f 1e fa          	endbr64 
    c4d0:	55                   	push   %rbp
    c4d1:	48 89 e5             	mov    %rsp,%rbp
    c4d4:	48 83 ec 10          	sub    $0x10,%rsp
    c4d8:	f3 0f 11 45 fc       	movss  %xmm0,-0x4(%rbp)
    c4dd:	8b 45 fc             	mov    -0x4(%rbp),%eax
    c4e0:	66 0f 6e c0          	movd   %eax,%xmm0
    c4e4:	e8 67 5c ff ff       	call   2150 <om_f32_to_f16@plt>
    c4e9:	c9                   	leave  
    c4ea:	c3                   	ret    

000000000000c4eb <__truncdfhf2>:
    c4eb:	f3 0f 1e fa          	endbr64 
    c4ef:	55                   	push   %rbp
    c4f0:	48 89 e5             	mov    %rsp,%rbp
    c4f3:	48 83 ec 10          	sub    $0x10,%rsp
    c4f7:	f2 0f 11 45 f8       	movsd  %xmm0,-0x8(%rbp)
    c4fc:	66 0f ef c9          	pxor   %xmm1,%xmm1
    c500:	f2 0f 5a 4d f8       	cvtsd2ss -0x8(%rbp),%xmm1
    c505:	66 0f 7e c8          	movd   %xmm1,%eax
    c509:	66 0f 6e c0          	movd   %eax,%xmm0
    c50d:	e8 3e 5c ff ff       	call   2150 <om_f32_to_f16@plt>
    c512:	c9                   	leave  
    c513:	c3                   	ret    

000000000000c514 <getNumElems>:
    c514:	55                   	push   %rbp
    c515:	48 89 e5             	mov    %rsp,%rbp
    c518:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    c51c:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    c520:	48 c7 45 f0 01 00 00 	movq   $0x1,-0x10(%rbp)
    c527:	00 
    c528:	48 c7 45 f8 00 00 00 	movq   $0x0,-0x8(%rbp)
    c52f:	00 
    c530:	eb 27                	jmp    c559 <getNumElems+0x45>
    c532:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c536:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
    c53d:	00 
    c53e:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    c542:	48 01 d0             	add    %rdx,%rax
    c545:	48 8b 00             	mov    (%rax),%rax
    c548:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    c54c:	48 0f af c2          	imul   %rdx,%rax
    c550:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
    c554:	48 83 45 f8 01       	addq   $0x1,-0x8(%rbp)
    c559:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c55d:	48 3b 45 e0          	cmp    -0x20(%rbp),%rax
    c561:	7c cf                	jl     c532 <getNumElems+0x1e>
    c563:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    c567:	5d                   	pop    %rbp
    c568:	c3                   	ret    

000000000000c569 <getDataTypeName>:
    c569:	55                   	push   %rbp
    c56a:	48 89 e5             	mov    %rsp,%rbp
    c56d:	89 7d fc             	mov    %edi,-0x4(%rbp)
    c570:	48 8b 05 59 2a 0d 00 	mov    0xd2a59(%rip),%rax        # defd0 <OM_DATA_TYPE_NAME@@Base-0x150>
    c577:	8b 55 fc             	mov    -0x4(%rbp),%edx
    c57a:	48 8b 04 d0          	mov    (%rax,%rdx,8),%rax
    c57e:	5d                   	pop    %rbp
    c57f:	c3                   	ret    

000000000000c580 <printElement>:
    c580:	55                   	push   %rbp
    c581:	48 89 e5             	mov    %rsp,%rbp
    c584:	48 83 ec 20          	sub    $0x20,%rsp
    c588:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    c58c:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    c590:	89 55 ec             	mov    %edx,-0x14(%rbp)
    c593:	48 83 7d f8 00       	cmpq   $0x0,-0x8(%rbp)
    c598:	75 28                	jne    c5c2 <printElement+0x42>
    c59a:	48 8d 05 8f fd 0c 00 	lea    0xcfd8f(%rip),%rax        # dc330 <__PRETTY_FUNCTION__.0>
    c5a1:	48 89 c1             	mov    %rax,%rcx
    c5a4:	ba 98 00 00 00       	mov    $0x98,%edx
    c5a9:	48 8d 05 60 fb 0c 00 	lea    0xcfb60(%rip),%rax        # dc110 <constant_11_mobilenet+0x200>
    c5b0:	48 89 c6             	mov    %rax,%rsi
    c5b3:	48 8d 05 96 fb 0c 00 	lea    0xcfb96(%rip),%rax        # dc150 <constant_11_mobilenet+0x240>
    c5ba:	48 89 c7             	mov    %rax,%rdi
    c5bd:	e8 be 5a ff ff       	call   2080 <__assert_fail@plt>
    c5c2:	48 83 7d f0 00       	cmpq   $0x0,-0x10(%rbp)
    c5c7:	79 28                	jns    c5f1 <printElement+0x71>
    c5c9:	48 8d 05 60 fd 0c 00 	lea    0xcfd60(%rip),%rax        # dc330 <__PRETTY_FUNCTION__.0>
    c5d0:	48 89 c1             	mov    %rax,%rcx
    c5d3:	ba 99 00 00 00       	mov    $0x99,%edx
    c5d8:	48 8d 05 31 fb 0c 00 	lea    0xcfb31(%rip),%rax        # dc110 <constant_11_mobilenet+0x200>
    c5df:	48 89 c6             	mov    %rax,%rsi
    c5e2:	48 8d 05 97 fb 0c 00 	lea    0xcfb97(%rip),%rax        # dc180 <constant_11_mobilenet+0x270>
    c5e9:	48 89 c7             	mov    %rax,%rdi
    c5ec:	e8 8f 5a ff ff       	call   2080 <__assert_fail@plt>
    c5f1:	83 7d ec 0d          	cmpl   $0xd,-0x14(%rbp)
    c5f5:	0f 87 b3 02 00 00    	ja     c8ae <printElement+0x32e>
    c5fb:	8b 45 ec             	mov    -0x14(%rbp),%eax
    c5fe:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    c605:	00 
    c606:	48 8d 05 f3 fb 0c 00 	lea    0xcfbf3(%rip),%rax        # dc200 <constant_11_mobilenet+0x2f0>
    c60d:	8b 04 02             	mov    (%rdx,%rax,1),%eax
    c610:	48 98                	cltq   
    c612:	48 8d 15 e7 fb 0c 00 	lea    0xcfbe7(%rip),%rdx        # dc200 <constant_11_mobilenet+0x2f0>
    c619:	48 01 d0             	add    %rdx,%rax
    c61c:	3e ff e0             	notrack jmp *%rax
    c61f:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    c623:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c627:	48 01 d0             	add    %rdx,%rax
    c62a:	0f b6 00             	movzbl (%rax),%eax
    c62d:	0f b6 c0             	movzbl %al,%eax
    c630:	89 c6                	mov    %eax,%esi
    c632:	48 8d 05 7d fb 0c 00 	lea    0xcfb7d(%rip),%rax        # dc1b6 <constant_11_mobilenet+0x2a6>
    c639:	48 89 c7             	mov    %rax,%rdi
    c63c:	b8 00 00 00 00       	mov    $0x0,%eax
    c641:	e8 ea 59 ff ff       	call   2030 <printf@plt>
    c646:	e9 8b 02 00 00       	jmp    c8d6 <printElement+0x356>
    c64b:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    c64f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c653:	48 01 d0             	add    %rdx,%rax
    c656:	0f b6 00             	movzbl (%rax),%eax
    c659:	0f b6 c0             	movzbl %al,%eax
    c65c:	89 c6                	mov    %eax,%esi
    c65e:	48 8d 05 54 fb 0c 00 	lea    0xcfb54(%rip),%rax        # dc1b9 <constant_11_mobilenet+0x2a9>
    c665:	48 89 c7             	mov    %rax,%rdi
    c668:	b8 00 00 00 00       	mov    $0x0,%eax
    c66d:	e8 be 59 ff ff       	call   2030 <printf@plt>
    c672:	e9 5f 02 00 00       	jmp    c8d6 <printElement+0x356>
    c677:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    c67b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c67f:	48 01 d0             	add    %rdx,%rax
    c682:	0f b6 00             	movzbl (%rax),%eax
    c685:	0f be c0             	movsbl %al,%eax
    c688:	89 c6                	mov    %eax,%esi
    c68a:	48 8d 05 2d fb 0c 00 	lea    0xcfb2d(%rip),%rax        # dc1be <constant_11_mobilenet+0x2ae>
    c691:	48 89 c7             	mov    %rax,%rdi
    c694:	b8 00 00 00 00       	mov    $0x0,%eax
    c699:	e8 92 59 ff ff       	call   2030 <printf@plt>
    c69e:	e9 33 02 00 00       	jmp    c8d6 <printElement+0x356>
    c6a3:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    c6a7:	48 8d 14 00          	lea    (%rax,%rax,1),%rdx
    c6ab:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c6af:	48 01 d0             	add    %rdx,%rax
    c6b2:	0f b7 00             	movzwl (%rax),%eax
    c6b5:	0f b7 c0             	movzwl %ax,%eax
    c6b8:	89 c6                	mov    %eax,%esi
    c6ba:	48 8d 05 02 fb 0c 00 	lea    0xcfb02(%rip),%rax        # dc1c3 <constant_11_mobilenet+0x2b3>
    c6c1:	48 89 c7             	mov    %rax,%rdi
    c6c4:	b8 00 00 00 00       	mov    $0x0,%eax
    c6c9:	e8 62 59 ff ff       	call   2030 <printf@plt>
    c6ce:	e9 03 02 00 00       	jmp    c8d6 <printElement+0x356>
    c6d3:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    c6d7:	48 8d 14 00          	lea    (%rax,%rax,1),%rdx
    c6db:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c6df:	48 01 d0             	add    %rdx,%rax
    c6e2:	0f b7 00             	movzwl (%rax),%eax
    c6e5:	98                   	cwtl   
    c6e6:	89 c6                	mov    %eax,%esi
    c6e8:	48 8d 05 d8 fa 0c 00 	lea    0xcfad8(%rip),%rax        # dc1c7 <constant_11_mobilenet+0x2b7>
    c6ef:	48 89 c7             	mov    %rax,%rdi
    c6f2:	b8 00 00 00 00       	mov    $0x0,%eax
    c6f7:	e8 34 59 ff ff       	call   2030 <printf@plt>
    c6fc:	e9 d5 01 00 00       	jmp    c8d6 <printElement+0x356>
    c701:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    c705:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    c70c:	00 
    c70d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c711:	48 01 d0             	add    %rdx,%rax
    c714:	8b 00                	mov    (%rax),%eax
    c716:	89 c6                	mov    %eax,%esi
    c718:	48 8d 05 ac fa 0c 00 	lea    0xcfaac(%rip),%rax        # dc1cb <constant_11_mobilenet+0x2bb>
    c71f:	48 89 c7             	mov    %rax,%rdi
    c722:	b8 00 00 00 00       	mov    $0x0,%eax
    c727:	e8 04 59 ff ff       	call   2030 <printf@plt>
    c72c:	e9 a5 01 00 00       	jmp    c8d6 <printElement+0x356>
    c731:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    c735:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    c73c:	00 
    c73d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c741:	48 01 d0             	add    %rdx,%rax
    c744:	8b 00                	mov    (%rax),%eax
    c746:	89 c6                	mov    %eax,%esi
    c748:	48 8d 05 67 fa 0c 00 	lea    0xcfa67(%rip),%rax        # dc1b6 <constant_11_mobilenet+0x2a6>
    c74f:	48 89 c7             	mov    %rax,%rdi
    c752:	b8 00 00 00 00       	mov    $0x0,%eax
    c757:	e8 d4 58 ff ff       	call   2030 <printf@plt>
    c75c:	e9 75 01 00 00       	jmp    c8d6 <printElement+0x356>
    c761:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    c765:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
    c76c:	00 
    c76d:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c771:	48 01 d0             	add    %rdx,%rax
    c774:	48 8b 00             	mov    (%rax),%rax
    c777:	48 89 c6             	mov    %rax,%rsi
    c77a:	48 8d 05 4d fa 0c 00 	lea    0xcfa4d(%rip),%rax        # dc1ce <constant_11_mobilenet+0x2be>
    c781:	48 89 c7             	mov    %rax,%rdi
    c784:	b8 00 00 00 00       	mov    $0x0,%eax
    c789:	e8 a2 58 ff ff       	call   2030 <printf@plt>
    c78e:	e9 43 01 00 00       	jmp    c8d6 <printElement+0x356>
    c793:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    c797:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
    c79e:	00 
    c79f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c7a3:	48 01 d0             	add    %rdx,%rax
    c7a6:	48 8b 00             	mov    (%rax),%rax
    c7a9:	48 89 c6             	mov    %rax,%rsi
    c7ac:	48 8d 05 20 fa 0c 00 	lea    0xcfa20(%rip),%rax        # dc1d3 <constant_11_mobilenet+0x2c3>
    c7b3:	48 89 c7             	mov    %rax,%rdi
    c7b6:	b8 00 00 00 00       	mov    $0x0,%eax
    c7bb:	e8 70 58 ff ff       	call   2030 <printf@plt>
    c7c0:	e9 11 01 00 00       	jmp    c8d6 <printElement+0x356>
    c7c5:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    c7c9:	48 8d 14 00          	lea    (%rax,%rax,1),%rdx
    c7cd:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c7d1:	48 01 d0             	add    %rdx,%rax
    c7d4:	0f b7 00             	movzwl (%rax),%eax
    c7d7:	0f b7 c0             	movzwl %ax,%eax
    c7da:	89 c7                	mov    %eax,%edi
    c7dc:	e8 5f 59 ff ff       	call   2140 <om_f16_to_f32@plt>
    c7e1:	66 0f ef c9          	pxor   %xmm1,%xmm1
    c7e5:	f3 0f 5a c8          	cvtss2sd %xmm0,%xmm1
    c7e9:	66 48 0f 7e c8       	movq   %xmm1,%rax
    c7ee:	66 48 0f 6e c0       	movq   %rax,%xmm0
    c7f3:	48 8d 05 de f9 0c 00 	lea    0xcf9de(%rip),%rax        # dc1d8 <constant_11_mobilenet+0x2c8>
    c7fa:	48 89 c7             	mov    %rax,%rdi
    c7fd:	b8 01 00 00 00       	mov    $0x1,%eax
    c802:	e8 29 58 ff ff       	call   2030 <printf@plt>
    c807:	e9 ca 00 00 00       	jmp    c8d6 <printElement+0x356>
    c80c:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    c810:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    c817:	00 
    c818:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c81c:	48 01 d0             	add    %rdx,%rax
    c81f:	f3 0f 10 00          	movss  (%rax),%xmm0
    c823:	66 0f ef d2          	pxor   %xmm2,%xmm2
    c827:	f3 0f 5a d0          	cvtss2sd %xmm0,%xmm2
    c82b:	66 48 0f 7e d0       	movq   %xmm2,%rax
    c830:	66 48 0f 6e c0       	movq   %rax,%xmm0
    c835:	48 8d 05 9c f9 0c 00 	lea    0xcf99c(%rip),%rax        # dc1d8 <constant_11_mobilenet+0x2c8>
    c83c:	48 89 c7             	mov    %rax,%rdi
    c83f:	b8 01 00 00 00       	mov    $0x1,%eax
    c844:	e8 e7 57 ff ff       	call   2030 <printf@plt>
    c849:	e9 88 00 00 00       	jmp    c8d6 <printElement+0x356>
    c84e:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    c852:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
    c859:	00 
    c85a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c85e:	48 01 d0             	add    %rdx,%rax
    c861:	48 8b 00             	mov    (%rax),%rax
    c864:	66 48 0f 6e c0       	movq   %rax,%xmm0
    c869:	48 8d 05 68 f9 0c 00 	lea    0xcf968(%rip),%rax        # dc1d8 <constant_11_mobilenet+0x2c8>
    c870:	48 89 c7             	mov    %rax,%rdi
    c873:	b8 01 00 00 00       	mov    $0x1,%eax
    c878:	e8 b3 57 ff ff       	call   2030 <printf@plt>
    c87d:	eb 57                	jmp    c8d6 <printElement+0x356>
    c87f:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    c883:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
    c88a:	00 
    c88b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c88f:	48 01 d0             	add    %rdx,%rax
    c892:	48 8b 00             	mov    (%rax),%rax
    c895:	48 89 c6             	mov    %rax,%rsi
    c898:	48 8d 05 3c f9 0c 00 	lea    0xcf93c(%rip),%rax        # dc1db <constant_11_mobilenet+0x2cb>
    c89f:	48 89 c7             	mov    %rax,%rdi
    c8a2:	b8 00 00 00 00       	mov    $0x0,%eax
    c8a7:	e8 84 57 ff ff       	call   2030 <printf@plt>
    c8ac:	eb 28                	jmp    c8d6 <printElement+0x356>
    c8ae:	48 8d 05 7b fa 0c 00 	lea    0xcfa7b(%rip),%rax        # dc330 <__PRETTY_FUNCTION__.0>
    c8b5:	48 89 c1             	mov    %rax,%rcx
    c8b8:	ba c4 00 00 00       	mov    $0xc4,%edx
    c8bd:	48 8d 05 4c f8 0c 00 	lea    0xcf84c(%rip),%rax        # dc110 <constant_11_mobilenet+0x200>
    c8c4:	48 89 c6             	mov    %rax,%rsi
    c8c7:	48 8d 05 12 f9 0c 00 	lea    0xcf912(%rip),%rax        # dc1e0 <constant_11_mobilenet+0x2d0>
    c8ce:	48 89 c7             	mov    %rax,%rdi
    c8d1:	e8 aa 57 ff ff       	call   2080 <__assert_fail@plt>
    c8d6:	90                   	nop
    c8d7:	c9                   	leave  
    c8d8:	c3                   	ret    

000000000000c8d9 <inner_product>:
    c8d9:	55                   	push   %rbp
    c8da:	48 89 e5             	mov    %rsp,%rbp
    c8dd:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    c8e1:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    c8e5:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
    c8e9:	48 89 4d d0          	mov    %rcx,-0x30(%rbp)
    c8ed:	48 8b 45 d0          	mov    -0x30(%rbp),%rax
    c8f1:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
    c8f5:	48 c7 45 f8 00 00 00 	movq   $0x0,-0x8(%rbp)
    c8fc:	00 
    c8fd:	eb 39                	jmp    c938 <inner_product+0x5f>
    c8ff:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c903:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
    c90a:	00 
    c90b:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    c90f:	48 01 d0             	add    %rdx,%rax
    c912:	48 8b 10             	mov    (%rax),%rdx
    c915:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c919:	48 8d 0c c5 00 00 00 	lea    0x0(,%rax,8),%rcx
    c920:	00 
    c921:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    c925:	48 01 c8             	add    %rcx,%rax
    c928:	48 8b 00             	mov    (%rax),%rax
    c92b:	48 0f af c2          	imul   %rdx,%rax
    c92f:	48 01 45 f0          	add    %rax,-0x10(%rbp)
    c933:	48 83 45 f8 01       	addq   $0x1,-0x8(%rbp)
    c938:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c93c:	48 3b 45 d8          	cmp    -0x28(%rbp),%rax
    c940:	7c bd                	jl     c8ff <inner_product+0x26>
    c942:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    c946:	5d                   	pop    %rbp
    c947:	c3                   	ret    

000000000000c948 <computeElemOffset>:
    c948:	55                   	push   %rbp
    c949:	48 89 e5             	mov    %rsp,%rbp
    c94c:	48 83 ec 18          	sub    $0x18,%rsp
    c950:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    c954:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    c958:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    c95c:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    c960:	48 8b 75 f0          	mov    -0x10(%rbp),%rsi
    c964:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c968:	b9 00 00 00 00       	mov    $0x0,%ecx
    c96d:	48 89 c7             	mov    %rax,%rdi
    c970:	e8 64 ff ff ff       	call   c8d9 <inner_product>
    c975:	c9                   	leave  
    c976:	c3                   	ret    

000000000000c977 <omTensorCreate>:
    c977:	f3 0f 1e fa          	endbr64 
    c97b:	55                   	push   %rbp
    c97c:	48 89 e5             	mov    %rsp,%rbp
    c97f:	48 83 ec 30          	sub    $0x30,%rsp
    c983:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    c987:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    c98b:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
    c98f:	89 4d d4             	mov    %ecx,-0x2c(%rbp)
    c992:	bf 40 00 00 00       	mov    $0x40,%edi
    c997:	e8 34 57 ff ff       	call   20d0 <malloc@plt>
    c99c:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    c9a0:	48 83 7d f8 00       	cmpq   $0x0,-0x8(%rbp)
    c9a5:	75 0a                	jne    c9b1 <omTensorCreate+0x3a>
    c9a7:	b8 00 00 00 00       	mov    $0x0,%eax
    c9ac:	e9 88 01 00 00       	jmp    cb39 <omTensorCreate+0x1c2>
    c9b1:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    c9b5:	48 c1 e0 03          	shl    $0x3,%rax
    c9b9:	48 89 c7             	mov    %rax,%rdi
    c9bc:	e8 0f 57 ff ff       	call   20d0 <malloc@plt>
    c9c1:	48 89 c2             	mov    %rax,%rdx
    c9c4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c9c8:	48 89 50 18          	mov    %rdx,0x18(%rax)
    c9cc:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c9d0:	48 8b 40 18          	mov    0x18(%rax),%rax
    c9d4:	48 85 c0             	test   %rax,%rax
    c9d7:	74 7e                	je     ca57 <omTensorCreate+0xe0>
    c9d9:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    c9dd:	48 c1 e0 03          	shl    $0x3,%rax
    c9e1:	48 89 c7             	mov    %rax,%rdi
    c9e4:	e8 e7 56 ff ff       	call   20d0 <malloc@plt>
    c9e9:	48 89 c2             	mov    %rax,%rdx
    c9ec:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c9f0:	48 89 50 20          	mov    %rdx,0x20(%rax)
    c9f4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    c9f8:	48 8b 40 20          	mov    0x20(%rax),%rax
    c9fc:	48 85 c0             	test   %rax,%rax
    c9ff:	74 56                	je     ca57 <omTensorCreate+0xe0>
    ca01:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ca05:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    ca09:	48 89 10             	mov    %rdx,(%rax)
    ca0c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ca10:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    ca14:	48 89 50 08          	mov    %rdx,0x8(%rax)
    ca18:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ca1c:	48 c7 40 10 00 00 00 	movq   $0x0,0x10(%rax)
    ca23:	00 
    ca24:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ca28:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
    ca2c:	48 89 50 28          	mov    %rdx,0x28(%rax)
    ca30:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ca34:	8b 55 d4             	mov    -0x2c(%rbp),%edx
    ca37:	89 50 30             	mov    %edx,0x30(%rax)
    ca3a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ca3e:	48 c7 40 38 00 00 00 	movq   $0x0,0x38(%rax)
    ca45:	00 
    ca46:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    ca4a:	48 83 e8 01          	sub    $0x1,%rax
    ca4e:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
    ca52:	e9 d3 00 00 00       	jmp    cb2a <omTensorCreate+0x1b3>
    ca57:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ca5b:	48 8b 40 18          	mov    0x18(%rax),%rax
    ca5f:	48 89 c7             	mov    %rax,%rdi
    ca62:	e8 a9 56 ff ff       	call   2110 <free@plt>
    ca67:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ca6b:	48 89 c7             	mov    %rax,%rdi
    ca6e:	e8 9d 56 ff ff       	call   2110 <free@plt>
    ca73:	b8 00 00 00 00       	mov    $0x0,%eax
    ca78:	e9 bc 00 00 00       	jmp    cb39 <omTensorCreate+0x1c2>
    ca7d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    ca81:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
    ca88:	00 
    ca89:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    ca8d:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
    ca91:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ca95:	48 8b 50 18          	mov    0x18(%rax),%rdx
    ca99:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    ca9d:	48 c1 e0 03          	shl    $0x3,%rax
    caa1:	48 01 c2             	add    %rax,%rdx
    caa4:	48 8b 01             	mov    (%rcx),%rax
    caa7:	48 89 02             	mov    %rax,(%rdx)
    caaa:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    caae:	48 83 e8 01          	sub    $0x1,%rax
    cab2:	48 39 45 f0          	cmp    %rax,-0x10(%rbp)
    cab6:	75 1c                	jne    cad4 <omTensorCreate+0x15d>
    cab8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cabc:	48 8b 50 20          	mov    0x20(%rax),%rdx
    cac0:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    cac4:	48 c1 e0 03          	shl    $0x3,%rax
    cac8:	48 01 d0             	add    %rdx,%rax
    cacb:	48 c7 00 01 00 00 00 	movq   $0x1,(%rax)
    cad2:	eb 51                	jmp    cb25 <omTensorCreate+0x1ae>
    cad4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cad8:	48 8b 50 20          	mov    0x20(%rax),%rdx
    cadc:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    cae0:	48 83 c0 01          	add    $0x1,%rax
    cae4:	48 c1 e0 03          	shl    $0x3,%rax
    cae8:	48 01 d0             	add    %rdx,%rax
    caeb:	48 8b 30             	mov    (%rax),%rsi
    caee:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    caf2:	48 8b 50 18          	mov    0x18(%rax),%rdx
    caf6:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    cafa:	48 83 c0 01          	add    $0x1,%rax
    cafe:	48 c1 e0 03          	shl    $0x3,%rax
    cb02:	48 01 d0             	add    %rdx,%rax
    cb05:	48 8b 08             	mov    (%rax),%rcx
    cb08:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cb0c:	48 8b 50 20          	mov    0x20(%rax),%rdx
    cb10:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    cb14:	48 c1 e0 03          	shl    $0x3,%rax
    cb18:	48 01 c2             	add    %rax,%rdx
    cb1b:	48 89 f0             	mov    %rsi,%rax
    cb1e:	48 0f af c1          	imul   %rcx,%rax
    cb22:	48 89 02             	mov    %rax,(%rdx)
    cb25:	48 83 6d f0 01       	subq   $0x1,-0x10(%rbp)
    cb2a:	48 83 7d f0 00       	cmpq   $0x0,-0x10(%rbp)
    cb2f:	0f 89 48 ff ff ff    	jns    ca7d <omTensorCreate+0x106>
    cb35:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cb39:	c9                   	leave  
    cb3a:	c3                   	ret    

000000000000cb3b <omTensorCreateWithOwnership>:
    cb3b:	f3 0f 1e fa          	endbr64 
    cb3f:	55                   	push   %rbp
    cb40:	48 89 e5             	mov    %rsp,%rbp
    cb43:	48 83 ec 40          	sub    $0x40,%rsp
    cb47:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    cb4b:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    cb4f:	48 89 55 d8          	mov    %rdx,-0x28(%rbp)
    cb53:	89 4d d4             	mov    %ecx,-0x2c(%rbp)
    cb56:	4c 89 45 c8          	mov    %r8,-0x38(%rbp)
    cb5a:	8b 4d d4             	mov    -0x2c(%rbp),%ecx
    cb5d:	48 8b 55 d8          	mov    -0x28(%rbp),%rdx
    cb61:	48 8b 75 e0          	mov    -0x20(%rbp),%rsi
    cb65:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    cb69:	48 89 c7             	mov    %rax,%rdi
    cb6c:	e8 06 fe ff ff       	call   c977 <omTensorCreate>
    cb71:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    cb75:	48 83 7d f8 00       	cmpq   $0x0,-0x8(%rbp)
    cb7a:	75 07                	jne    cb83 <omTensorCreateWithOwnership+0x48>
    cb7c:	b8 00 00 00 00       	mov    $0x0,%eax
    cb81:	eb 10                	jmp    cb93 <omTensorCreateWithOwnership+0x58>
    cb83:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cb87:	48 8b 55 c8          	mov    -0x38(%rbp),%rdx
    cb8b:	48 89 50 38          	mov    %rdx,0x38(%rax)
    cb8f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cb93:	c9                   	leave  
    cb94:	c3                   	ret    

000000000000cb95 <omTensorCreateUntyped>:
    cb95:	f3 0f 1e fa          	endbr64 
    cb99:	55                   	push   %rbp
    cb9a:	48 89 e5             	mov    %rsp,%rbp
    cb9d:	48 83 ec 20          	sub    $0x20,%rsp
    cba1:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    cba5:	bf 40 00 00 00       	mov    $0x40,%edi
    cbaa:	e8 21 55 ff ff       	call   20d0 <malloc@plt>
    cbaf:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    cbb3:	48 83 7d f8 00       	cmpq   $0x0,-0x8(%rbp)
    cbb8:	75 0a                	jne    cbc4 <omTensorCreateUntyped+0x2f>
    cbba:	b8 00 00 00 00       	mov    $0x0,%eax
    cbbf:	e9 bd 00 00 00       	jmp    cc81 <omTensorCreateUntyped+0xec>
    cbc4:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    cbc8:	48 c1 e0 03          	shl    $0x3,%rax
    cbcc:	48 89 c7             	mov    %rax,%rdi
    cbcf:	e8 fc 54 ff ff       	call   20d0 <malloc@plt>
    cbd4:	48 89 c2             	mov    %rax,%rdx
    cbd7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cbdb:	48 89 50 18          	mov    %rdx,0x18(%rax)
    cbdf:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cbe3:	48 8b 40 18          	mov    0x18(%rax),%rax
    cbe7:	48 85 c0             	test   %rax,%rax
    cbea:	74 74                	je     cc60 <omTensorCreateUntyped+0xcb>
    cbec:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    cbf0:	48 c1 e0 03          	shl    $0x3,%rax
    cbf4:	48 89 c7             	mov    %rax,%rdi
    cbf7:	e8 d4 54 ff ff       	call   20d0 <malloc@plt>
    cbfc:	48 89 c2             	mov    %rax,%rdx
    cbff:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cc03:	48 89 50 20          	mov    %rdx,0x20(%rax)
    cc07:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cc0b:	48 8b 40 20          	mov    0x20(%rax),%rax
    cc0f:	48 85 c0             	test   %rax,%rax
    cc12:	74 4c                	je     cc60 <omTensorCreateUntyped+0xcb>
    cc14:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cc18:	48 c7 00 00 00 00 00 	movq   $0x0,(%rax)
    cc1f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cc23:	48 c7 40 08 00 00 00 	movq   $0x0,0x8(%rax)
    cc2a:	00 
    cc2b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cc2f:	48 c7 40 10 00 00 00 	movq   $0x0,0x10(%rax)
    cc36:	00 
    cc37:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cc3b:	c7 40 30 00 00 00 00 	movl   $0x0,0x30(%rax)
    cc42:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cc46:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    cc4a:	48 89 50 28          	mov    %rdx,0x28(%rax)
    cc4e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cc52:	48 c7 40 38 00 00 00 	movq   $0x0,0x38(%rax)
    cc59:	00 
    cc5a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cc5e:	eb 21                	jmp    cc81 <omTensorCreateUntyped+0xec>
    cc60:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cc64:	48 8b 40 18          	mov    0x18(%rax),%rax
    cc68:	48 89 c7             	mov    %rax,%rdi
    cc6b:	e8 a0 54 ff ff       	call   2110 <free@plt>
    cc70:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cc74:	48 89 c7             	mov    %rax,%rdi
    cc77:	e8 94 54 ff ff       	call   2110 <free@plt>
    cc7c:	b8 00 00 00 00       	mov    $0x0,%eax
    cc81:	c9                   	leave  
    cc82:	c3                   	ret    

000000000000cc83 <omTensorCreateEmpty>:
    cc83:	f3 0f 1e fa          	endbr64 
    cc87:	55                   	push   %rbp
    cc88:	48 89 e5             	mov    %rsp,%rbp
    cc8b:	53                   	push   %rbx
    cc8c:	48 83 ec 38          	sub    $0x38,%rsp
    cc90:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
    cc94:	48 89 75 d0          	mov    %rsi,-0x30(%rbp)
    cc98:	89 55 cc             	mov    %edx,-0x34(%rbp)
    cc9b:	48 8b 55 d0          	mov    -0x30(%rbp),%rdx
    cc9f:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    cca3:	48 89 d6             	mov    %rdx,%rsi
    cca6:	48 89 c7             	mov    %rax,%rdi
    cca9:	e8 66 f8 ff ff       	call   c514 <getNumElems>
    ccae:	48 89 c3             	mov    %rax,%rbx
    ccb1:	8b 45 cc             	mov    -0x34(%rbp),%eax
    ccb4:	89 c7                	mov    %eax,%edi
    ccb6:	e8 9c f7 ff ff       	call   c457 <getDataTypeSize>
    ccbb:	48 0f af c3          	imul   %rbx,%rax
    ccbf:	48 89 c7             	mov    %rax,%rdi
    ccc2:	e8 09 54 ff ff       	call   20d0 <malloc@plt>
    ccc7:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
    cccb:	48 83 7d e0 00       	cmpq   $0x0,-0x20(%rbp)
    ccd0:	74 28                	je     ccfa <omTensorCreateEmpty+0x77>
    ccd2:	8b 4d cc             	mov    -0x34(%rbp),%ecx
    ccd5:	48 8b 55 d0          	mov    -0x30(%rbp),%rdx
    ccd9:	48 8b 75 d8          	mov    -0x28(%rbp),%rsi
    ccdd:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    cce1:	41 b8 01 00 00 00    	mov    $0x1,%r8d
    cce7:	48 89 c7             	mov    %rax,%rdi
    ccea:	e8 4c fe ff ff       	call   cb3b <omTensorCreateWithOwnership>
    ccef:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    ccf3:	48 83 7d e8 00       	cmpq   $0x0,-0x18(%rbp)
    ccf8:	75 13                	jne    cd0d <omTensorCreateEmpty+0x8a>
    ccfa:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    ccfe:	48 89 c7             	mov    %rax,%rdi
    cd01:	e8 0a 54 ff ff       	call   2110 <free@plt>
    cd06:	b8 00 00 00 00       	mov    $0x0,%eax
    cd0b:	eb 04                	jmp    cd11 <omTensorCreateEmpty+0x8e>
    cd0d:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    cd11:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    cd15:	c9                   	leave  
    cd16:	c3                   	ret    

000000000000cd17 <omTensorDestroy>:
    cd17:	f3 0f 1e fa          	endbr64 
    cd1b:	55                   	push   %rbp
    cd1c:	48 89 e5             	mov    %rsp,%rbp
    cd1f:	48 83 ec 10          	sub    $0x10,%rsp
    cd23:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    cd27:	48 83 7d f8 00       	cmpq   $0x0,-0x8(%rbp)
    cd2c:	74 4a                	je     cd78 <omTensorDestroy+0x61>
    cd2e:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cd32:	48 8b 40 38          	mov    0x38(%rax),%rax
    cd36:	48 85 c0             	test   %rax,%rax
    cd39:	74 0f                	je     cd4a <omTensorDestroy+0x33>
    cd3b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cd3f:	48 8b 00             	mov    (%rax),%rax
    cd42:	48 89 c7             	mov    %rax,%rdi
    cd45:	e8 c6 53 ff ff       	call   2110 <free@plt>
    cd4a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cd4e:	48 8b 40 18          	mov    0x18(%rax),%rax
    cd52:	48 89 c7             	mov    %rax,%rdi
    cd55:	e8 b6 53 ff ff       	call   2110 <free@plt>
    cd5a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cd5e:	48 8b 40 20          	mov    0x20(%rax),%rax
    cd62:	48 89 c7             	mov    %rax,%rdi
    cd65:	e8 a6 53 ff ff       	call   2110 <free@plt>
    cd6a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cd6e:	48 89 c7             	mov    %rax,%rdi
    cd71:	e8 9a 53 ff ff       	call   2110 <free@plt>
    cd76:	eb 01                	jmp    cd79 <omTensorDestroy+0x62>
    cd78:	90                   	nop
    cd79:	c9                   	leave  
    cd7a:	c3                   	ret    

000000000000cd7b <omTensorGetDataPtr>:
    cd7b:	f3 0f 1e fa          	endbr64 
    cd7f:	55                   	push   %rbp
    cd80:	48 89 e5             	mov    %rsp,%rbp
    cd83:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    cd87:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cd8b:	48 8b 40 08          	mov    0x8(%rax),%rax
    cd8f:	5d                   	pop    %rbp
    cd90:	c3                   	ret    

000000000000cd91 <omTensorSetDataPtr>:
    cd91:	f3 0f 1e fa          	endbr64 
    cd95:	55                   	push   %rbp
    cd96:	48 89 e5             	mov    %rsp,%rbp
    cd99:	48 83 ec 20          	sub    $0x20,%rsp
    cd9d:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    cda1:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    cda5:	48 89 55 e8          	mov    %rdx,-0x18(%rbp)
    cda9:	48 89 4d e0          	mov    %rcx,-0x20(%rbp)
    cdad:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cdb1:	48 8b 40 38          	mov    0x38(%rax),%rax
    cdb5:	48 85 c0             	test   %rax,%rax
    cdb8:	74 0f                	je     cdc9 <omTensorSetDataPtr+0x38>
    cdba:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cdbe:	48 8b 00             	mov    (%rax),%rax
    cdc1:	48 89 c7             	mov    %rax,%rdi
    cdc4:	e8 47 53 ff ff       	call   2110 <free@plt>
    cdc9:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cdcd:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    cdd1:	48 89 50 38          	mov    %rdx,0x38(%rax)
    cdd5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cdd9:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    cddd:	48 89 10             	mov    %rdx,(%rax)
    cde0:	48 83 7d e0 00       	cmpq   $0x0,-0x20(%rbp)
    cde5:	74 0e                	je     cdf5 <omTensorSetDataPtr+0x64>
    cde7:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cdeb:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
    cdef:	48 89 50 08          	mov    %rdx,0x8(%rax)
    cdf3:	eb 0c                	jmp    ce01 <omTensorSetDataPtr+0x70>
    cdf5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cdf9:	48 8b 55 e8          	mov    -0x18(%rbp),%rdx
    cdfd:	48 89 50 08          	mov    %rdx,0x8(%rax)
    ce01:	90                   	nop
    ce02:	c9                   	leave  
    ce03:	c3                   	ret    

000000000000ce04 <omTensorGetShape>:
    ce04:	f3 0f 1e fa          	endbr64 
    ce08:	55                   	push   %rbp
    ce09:	48 89 e5             	mov    %rsp,%rbp
    ce0c:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    ce10:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ce14:	48 8b 40 18          	mov    0x18(%rax),%rax
    ce18:	5d                   	pop    %rbp
    ce19:	c3                   	ret    

000000000000ce1a <omTensorSetShape>:
    ce1a:	f3 0f 1e fa          	endbr64 
    ce1e:	55                   	push   %rbp
    ce1f:	48 89 e5             	mov    %rsp,%rbp
    ce22:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    ce26:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    ce2a:	48 c7 45 f8 00 00 00 	movq   $0x0,-0x8(%rbp)
    ce31:	00 
    ce32:	eb 32                	jmp    ce66 <omTensorSetShape+0x4c>
    ce34:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ce38:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
    ce3f:	00 
    ce40:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    ce44:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
    ce48:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    ce4c:	48 8b 50 18          	mov    0x18(%rax),%rdx
    ce50:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ce54:	48 c1 e0 03          	shl    $0x3,%rax
    ce58:	48 01 c2             	add    %rax,%rdx
    ce5b:	48 8b 01             	mov    (%rcx),%rax
    ce5e:	48 89 02             	mov    %rax,(%rdx)
    ce61:	48 83 45 f8 01       	addq   $0x1,-0x8(%rbp)
    ce66:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    ce6a:	48 8b 40 28          	mov    0x28(%rax),%rax
    ce6e:	48 39 45 f8          	cmp    %rax,-0x8(%rbp)
    ce72:	7c c0                	jl     ce34 <omTensorSetShape+0x1a>
    ce74:	90                   	nop
    ce75:	90                   	nop
    ce76:	5d                   	pop    %rbp
    ce77:	c3                   	ret    

000000000000ce78 <omTensorGetStrides>:
    ce78:	f3 0f 1e fa          	endbr64 
    ce7c:	55                   	push   %rbp
    ce7d:	48 89 e5             	mov    %rsp,%rbp
    ce80:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    ce84:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ce88:	48 8b 40 20          	mov    0x20(%rax),%rax
    ce8c:	5d                   	pop    %rbp
    ce8d:	c3                   	ret    

000000000000ce8e <omTensorSetStrides>:
    ce8e:	f3 0f 1e fa          	endbr64 
    ce92:	55                   	push   %rbp
    ce93:	48 89 e5             	mov    %rsp,%rbp
    ce96:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    ce9a:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    ce9e:	48 c7 45 f8 00 00 00 	movq   $0x0,-0x8(%rbp)
    cea5:	00 
    cea6:	eb 32                	jmp    ceda <omTensorSetStrides+0x4c>
    cea8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    ceac:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
    ceb3:	00 
    ceb4:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    ceb8:	48 8d 0c 02          	lea    (%rdx,%rax,1),%rcx
    cebc:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    cec0:	48 8b 50 20          	mov    0x20(%rax),%rdx
    cec4:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cec8:	48 c1 e0 03          	shl    $0x3,%rax
    cecc:	48 01 c2             	add    %rax,%rdx
    cecf:	48 8b 01             	mov    (%rcx),%rax
    ced2:	48 89 02             	mov    %rax,(%rdx)
    ced5:	48 83 45 f8 01       	addq   $0x1,-0x8(%rbp)
    ceda:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    cede:	48 8b 40 28          	mov    0x28(%rax),%rax
    cee2:	48 39 45 f8          	cmp    %rax,-0x8(%rbp)
    cee6:	7c c0                	jl     cea8 <omTensorSetStrides+0x1a>
    cee8:	90                   	nop
    cee9:	90                   	nop
    ceea:	5d                   	pop    %rbp
    ceeb:	c3                   	ret    

000000000000ceec <omTensorSetStridesWithPyArrayStrides>:
    ceec:	f3 0f 1e fa          	endbr64 
    cef0:	55                   	push   %rbp
    cef1:	48 89 e5             	mov    %rsp,%rbp
    cef4:	53                   	push   %rbx
    cef5:	48 83 ec 20          	sub    $0x20,%rsp
    cef9:	48 89 7d e0          	mov    %rdi,-0x20(%rbp)
    cefd:	48 89 75 d8          	mov    %rsi,-0x28(%rbp)
    cf01:	48 c7 45 f0 00 00 00 	movq   $0x0,-0x10(%rbp)
    cf08:	00 
    cf09:	eb 4a                	jmp    cf55 <omTensorSetStridesWithPyArrayStrides+0x69>
    cf0b:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    cf0f:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
    cf16:	00 
    cf17:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    cf1b:	48 01 d0             	add    %rdx,%rax
    cf1e:	48 8b 18             	mov    (%rax),%rbx
    cf21:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    cf25:	8b 40 30             	mov    0x30(%rax),%eax
    cf28:	89 c7                	mov    %eax,%edi
    cf2a:	e8 28 f5 ff ff       	call   c457 <getDataTypeSize>
    cf2f:	48 89 c6             	mov    %rax,%rsi
    cf32:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    cf36:	48 8b 48 20          	mov    0x20(%rax),%rcx
    cf3a:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    cf3e:	48 c1 e0 03          	shl    $0x3,%rax
    cf42:	48 01 c1             	add    %rax,%rcx
    cf45:	48 89 d8             	mov    %rbx,%rax
    cf48:	48 99                	cqto   
    cf4a:	48 f7 fe             	idiv   %rsi
    cf4d:	48 89 01             	mov    %rax,(%rcx)
    cf50:	48 83 45 f0 01       	addq   $0x1,-0x10(%rbp)
    cf55:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    cf59:	48 8b 40 28          	mov    0x28(%rax),%rax
    cf5d:	48 39 45 f0          	cmp    %rax,-0x10(%rbp)
    cf61:	7c a8                	jl     cf0b <omTensorSetStridesWithPyArrayStrides+0x1f>
    cf63:	90                   	nop
    cf64:	90                   	nop
    cf65:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    cf69:	c9                   	leave  
    cf6a:	c3                   	ret    

000000000000cf6b <omTensorGetDataType>:
    cf6b:	f3 0f 1e fa          	endbr64 
    cf6f:	55                   	push   %rbp
    cf70:	48 89 e5             	mov    %rsp,%rbp
    cf73:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    cf77:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cf7b:	8b 40 30             	mov    0x30(%rax),%eax
    cf7e:	5d                   	pop    %rbp
    cf7f:	c3                   	ret    

000000000000cf80 <omTensorSetDataType>:
    cf80:	f3 0f 1e fa          	endbr64 
    cf84:	55                   	push   %rbp
    cf85:	48 89 e5             	mov    %rsp,%rbp
    cf88:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    cf8c:	89 75 f4             	mov    %esi,-0xc(%rbp)
    cf8f:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cf93:	8b 55 f4             	mov    -0xc(%rbp),%edx
    cf96:	89 50 30             	mov    %edx,0x30(%rax)
    cf99:	90                   	nop
    cf9a:	5d                   	pop    %rbp
    cf9b:	c3                   	ret    

000000000000cf9c <omTensorGetBufferSize>:
    cf9c:	f3 0f 1e fa          	endbr64 
    cfa0:	55                   	push   %rbp
    cfa1:	48 89 e5             	mov    %rsp,%rbp
    cfa4:	53                   	push   %rbx
    cfa5:	48 83 ec 08          	sub    $0x8,%rsp
    cfa9:	48 89 7d f0          	mov    %rdi,-0x10(%rbp)
    cfad:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    cfb1:	48 8b 50 28          	mov    0x28(%rax),%rdx
    cfb5:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    cfb9:	48 8b 40 18          	mov    0x18(%rax),%rax
    cfbd:	48 89 d6             	mov    %rdx,%rsi
    cfc0:	48 89 c7             	mov    %rax,%rdi
    cfc3:	e8 4c f5 ff ff       	call   c514 <getNumElems>
    cfc8:	48 89 c3             	mov    %rax,%rbx
    cfcb:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    cfcf:	8b 40 30             	mov    0x30(%rax),%eax
    cfd2:	89 c7                	mov    %eax,%edi
    cfd4:	e8 7e f4 ff ff       	call   c457 <getDataTypeSize>
    cfd9:	48 0f af c3          	imul   %rbx,%rax
    cfdd:	48 8b 5d f8          	mov    -0x8(%rbp),%rbx
    cfe1:	c9                   	leave  
    cfe2:	c3                   	ret    

000000000000cfe3 <omTensorGetRank>:
    cfe3:	f3 0f 1e fa          	endbr64 
    cfe7:	55                   	push   %rbp
    cfe8:	48 89 e5             	mov    %rsp,%rbp
    cfeb:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    cfef:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    cff3:	48 8b 40 28          	mov    0x28(%rax),%rax
    cff7:	5d                   	pop    %rbp
    cff8:	c3                   	ret    

000000000000cff9 <omTensorGetNumElems>:
    cff9:	f3 0f 1e fa          	endbr64 
    cffd:	55                   	push   %rbp
    cffe:	48 89 e5             	mov    %rsp,%rbp
    d001:	48 83 ec 28          	sub    $0x28,%rsp
    d005:	48 89 7d d8          	mov    %rdi,-0x28(%rbp)
    d009:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    d00d:	48 8b 40 28          	mov    0x28(%rax),%rax
    d011:	48 83 e8 01          	sub    $0x1,%rax
    d015:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    d019:	eb 50                	jmp    d06b <omTensorGetNumElems+0x72>
    d01b:	48 c7 45 f0 01 00 00 	movq   $0x1,-0x10(%rbp)
    d022:	00 
    d023:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    d027:	48 83 c0 01          	add    $0x1,%rax
    d02b:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    d02f:	eb 27                	jmp    d058 <omTensorGetNumElems+0x5f>
    d031:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    d035:	48 8b 50 18          	mov    0x18(%rax),%rdx
    d039:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    d03d:	48 c1 e0 03          	shl    $0x3,%rax
    d041:	48 01 d0             	add    %rdx,%rax
    d044:	48 8b 00             	mov    (%rax),%rax
    d047:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    d04b:	48 0f af c2          	imul   %rdx,%rax
    d04f:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
    d053:	48 83 45 f8 01       	addq   $0x1,-0x8(%rbp)
    d058:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    d05c:	48 8b 40 28          	mov    0x28(%rax),%rax
    d060:	48 39 45 f8          	cmp    %rax,-0x8(%rbp)
    d064:	7c cb                	jl     d031 <omTensorGetNumElems+0x38>
    d066:	48 83 6d e8 01       	subq   $0x1,-0x18(%rbp)
    d06b:	48 83 7d e8 00       	cmpq   $0x0,-0x18(%rbp)
    d070:	79 a9                	jns    d01b <omTensorGetNumElems+0x22>
    d072:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    d076:	48 8b 50 28          	mov    0x28(%rax),%rdx
    d07a:	48 8b 45 d8          	mov    -0x28(%rbp),%rax
    d07e:	48 8b 40 18          	mov    0x18(%rax),%rax
    d082:	48 89 d6             	mov    %rdx,%rsi
    d085:	48 89 c7             	mov    %rax,%rdi
    d088:	e8 87 f4 ff ff       	call   c514 <getNumElems>
    d08d:	c9                   	leave  
    d08e:	c3                   	ret    

000000000000d08f <omTensorGetOwning>:
    d08f:	f3 0f 1e fa          	endbr64 
    d093:	55                   	push   %rbp
    d094:	48 89 e5             	mov    %rsp,%rbp
    d097:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    d09b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    d09f:	48 8b 40 38          	mov    0x38(%rax),%rax
    d0a3:	5d                   	pop    %rbp
    d0a4:	c3                   	ret    

000000000000d0a5 <omTensorSetOwning>:
    d0a5:	f3 0f 1e fa          	endbr64 
    d0a9:	55                   	push   %rbp
    d0aa:	48 89 e5             	mov    %rsp,%rbp
    d0ad:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    d0b1:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    d0b5:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    d0b9:	48 8b 55 f0          	mov    -0x10(%rbp),%rdx
    d0bd:	48 89 50 38          	mov    %rdx,0x38(%rax)
    d0c1:	90                   	nop
    d0c2:	5d                   	pop    %rbp
    d0c3:	c3                   	ret    

000000000000d0c4 <omTensorGetAllocatedPtr>:
    d0c4:	f3 0f 1e fa          	endbr64 
    d0c8:	55                   	push   %rbp
    d0c9:	48 89 e5             	mov    %rsp,%rbp
    d0cc:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    d0d0:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    d0d4:	48 8b 00             	mov    (%rax),%rax
    d0d7:	5d                   	pop    %rbp
    d0d8:	c3                   	ret    

000000000000d0d9 <omTensorPrint>:
    d0d9:	f3 0f 1e fa          	endbr64 
    d0dd:	55                   	push   %rbp
    d0de:	48 89 e5             	mov    %rsp,%rbp
    d0e1:	48 81 ec f0 00 00 00 	sub    $0xf0,%rsp
    d0e8:	48 89 bd 18 ff ff ff 	mov    %rdi,-0xe8(%rbp)
    d0ef:	48 89 b5 10 ff ff ff 	mov    %rsi,-0xf0(%rbp)
    d0f6:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    d0fd:	00 00 
    d0ff:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    d103:	31 c0                	xor    %eax,%eax
    d105:	48 83 bd 10 ff ff ff 	cmpq   $0x0,-0xf0(%rbp)
    d10c:	00 
    d10d:	75 28                	jne    d137 <omTensorPrint+0x5e>
    d10f:	48 8d 05 0a f2 0c 00 	lea    0xcf20a(%rip),%rax        # dc320 <__PRETTY_FUNCTION__.1>
    d116:	48 89 c1             	mov    %rax,%rcx
    d119:	ba b4 01 00 00       	mov    $0x1b4,%edx
    d11e:	48 8d 05 eb ef 0c 00 	lea    0xcefeb(%rip),%rax        # dc110 <constant_11_mobilenet+0x200>
    d125:	48 89 c6             	mov    %rax,%rsi
    d128:	48 8d 05 09 f1 0c 00 	lea    0xcf109(%rip),%rax        # dc238 <constant_11_mobilenet+0x328>
    d12f:	48 89 c7             	mov    %rax,%rdi
    d132:	e8 49 4f ff ff       	call   2080 <__assert_fail@plt>
    d137:	48 8b 85 10 ff ff ff 	mov    -0xf0(%rbp),%rax
    d13e:	48 89 c7             	mov    %rax,%rdi
    d141:	e8 25 fe ff ff       	call   cf6b <omTensorGetDataType>
    d146:	89 85 24 ff ff ff    	mov    %eax,-0xdc(%rbp)
    d14c:	48 8b 85 10 ff ff ff 	mov    -0xf0(%rbp),%rax
    d153:	48 89 c7             	mov    %rax,%rdi
    d156:	e8 88 fe ff ff       	call   cfe3 <omTensorGetRank>
    d15b:	48 89 45 90          	mov    %rax,-0x70(%rbp)
    d15f:	48 8b 85 10 ff ff ff 	mov    -0xf0(%rbp),%rax
    d166:	48 89 c7             	mov    %rax,%rdi
    d169:	e8 96 fc ff ff       	call   ce04 <omTensorGetShape>
    d16e:	48 89 45 98          	mov    %rax,-0x68(%rbp)
    d172:	48 8b 85 10 ff ff ff 	mov    -0xf0(%rbp),%rax
    d179:	48 89 c7             	mov    %rax,%rdi
    d17c:	e8 f7 fc ff ff       	call   ce78 <omTensorGetStrides>
    d181:	48 89 45 a0          	mov    %rax,-0x60(%rbp)
    d185:	48 8b 85 10 ff ff ff 	mov    -0xf0(%rbp),%rax
    d18c:	48 89 c7             	mov    %rax,%rdi
    d18f:	e8 e7 fb ff ff       	call   cd7b <omTensorGetDataPtr>
    d194:	48 89 45 a8          	mov    %rax,-0x58(%rbp)
    d198:	c6 85 20 ff ff ff 00 	movb   $0x0,-0xe0(%rbp)
    d19f:	c6 85 21 ff ff ff 00 	movb   $0x0,-0xdf(%rbp)
    d1a6:	c6 85 22 ff ff ff 00 	movb   $0x0,-0xde(%rbp)
    d1ad:	c6 85 23 ff ff ff 00 	movb   $0x0,-0xdd(%rbp)
    d1b4:	e9 a7 00 00 00       	jmp    d260 <omTensorPrint+0x187>
    d1b9:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    d1c0:	48 83 c0 01          	add    $0x1,%rax
    d1c4:	0f b6 00             	movzbl (%rax),%eax
    d1c7:	3c 73                	cmp    $0x73,%al
    d1c9:	75 16                	jne    d1e1 <omTensorPrint+0x108>
    d1cb:	c6 85 20 ff ff ff 01 	movb   $0x1,-0xe0(%rbp)
    d1d2:	0f b6 85 20 ff ff ff 	movzbl -0xe0(%rbp),%eax
    d1d9:	88 85 23 ff ff ff    	mov    %al,-0xdd(%rbp)
    d1df:	eb 77                	jmp    d258 <omTensorPrint+0x17f>
    d1e1:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    d1e8:	48 83 c0 01          	add    $0x1,%rax
    d1ec:	0f b6 00             	movzbl (%rax),%eax
    d1ef:	3c 74                	cmp    $0x74,%al
    d1f1:	75 16                	jne    d209 <omTensorPrint+0x130>
    d1f3:	c6 85 21 ff ff ff 01 	movb   $0x1,-0xdf(%rbp)
    d1fa:	0f b6 85 21 ff ff ff 	movzbl -0xdf(%rbp),%eax
    d201:	88 85 23 ff ff ff    	mov    %al,-0xdd(%rbp)
    d207:	eb 4f                	jmp    d258 <omTensorPrint+0x17f>
    d209:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    d210:	48 83 c0 01          	add    $0x1,%rax
    d214:	0f b6 00             	movzbl (%rax),%eax
    d217:	3c 64                	cmp    $0x64,%al
    d219:	75 16                	jne    d231 <omTensorPrint+0x158>
    d21b:	c6 85 22 ff ff ff 01 	movb   $0x1,-0xde(%rbp)
    d222:	0f b6 85 22 ff ff ff 	movzbl -0xde(%rbp),%eax
    d229:	88 85 23 ff ff ff    	mov    %al,-0xdd(%rbp)
    d22f:	eb 27                	jmp    d258 <omTensorPrint+0x17f>
    d231:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    d238:	48 83 c0 01          	add    $0x1,%rax
    d23c:	0f b6 00             	movzbl (%rax),%eax
    d23f:	0f be c0             	movsbl %al,%eax
    d242:	89 c6                	mov    %eax,%esi
    d244:	48 8d 05 1a f0 0c 00 	lea    0xcf01a(%rip),%rax        # dc265 <constant_11_mobilenet+0x355>
    d24b:	48 89 c7             	mov    %rax,%rdi
    d24e:	b8 00 00 00 00       	mov    $0x0,%eax
    d253:	e8 d8 4d ff ff       	call   2030 <printf@plt>
    d258:	48 83 85 18 ff ff ff 	addq   $0x2,-0xe8(%rbp)
    d25f:	02 
    d260:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    d267:	48 89 c7             	mov    %rax,%rdi
    d26a:	e8 b1 4e ff ff       	call   2120 <strlen@plt>
    d26f:	48 83 f8 01          	cmp    $0x1,%rax
    d273:	76 12                	jbe    d287 <omTensorPrint+0x1ae>
    d275:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    d27c:	0f b6 00             	movzbl (%rax),%eax
    d27f:	3c 25                	cmp    $0x25,%al
    d281:	0f 84 32 ff ff ff    	je     d1b9 <omTensorPrint+0xe0>
    d287:	0f b6 85 23 ff ff ff 	movzbl -0xdd(%rbp),%eax
    d28e:	83 f0 01             	xor    $0x1,%eax
    d291:	84 c0                	test   %al,%al
    d293:	74 14                	je     d2a9 <omTensorPrint+0x1d0>
    d295:	c6 85 22 ff ff ff 01 	movb   $0x1,-0xde(%rbp)
    d29c:	0f b6 85 22 ff ff ff 	movzbl -0xde(%rbp),%eax
    d2a3:	88 85 20 ff ff ff    	mov    %al,-0xe0(%rbp)
    d2a9:	48 83 bd 18 ff ff ff 	cmpq   $0x0,-0xe8(%rbp)
    d2b0:	00 
    d2b1:	74 1e                	je     d2d1 <omTensorPrint+0x1f8>
    d2b3:	48 8b 85 18 ff ff ff 	mov    -0xe8(%rbp),%rax
    d2ba:	48 89 c6             	mov    %rax,%rsi
    d2bd:	48 8d 05 17 ef 0c 00 	lea    0xcef17(%rip),%rax        # dc1db <constant_11_mobilenet+0x2cb>
    d2c4:	48 89 c7             	mov    %rax,%rdi
    d2c7:	b8 00 00 00 00       	mov    $0x0,%eax
    d2cc:	e8 5f 4d ff ff       	call   2030 <printf@plt>
    d2d1:	80 bd 20 ff ff ff 00 	cmpb   $0x0,-0xe0(%rbp)
    d2d8:	0f 84 45 01 00 00    	je     d423 <omTensorPrint+0x34a>
    d2de:	48 8b 45 90          	mov    -0x70(%rbp),%rax
    d2e2:	48 89 c6             	mov    %rax,%rsi
    d2e5:	48 8d 05 97 ef 0c 00 	lea    0xcef97(%rip),%rax        # dc283 <constant_11_mobilenet+0x373>
    d2ec:	48 89 c7             	mov    %rax,%rdi
    d2ef:	b8 00 00 00 00       	mov    $0x0,%eax
    d2f4:	e8 37 4d ff ff       	call   2030 <printf@plt>
    d2f9:	8b 85 24 ff ff ff    	mov    -0xdc(%rbp),%eax
    d2ff:	89 c7                	mov    %eax,%edi
    d301:	e8 63 f2 ff ff       	call   c569 <getDataTypeName>
    d306:	48 89 c6             	mov    %rax,%rsi
    d309:	48 8d 05 81 ef 0c 00 	lea    0xcef81(%rip),%rax        # dc291 <constant_11_mobilenet+0x381>
    d310:	48 89 c7             	mov    %rax,%rdi
    d313:	b8 00 00 00 00       	mov    $0x0,%eax
    d318:	e8 13 4d ff ff       	call   2030 <printf@plt>
    d31d:	48 8b 85 10 ff ff ff 	mov    -0xf0(%rbp),%rax
    d324:	48 89 c7             	mov    %rax,%rdi
    d327:	e8 cd fc ff ff       	call   cff9 <omTensorGetNumElems>
    d32c:	48 89 c6             	mov    %rax,%rsi
    d32f:	48 8d 05 6b ef 0c 00 	lea    0xcef6b(%rip),%rax        # dc2a1 <constant_11_mobilenet+0x391>
    d336:	48 89 c7             	mov    %rax,%rdi
    d339:	b8 00 00 00 00       	mov    $0x0,%eax
    d33e:	e8 ed 4c ff ff       	call   2030 <printf@plt>
    d343:	48 8d 05 69 ef 0c 00 	lea    0xcef69(%rip),%rax        # dc2b3 <constant_11_mobilenet+0x3a3>
    d34a:	48 89 c7             	mov    %rax,%rdi
    d34d:	b8 00 00 00 00       	mov    $0x0,%eax
    d352:	e8 d9 4c ff ff       	call   2030 <printf@plt>
    d357:	48 c7 85 28 ff ff ff 	movq   $0x0,-0xd8(%rbp)
    d35e:	00 00 00 00 
    d362:	eb 38                	jmp    d39c <omTensorPrint+0x2c3>
    d364:	48 8b 85 28 ff ff ff 	mov    -0xd8(%rbp),%rax
    d36b:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
    d372:	00 
    d373:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    d377:	48 01 d0             	add    %rdx,%rax
    d37a:	48 8b 00             	mov    (%rax),%rax
    d37d:	48 89 c6             	mov    %rax,%rsi
    d380:	48 8d 05 35 ef 0c 00 	lea    0xcef35(%rip),%rax        # dc2bc <constant_11_mobilenet+0x3ac>
    d387:	48 89 c7             	mov    %rax,%rdi
    d38a:	b8 00 00 00 00       	mov    $0x0,%eax
    d38f:	e8 9c 4c ff ff       	call   2030 <printf@plt>
    d394:	48 83 85 28 ff ff ff 	addq   $0x1,-0xd8(%rbp)
    d39b:	01 
    d39c:	48 8b 85 28 ff ff ff 	mov    -0xd8(%rbp),%rax
    d3a3:	48 3b 45 90          	cmp    -0x70(%rbp),%rax
    d3a7:	7c bb                	jl     d364 <omTensorPrint+0x28b>
    d3a9:	bf 0a 00 00 00       	mov    $0xa,%edi
    d3ae:	e8 fd 4c ff ff       	call   20b0 <putchar@plt>
    d3b3:	48 8d 05 09 ef 0c 00 	lea    0xcef09(%rip),%rax        # dc2c3 <constant_11_mobilenet+0x3b3>
    d3ba:	48 89 c7             	mov    %rax,%rdi
    d3bd:	b8 00 00 00 00       	mov    $0x0,%eax
    d3c2:	e8 69 4c ff ff       	call   2030 <printf@plt>
    d3c7:	48 c7 85 30 ff ff ff 	movq   $0x0,-0xd0(%rbp)
    d3ce:	00 00 00 00 
    d3d2:	eb 38                	jmp    d40c <omTensorPrint+0x333>
    d3d4:	48 8b 85 30 ff ff ff 	mov    -0xd0(%rbp),%rax
    d3db:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
    d3e2:	00 
    d3e3:	48 8b 45 a0          	mov    -0x60(%rbp),%rax
    d3e7:	48 01 d0             	add    %rdx,%rax
    d3ea:	48 8b 00             	mov    (%rax),%rax
    d3ed:	48 89 c6             	mov    %rax,%rsi
    d3f0:	48 8d 05 c5 ee 0c 00 	lea    0xceec5(%rip),%rax        # dc2bc <constant_11_mobilenet+0x3ac>
    d3f7:	48 89 c7             	mov    %rax,%rdi
    d3fa:	b8 00 00 00 00       	mov    $0x0,%eax
    d3ff:	e8 2c 4c ff ff       	call   2030 <printf@plt>
    d404:	48 83 85 30 ff ff ff 	addq   $0x1,-0xd0(%rbp)
    d40b:	01 
    d40c:	48 8b 85 30 ff ff ff 	mov    -0xd0(%rbp),%rax
    d413:	48 3b 45 90          	cmp    -0x70(%rbp),%rax
    d417:	7c bb                	jl     d3d4 <omTensorPrint+0x2fb>
    d419:	bf 0a 00 00 00       	mov    $0xa,%edi
    d41e:	e8 8d 4c ff ff       	call   20b0 <putchar@plt>
    d423:	80 bd 21 ff ff ff 00 	cmpb   $0x0,-0xdf(%rbp)
    d42a:	0f 84 a6 00 00 00    	je     d4d6 <omTensorPrint+0x3fd>
    d430:	80 bd 20 ff ff ff 00 	cmpb   $0x0,-0xe0(%rbp)
    d437:	74 14                	je     d44d <omTensorPrint+0x374>
    d439:	48 8d 05 8e ee 0c 00 	lea    0xcee8e(%rip),%rax        # dc2ce <constant_11_mobilenet+0x3be>
    d440:	48 89 c7             	mov    %rax,%rdi
    d443:	b8 00 00 00 00       	mov    $0x0,%eax
    d448:	e8 e3 4b ff ff       	call   2030 <printf@plt>
    d44d:	48 c7 85 38 ff ff ff 	movq   $0x0,-0xc8(%rbp)
    d454:	00 00 00 00 
    d458:	eb 38                	jmp    d492 <omTensorPrint+0x3b9>
    d45a:	48 8b 85 38 ff ff ff 	mov    -0xc8(%rbp),%rax
    d461:	48 8d 14 c5 00 00 00 	lea    0x0(,%rax,8),%rdx
    d468:	00 
    d469:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    d46d:	48 01 d0             	add    %rdx,%rax
    d470:	48 8b 00             	mov    (%rax),%rax
    d473:	48 89 c6             	mov    %rax,%rsi
    d476:	48 8d 05 59 ee 0c 00 	lea    0xcee59(%rip),%rax        # dc2d6 <constant_11_mobilenet+0x3c6>
    d47d:	48 89 c7             	mov    %rax,%rdi
    d480:	b8 00 00 00 00       	mov    $0x0,%eax
    d485:	e8 a6 4b ff ff       	call   2030 <printf@plt>
    d48a:	48 83 85 38 ff ff ff 	addq   $0x1,-0xc8(%rbp)
    d491:	01 
    d492:	48 8b 85 38 ff ff ff 	mov    -0xc8(%rbp),%rax
    d499:	48 3b 45 90          	cmp    -0x70(%rbp),%rax
    d49d:	7c bb                	jl     d45a <omTensorPrint+0x381>
    d49f:	8b 85 24 ff ff ff    	mov    -0xdc(%rbp),%eax
    d4a5:	89 c7                	mov    %eax,%edi
    d4a7:	e8 bd f0 ff ff       	call   c569 <getDataTypeName>
    d4ac:	48 89 c6             	mov    %rax,%rsi
    d4af:	48 8d 05 25 ed 0c 00 	lea    0xced25(%rip),%rax        # dc1db <constant_11_mobilenet+0x2cb>
    d4b6:	48 89 c7             	mov    %rax,%rdi
    d4b9:	b8 00 00 00 00       	mov    $0x0,%eax
    d4be:	e8 6d 4b ff ff       	call   2030 <printf@plt>
    d4c3:	80 bd 20 ff ff ff 00 	cmpb   $0x0,-0xe0(%rbp)
    d4ca:	74 0a                	je     d4d6 <omTensorPrint+0x3fd>
    d4cc:	bf 0a 00 00 00       	mov    $0xa,%edi
    d4d1:	e8 da 4b ff ff       	call   20b0 <putchar@plt>
    d4d6:	80 bd 22 ff ff ff 00 	cmpb   $0x0,-0xde(%rbp)
    d4dd:	0f 84 8c 05 00 00    	je     da6f <omTensorPrint+0x996>
    d4e3:	48 8d 05 f2 ed 0c 00 	lea    0xcedf2(%rip),%rax        # dc2dc <constant_11_mobilenet+0x3cc>
    d4ea:	48 89 c7             	mov    %rax,%rdi
    d4ed:	b8 00 00 00 00       	mov    $0x0,%eax
    d4f2:	e8 39 4b ff ff       	call   2030 <printf@plt>
    d4f7:	48 83 7d 90 04       	cmpq   $0x4,-0x70(%rbp)
    d4fc:	0f 87 36 05 00 00    	ja     da38 <omTensorPrint+0x95f>
    d502:	48 8b 45 90          	mov    -0x70(%rbp),%rax
    d506:	48 8d 14 85 00 00 00 	lea    0x0(,%rax,4),%rdx
    d50d:	00 
    d50e:	48 8d 05 f3 ed 0c 00 	lea    0xcedf3(%rip),%rax        # dc308 <constant_11_mobilenet+0x3f8>
    d515:	8b 04 02             	mov    (%rdx,%rax,1),%eax
    d518:	48 98                	cltq   
    d51a:	48 8d 15 e7 ed 0c 00 	lea    0xcede7(%rip),%rdx        # dc308 <constant_11_mobilenet+0x3f8>
    d521:	48 01 d0             	add    %rdx,%rax
    d524:	3e ff e0             	notrack jmp *%rax
    d527:	8b 95 24 ff ff ff    	mov    -0xdc(%rbp),%edx
    d52d:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    d531:	be 00 00 00 00       	mov    $0x0,%esi
    d536:	48 89 c7             	mov    %rax,%rdi
    d539:	e8 42 f0 ff ff       	call   c580 <printElement>
    d53e:	e9 1d 05 00 00       	jmp    da60 <omTensorPrint+0x987>
    d543:	bf 5b 00 00 00       	mov    $0x5b,%edi
    d548:	e8 63 4b ff ff       	call   20b0 <putchar@plt>
    d54d:	48 c7 85 40 ff ff ff 	movq   $0x0,-0xc0(%rbp)
    d554:	00 00 00 00 
    d558:	eb 6c                	jmp    d5c6 <omTensorPrint+0x4ed>
    d55a:	48 83 bd 40 ff ff ff 	cmpq   $0x0,-0xc0(%rbp)
    d561:	00 
    d562:	74 14                	je     d578 <omTensorPrint+0x49f>
    d564:	48 8d 05 7a ed 0c 00 	lea    0xced7a(%rip),%rax        # dc2e5 <constant_11_mobilenet+0x3d5>
    d56b:	48 89 c7             	mov    %rax,%rdi
    d56e:	b8 00 00 00 00       	mov    $0x0,%eax
    d573:	e8 b8 4a ff ff       	call   2030 <printf@plt>
    d578:	48 8b 85 40 ff ff ff 	mov    -0xc0(%rbp),%rax
    d57f:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
    d583:	48 8b 85 10 ff ff ff 	mov    -0xf0(%rbp),%rax
    d58a:	48 8b 40 20          	mov    0x20(%rax),%rax
    d58e:	48 8b 55 90          	mov    -0x70(%rbp),%rdx
    d592:	48 8d 4d d0          	lea    -0x30(%rbp),%rcx
    d596:	48 89 ce             	mov    %rcx,%rsi
    d599:	48 89 c7             	mov    %rax,%rdi
    d59c:	e8 a7 f3 ff ff       	call   c948 <computeElemOffset>
    d5a1:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
    d5a5:	8b 95 24 ff ff ff    	mov    -0xdc(%rbp),%edx
    d5ab:	48 8b 4d c8          	mov    -0x38(%rbp),%rcx
    d5af:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    d5b3:	48 89 ce             	mov    %rcx,%rsi
    d5b6:	48 89 c7             	mov    %rax,%rdi
    d5b9:	e8 c2 ef ff ff       	call   c580 <printElement>
    d5be:	48 83 85 40 ff ff ff 	addq   $0x1,-0xc0(%rbp)
    d5c5:	01 
    d5c6:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    d5ca:	48 8b 00             	mov    (%rax),%rax
    d5cd:	48 39 85 40 ff ff ff 	cmp    %rax,-0xc0(%rbp)
    d5d4:	7c 84                	jl     d55a <omTensorPrint+0x481>
    d5d6:	bf 5d 00 00 00       	mov    $0x5d,%edi
    d5db:	e8 d0 4a ff ff       	call   20b0 <putchar@plt>
    d5e0:	e9 7b 04 00 00       	jmp    da60 <omTensorPrint+0x987>
    d5e5:	bf 5b 00 00 00       	mov    $0x5b,%edi
    d5ea:	e8 c1 4a ff ff       	call   20b0 <putchar@plt>
    d5ef:	48 c7 85 48 ff ff ff 	movq   $0x0,-0xb8(%rbp)
    d5f6:	00 00 00 00 
    d5fa:	e9 d2 00 00 00       	jmp    d6d1 <omTensorPrint+0x5f8>
    d5ff:	48 83 bd 48 ff ff ff 	cmpq   $0x0,-0xb8(%rbp)
    d606:	00 
    d607:	74 14                	je     d61d <omTensorPrint+0x544>
    d609:	48 8d 05 d5 ec 0c 00 	lea    0xcecd5(%rip),%rax        # dc2e5 <constant_11_mobilenet+0x3d5>
    d610:	48 89 c7             	mov    %rax,%rdi
    d613:	b8 00 00 00 00       	mov    $0x0,%eax
    d618:	e8 13 4a ff ff       	call   2030 <printf@plt>
    d61d:	48 8b 85 48 ff ff ff 	mov    -0xb8(%rbp),%rax
    d624:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
    d628:	bf 5b 00 00 00       	mov    $0x5b,%edi
    d62d:	e8 7e 4a ff ff       	call   20b0 <putchar@plt>
    d632:	48 c7 85 50 ff ff ff 	movq   $0x0,-0xb0(%rbp)
    d639:	00 00 00 00 
    d63d:	eb 6c                	jmp    d6ab <omTensorPrint+0x5d2>
    d63f:	48 83 bd 50 ff ff ff 	cmpq   $0x0,-0xb0(%rbp)
    d646:	00 
    d647:	74 14                	je     d65d <omTensorPrint+0x584>
    d649:	48 8d 05 95 ec 0c 00 	lea    0xcec95(%rip),%rax        # dc2e5 <constant_11_mobilenet+0x3d5>
    d650:	48 89 c7             	mov    %rax,%rdi
    d653:	b8 00 00 00 00       	mov    $0x0,%eax
    d658:	e8 d3 49 ff ff       	call   2030 <printf@plt>
    d65d:	48 8b 85 50 ff ff ff 	mov    -0xb0(%rbp),%rax
    d664:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
    d668:	48 8b 85 10 ff ff ff 	mov    -0xf0(%rbp),%rax
    d66f:	48 8b 40 20          	mov    0x20(%rax),%rax
    d673:	48 8b 55 90          	mov    -0x70(%rbp),%rdx
    d677:	48 8d 4d d0          	lea    -0x30(%rbp),%rcx
    d67b:	48 89 ce             	mov    %rcx,%rsi
    d67e:	48 89 c7             	mov    %rax,%rdi
    d681:	e8 c2 f2 ff ff       	call   c948 <computeElemOffset>
    d686:	48 89 45 c0          	mov    %rax,-0x40(%rbp)
    d68a:	8b 95 24 ff ff ff    	mov    -0xdc(%rbp),%edx
    d690:	48 8b 4d c0          	mov    -0x40(%rbp),%rcx
    d694:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    d698:	48 89 ce             	mov    %rcx,%rsi
    d69b:	48 89 c7             	mov    %rax,%rdi
    d69e:	e8 dd ee ff ff       	call   c580 <printElement>
    d6a3:	48 83 85 50 ff ff ff 	addq   $0x1,-0xb0(%rbp)
    d6aa:	01 
    d6ab:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    d6af:	48 83 c0 08          	add    $0x8,%rax
    d6b3:	48 8b 00             	mov    (%rax),%rax
    d6b6:	48 39 85 50 ff ff ff 	cmp    %rax,-0xb0(%rbp)
    d6bd:	7c 80                	jl     d63f <omTensorPrint+0x566>
    d6bf:	bf 5d 00 00 00       	mov    $0x5d,%edi
    d6c4:	e8 e7 49 ff ff       	call   20b0 <putchar@plt>
    d6c9:	48 83 85 48 ff ff ff 	addq   $0x1,-0xb8(%rbp)
    d6d0:	01 
    d6d1:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    d6d5:	48 8b 00             	mov    (%rax),%rax
    d6d8:	48 39 85 48 ff ff ff 	cmp    %rax,-0xb8(%rbp)
    d6df:	0f 8c 1a ff ff ff    	jl     d5ff <omTensorPrint+0x526>
    d6e5:	bf 5d 00 00 00       	mov    $0x5d,%edi
    d6ea:	e8 c1 49 ff ff       	call   20b0 <putchar@plt>
    d6ef:	e9 6c 03 00 00       	jmp    da60 <omTensorPrint+0x987>
    d6f4:	bf 5b 00 00 00       	mov    $0x5b,%edi
    d6f9:	e8 b2 49 ff ff       	call   20b0 <putchar@plt>
    d6fe:	48 c7 85 58 ff ff ff 	movq   $0x0,-0xa8(%rbp)
    d705:	00 00 00 00 
    d709:	e9 3f 01 00 00       	jmp    d84d <omTensorPrint+0x774>
    d70e:	48 83 bd 58 ff ff ff 	cmpq   $0x0,-0xa8(%rbp)
    d715:	00 
    d716:	74 14                	je     d72c <omTensorPrint+0x653>
    d718:	48 8d 05 c6 eb 0c 00 	lea    0xcebc6(%rip),%rax        # dc2e5 <constant_11_mobilenet+0x3d5>
    d71f:	48 89 c7             	mov    %rax,%rdi
    d722:	b8 00 00 00 00       	mov    $0x0,%eax
    d727:	e8 04 49 ff ff       	call   2030 <printf@plt>
    d72c:	48 8b 85 58 ff ff ff 	mov    -0xa8(%rbp),%rax
    d733:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
    d737:	bf 5b 00 00 00       	mov    $0x5b,%edi
    d73c:	e8 6f 49 ff ff       	call   20b0 <putchar@plt>
    d741:	48 c7 85 60 ff ff ff 	movq   $0x0,-0xa0(%rbp)
    d748:	00 00 00 00 
    d74c:	e9 d2 00 00 00       	jmp    d823 <omTensorPrint+0x74a>
    d751:	48 83 bd 60 ff ff ff 	cmpq   $0x0,-0xa0(%rbp)
    d758:	00 
    d759:	74 14                	je     d76f <omTensorPrint+0x696>
    d75b:	48 8d 05 83 eb 0c 00 	lea    0xceb83(%rip),%rax        # dc2e5 <constant_11_mobilenet+0x3d5>
    d762:	48 89 c7             	mov    %rax,%rdi
    d765:	b8 00 00 00 00       	mov    $0x0,%eax
    d76a:	e8 c1 48 ff ff       	call   2030 <printf@plt>
    d76f:	48 8b 85 60 ff ff ff 	mov    -0xa0(%rbp),%rax
    d776:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
    d77a:	bf 5b 00 00 00       	mov    $0x5b,%edi
    d77f:	e8 2c 49 ff ff       	call   20b0 <putchar@plt>
    d784:	48 c7 85 68 ff ff ff 	movq   $0x0,-0x98(%rbp)
    d78b:	00 00 00 00 
    d78f:	eb 6c                	jmp    d7fd <omTensorPrint+0x724>
    d791:	48 83 bd 68 ff ff ff 	cmpq   $0x0,-0x98(%rbp)
    d798:	00 
    d799:	74 14                	je     d7af <omTensorPrint+0x6d6>
    d79b:	48 8d 05 43 eb 0c 00 	lea    0xceb43(%rip),%rax        # dc2e5 <constant_11_mobilenet+0x3d5>
    d7a2:	48 89 c7             	mov    %rax,%rdi
    d7a5:	b8 00 00 00 00       	mov    $0x0,%eax
    d7aa:	e8 81 48 ff ff       	call   2030 <printf@plt>
    d7af:	48 8b 85 68 ff ff ff 	mov    -0x98(%rbp),%rax
    d7b6:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
    d7ba:	48 8b 85 10 ff ff ff 	mov    -0xf0(%rbp),%rax
    d7c1:	48 8b 40 20          	mov    0x20(%rax),%rax
    d7c5:	48 8b 55 90          	mov    -0x70(%rbp),%rdx
    d7c9:	48 8d 4d d0          	lea    -0x30(%rbp),%rcx
    d7cd:	48 89 ce             	mov    %rcx,%rsi
    d7d0:	48 89 c7             	mov    %rax,%rdi
    d7d3:	e8 70 f1 ff ff       	call   c948 <computeElemOffset>
    d7d8:	48 89 45 b8          	mov    %rax,-0x48(%rbp)
    d7dc:	8b 95 24 ff ff ff    	mov    -0xdc(%rbp),%edx
    d7e2:	48 8b 4d b8          	mov    -0x48(%rbp),%rcx
    d7e6:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    d7ea:	48 89 ce             	mov    %rcx,%rsi
    d7ed:	48 89 c7             	mov    %rax,%rdi
    d7f0:	e8 8b ed ff ff       	call   c580 <printElement>
    d7f5:	48 83 85 68 ff ff ff 	addq   $0x1,-0x98(%rbp)
    d7fc:	01 
    d7fd:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    d801:	48 83 c0 10          	add    $0x10,%rax
    d805:	48 8b 00             	mov    (%rax),%rax
    d808:	48 39 85 68 ff ff ff 	cmp    %rax,-0x98(%rbp)
    d80f:	7c 80                	jl     d791 <omTensorPrint+0x6b8>
    d811:	bf 5d 00 00 00       	mov    $0x5d,%edi
    d816:	e8 95 48 ff ff       	call   20b0 <putchar@plt>
    d81b:	48 83 85 60 ff ff ff 	addq   $0x1,-0xa0(%rbp)
    d822:	01 
    d823:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    d827:	48 83 c0 08          	add    $0x8,%rax
    d82b:	48 8b 00             	mov    (%rax),%rax
    d82e:	48 39 85 60 ff ff ff 	cmp    %rax,-0xa0(%rbp)
    d835:	0f 8c 16 ff ff ff    	jl     d751 <omTensorPrint+0x678>
    d83b:	bf 5d 00 00 00       	mov    $0x5d,%edi
    d840:	e8 6b 48 ff ff       	call   20b0 <putchar@plt>
    d845:	48 83 85 58 ff ff ff 	addq   $0x1,-0xa8(%rbp)
    d84c:	01 
    d84d:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    d851:	48 8b 00             	mov    (%rax),%rax
    d854:	48 39 85 58 ff ff ff 	cmp    %rax,-0xa8(%rbp)
    d85b:	0f 8c ad fe ff ff    	jl     d70e <omTensorPrint+0x635>
    d861:	bf 5d 00 00 00       	mov    $0x5d,%edi
    d866:	e8 45 48 ff ff       	call   20b0 <putchar@plt>
    d86b:	e9 f0 01 00 00       	jmp    da60 <omTensorPrint+0x987>
    d870:	bf 5b 00 00 00       	mov    $0x5b,%edi
    d875:	e8 36 48 ff ff       	call   20b0 <putchar@plt>
    d87a:	48 c7 85 70 ff ff ff 	movq   $0x0,-0x90(%rbp)
    d881:	00 00 00 00 
    d885:	e9 8e 01 00 00       	jmp    da18 <omTensorPrint+0x93f>
    d88a:	48 83 bd 70 ff ff ff 	cmpq   $0x0,-0x90(%rbp)
    d891:	00 
    d892:	74 14                	je     d8a8 <omTensorPrint+0x7cf>
    d894:	48 8d 05 4a ea 0c 00 	lea    0xcea4a(%rip),%rax        # dc2e5 <constant_11_mobilenet+0x3d5>
    d89b:	48 89 c7             	mov    %rax,%rdi
    d89e:	b8 00 00 00 00       	mov    $0x0,%eax
    d8a3:	e8 88 47 ff ff       	call   2030 <printf@plt>
    d8a8:	48 8b 85 70 ff ff ff 	mov    -0x90(%rbp),%rax
    d8af:	48 89 45 d0          	mov    %rax,-0x30(%rbp)
    d8b3:	bf 5b 00 00 00       	mov    $0x5b,%edi
    d8b8:	e8 f3 47 ff ff       	call   20b0 <putchar@plt>
    d8bd:	48 c7 85 78 ff ff ff 	movq   $0x0,-0x88(%rbp)
    d8c4:	00 00 00 00 
    d8c8:	e9 21 01 00 00       	jmp    d9ee <omTensorPrint+0x915>
    d8cd:	48 83 bd 78 ff ff ff 	cmpq   $0x0,-0x88(%rbp)
    d8d4:	00 
    d8d5:	74 14                	je     d8eb <omTensorPrint+0x812>
    d8d7:	48 8d 05 07 ea 0c 00 	lea    0xcea07(%rip),%rax        # dc2e5 <constant_11_mobilenet+0x3d5>
    d8de:	48 89 c7             	mov    %rax,%rdi
    d8e1:	b8 00 00 00 00       	mov    $0x0,%eax
    d8e6:	e8 45 47 ff ff       	call   2030 <printf@plt>
    d8eb:	48 8b 85 78 ff ff ff 	mov    -0x88(%rbp),%rax
    d8f2:	48 89 45 d8          	mov    %rax,-0x28(%rbp)
    d8f6:	bf 5b 00 00 00       	mov    $0x5b,%edi
    d8fb:	e8 b0 47 ff ff       	call   20b0 <putchar@plt>
    d900:	48 c7 45 80 00 00 00 	movq   $0x0,-0x80(%rbp)
    d907:	00 
    d908:	e9 ba 00 00 00       	jmp    d9c7 <omTensorPrint+0x8ee>
    d90d:	48 83 7d 80 00       	cmpq   $0x0,-0x80(%rbp)
    d912:	74 14                	je     d928 <omTensorPrint+0x84f>
    d914:	48 8d 05 ca e9 0c 00 	lea    0xce9ca(%rip),%rax        # dc2e5 <constant_11_mobilenet+0x3d5>
    d91b:	48 89 c7             	mov    %rax,%rdi
    d91e:	b8 00 00 00 00       	mov    $0x0,%eax
    d923:	e8 08 47 ff ff       	call   2030 <printf@plt>
    d928:	48 8b 45 80          	mov    -0x80(%rbp),%rax
    d92c:	48 89 45 e0          	mov    %rax,-0x20(%rbp)
    d930:	bf 5b 00 00 00       	mov    $0x5b,%edi
    d935:	e8 76 47 ff ff       	call   20b0 <putchar@plt>
    d93a:	48 c7 45 88 00 00 00 	movq   $0x0,-0x78(%rbp)
    d941:	00 
    d942:	eb 63                	jmp    d9a7 <omTensorPrint+0x8ce>
    d944:	48 83 7d 88 00       	cmpq   $0x0,-0x78(%rbp)
    d949:	74 14                	je     d95f <omTensorPrint+0x886>
    d94b:	48 8d 05 93 e9 0c 00 	lea    0xce993(%rip),%rax        # dc2e5 <constant_11_mobilenet+0x3d5>
    d952:	48 89 c7             	mov    %rax,%rdi
    d955:	b8 00 00 00 00       	mov    $0x0,%eax
    d95a:	e8 d1 46 ff ff       	call   2030 <printf@plt>
    d95f:	48 8b 45 88          	mov    -0x78(%rbp),%rax
    d963:	48 89 45 e8          	mov    %rax,-0x18(%rbp)
    d967:	48 8b 85 10 ff ff ff 	mov    -0xf0(%rbp),%rax
    d96e:	48 8b 40 20          	mov    0x20(%rax),%rax
    d972:	48 8b 55 90          	mov    -0x70(%rbp),%rdx
    d976:	48 8d 4d d0          	lea    -0x30(%rbp),%rcx
    d97a:	48 89 ce             	mov    %rcx,%rsi
    d97d:	48 89 c7             	mov    %rax,%rdi
    d980:	e8 c3 ef ff ff       	call   c948 <computeElemOffset>
    d985:	48 89 45 b0          	mov    %rax,-0x50(%rbp)
    d989:	8b 95 24 ff ff ff    	mov    -0xdc(%rbp),%edx
    d98f:	48 8b 4d b0          	mov    -0x50(%rbp),%rcx
    d993:	48 8b 45 a8          	mov    -0x58(%rbp),%rax
    d997:	48 89 ce             	mov    %rcx,%rsi
    d99a:	48 89 c7             	mov    %rax,%rdi
    d99d:	e8 de eb ff ff       	call   c580 <printElement>
    d9a2:	48 83 45 88 01       	addq   $0x1,-0x78(%rbp)
    d9a7:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    d9ab:	48 83 c0 18          	add    $0x18,%rax
    d9af:	48 8b 00             	mov    (%rax),%rax
    d9b2:	48 39 45 88          	cmp    %rax,-0x78(%rbp)
    d9b6:	7c 8c                	jl     d944 <omTensorPrint+0x86b>
    d9b8:	bf 5d 00 00 00       	mov    $0x5d,%edi
    d9bd:	e8 ee 46 ff ff       	call   20b0 <putchar@plt>
    d9c2:	48 83 45 80 01       	addq   $0x1,-0x80(%rbp)
    d9c7:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    d9cb:	48 83 c0 10          	add    $0x10,%rax
    d9cf:	48 8b 00             	mov    (%rax),%rax
    d9d2:	48 39 45 80          	cmp    %rax,-0x80(%rbp)
    d9d6:	0f 8c 31 ff ff ff    	jl     d90d <omTensorPrint+0x834>
    d9dc:	bf 5d 00 00 00       	mov    $0x5d,%edi
    d9e1:	e8 ca 46 ff ff       	call   20b0 <putchar@plt>
    d9e6:	48 83 85 78 ff ff ff 	addq   $0x1,-0x88(%rbp)
    d9ed:	01 
    d9ee:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    d9f2:	48 83 c0 08          	add    $0x8,%rax
    d9f6:	48 8b 00             	mov    (%rax),%rax
    d9f9:	48 39 85 78 ff ff ff 	cmp    %rax,-0x88(%rbp)
    da00:	0f 8c c7 fe ff ff    	jl     d8cd <omTensorPrint+0x7f4>
    da06:	bf 5d 00 00 00       	mov    $0x5d,%edi
    da0b:	e8 a0 46 ff ff       	call   20b0 <putchar@plt>
    da10:	48 83 85 70 ff ff ff 	addq   $0x1,-0x90(%rbp)
    da17:	01 
    da18:	48 8b 45 98          	mov    -0x68(%rbp),%rax
    da1c:	48 8b 00             	mov    (%rax),%rax
    da1f:	48 39 85 70 ff ff ff 	cmp    %rax,-0x90(%rbp)
    da26:	0f 8c 5e fe ff ff    	jl     d88a <omTensorPrint+0x7b1>
    da2c:	bf 5d 00 00 00       	mov    $0x5d,%edi
    da31:	e8 7a 46 ff ff       	call   20b0 <putchar@plt>
    da36:	eb 28                	jmp    da60 <omTensorPrint+0x987>
    da38:	48 8d 05 e1 e8 0c 00 	lea    0xce8e1(%rip),%rax        # dc320 <__PRETTY_FUNCTION__.1>
    da3f:	48 89 c1             	mov    %rax,%rcx
    da42:	ba 2d 02 00 00       	mov    $0x22d,%edx
    da47:	48 8d 05 c2 e6 0c 00 	lea    0xce6c2(%rip),%rax        # dc110 <constant_11_mobilenet+0x200>
    da4e:	48 89 c6             	mov    %rax,%rsi
    da51:	48 8d 05 90 e8 0c 00 	lea    0xce890(%rip),%rax        # dc2e8 <constant_11_mobilenet+0x3d8>
    da58:	48 89 c7             	mov    %rax,%rdi
    da5b:	e8 20 46 ff ff       	call   2080 <__assert_fail@plt>
    da60:	48 8d 05 9c e8 0c 00 	lea    0xce89c(%rip),%rax        # dc303 <constant_11_mobilenet+0x3f3>
    da67:	48 89 c7             	mov    %rax,%rdi
    da6a:	e8 f1 45 ff ff       	call   2060 <puts@plt>
    da6f:	90                   	nop
    da70:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    da74:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    da7b:	00 00 
    da7d:	74 05                	je     da84 <omTensorPrint+0x9ab>
    da7f:	e8 fc 46 ff ff       	call   2180 <__stack_chk_fail@plt>
    da84:	c9                   	leave  
    da85:	c3                   	ret    

000000000000da86 <omTensorListCreate>:
    da86:	f3 0f 1e fa          	endbr64 
    da8a:	55                   	push   %rbp
    da8b:	48 89 e5             	mov    %rsp,%rbp
    da8e:	48 83 ec 20          	sub    $0x20,%rsp
    da92:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    da96:	48 89 75 e0          	mov    %rsi,-0x20(%rbp)
    da9a:	bf 18 00 00 00       	mov    $0x18,%edi
    da9f:	e8 2c 46 ff ff       	call   20d0 <malloc@plt>
    daa4:	48 89 45 f0          	mov    %rax,-0x10(%rbp)
    daa8:	48 83 7d f0 00       	cmpq   $0x0,-0x10(%rbp)
    daad:	75 07                	jne    dab6 <omTensorListCreate+0x30>
    daaf:	b8 00 00 00 00       	mov    $0x0,%eax
    dab4:	eb 6b                	jmp    db21 <omTensorListCreate+0x9b>
    dab6:	48 8b 45 e0          	mov    -0x20(%rbp),%rax
    daba:	48 c1 e0 03          	shl    $0x3,%rax
    dabe:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    dac2:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    dac6:	48 8b 55 e0          	mov    -0x20(%rbp),%rdx
    daca:	48 89 50 08          	mov    %rdx,0x8(%rax)
    dace:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    dad2:	48 89 c7             	mov    %rax,%rdi
    dad5:	e8 f6 45 ff ff       	call   20d0 <malloc@plt>
    dada:	48 89 c2             	mov    %rax,%rdx
    dadd:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    dae1:	48 89 10             	mov    %rdx,(%rax)
    dae4:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    dae8:	48 8b 00             	mov    (%rax),%rax
    daeb:	48 85 c0             	test   %rax,%rax
    daee:	75 13                	jne    db03 <omTensorListCreate+0x7d>
    daf0:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    daf4:	48 89 c7             	mov    %rax,%rdi
    daf7:	e8 14 46 ff ff       	call   2110 <free@plt>
    dafc:	b8 00 00 00 00       	mov    $0x0,%eax
    db01:	eb 1e                	jmp    db21 <omTensorListCreate+0x9b>
    db03:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    db07:	48 8b 00             	mov    (%rax),%rax
    db0a:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    db0e:	48 8b 4d e8          	mov    -0x18(%rbp),%rcx
    db12:	48 89 ce             	mov    %rcx,%rsi
    db15:	48 89 c7             	mov    %rax,%rdi
    db18:	e8 c3 46 ff ff       	call   21e0 <memcpy@plt>
    db1d:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    db21:	c9                   	leave  
    db22:	c3                   	ret    

000000000000db23 <omTensorListDestroy>:
    db23:	f3 0f 1e fa          	endbr64 
    db27:	55                   	push   %rbp
    db28:	48 89 e5             	mov    %rsp,%rbp
    db2b:	48 83 ec 20          	sub    $0x20,%rsp
    db2f:	48 89 7d e8          	mov    %rdi,-0x18(%rbp)
    db33:	48 83 7d e8 00       	cmpq   $0x0,-0x18(%rbp)
    db38:	74 48                	je     db82 <omTensorListDestroy+0x5f>
    db3a:	48 c7 45 f8 00 00 00 	movq   $0x0,-0x8(%rbp)
    db41:	00 
    db42:	eb 22                	jmp    db66 <omTensorListDestroy+0x43>
    db44:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    db48:	48 8b 10             	mov    (%rax),%rdx
    db4b:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    db4f:	48 c1 e0 03          	shl    $0x3,%rax
    db53:	48 01 d0             	add    %rdx,%rax
    db56:	48 8b 00             	mov    (%rax),%rax
    db59:	48 89 c7             	mov    %rax,%rdi
    db5c:	e8 4f 46 ff ff       	call   21b0 <omTensorDestroy@plt>
    db61:	48 83 45 f8 01       	addq   $0x1,-0x8(%rbp)
    db66:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    db6a:	48 8b 40 08          	mov    0x8(%rax),%rax
    db6e:	48 39 45 f8          	cmp    %rax,-0x8(%rbp)
    db72:	7c d0                	jl     db44 <omTensorListDestroy+0x21>
    db74:	48 8b 45 e8          	mov    -0x18(%rbp),%rax
    db78:	48 89 c7             	mov    %rax,%rdi
    db7b:	e8 05 00 00 00       	call   db85 <omTensorListDestroyShallow>
    db80:	eb 01                	jmp    db83 <omTensorListDestroy+0x60>
    db82:	90                   	nop
    db83:	c9                   	leave  
    db84:	c3                   	ret    

000000000000db85 <omTensorListDestroyShallow>:
    db85:	f3 0f 1e fa          	endbr64 
    db89:	55                   	push   %rbp
    db8a:	48 89 e5             	mov    %rsp,%rbp
    db8d:	48 83 ec 10          	sub    $0x10,%rsp
    db91:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    db95:	48 83 7d f8 00       	cmpq   $0x0,-0x8(%rbp)
    db9a:	74 1d                	je     dbb9 <omTensorListDestroyShallow+0x34>
    db9c:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    dba0:	48 8b 00             	mov    (%rax),%rax
    dba3:	48 89 c7             	mov    %rax,%rdi
    dba6:	e8 65 45 ff ff       	call   2110 <free@plt>
    dbab:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    dbaf:	48 89 c7             	mov    %rax,%rdi
    dbb2:	e8 59 45 ff ff       	call   2110 <free@plt>
    dbb7:	eb 01                	jmp    dbba <omTensorListDestroyShallow+0x35>
    dbb9:	90                   	nop
    dbba:	c9                   	leave  
    dbbb:	c3                   	ret    

000000000000dbbc <omTensorListGetOmtArray>:
    dbbc:	f3 0f 1e fa          	endbr64 
    dbc0:	55                   	push   %rbp
    dbc1:	48 89 e5             	mov    %rsp,%rbp
    dbc4:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    dbc8:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    dbcc:	48 8b 00             	mov    (%rax),%rax
    dbcf:	5d                   	pop    %rbp
    dbd0:	c3                   	ret    

000000000000dbd1 <omTensorListGetSize>:
    dbd1:	f3 0f 1e fa          	endbr64 
    dbd5:	55                   	push   %rbp
    dbd6:	48 89 e5             	mov    %rsp,%rbp
    dbd9:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    dbdd:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    dbe1:	48 8b 40 08          	mov    0x8(%rax),%rax
    dbe5:	5d                   	pop    %rbp
    dbe6:	c3                   	ret    

000000000000dbe7 <omTensorListGetOmtByIndex>:
    dbe7:	f3 0f 1e fa          	endbr64 
    dbeb:	55                   	push   %rbp
    dbec:	48 89 e5             	mov    %rsp,%rbp
    dbef:	48 83 ec 10          	sub    $0x10,%rsp
    dbf3:	48 89 7d f8          	mov    %rdi,-0x8(%rbp)
    dbf7:	48 89 75 f0          	mov    %rsi,-0x10(%rbp)
    dbfb:	48 83 7d f0 00       	cmpq   $0x0,-0x10(%rbp)
    dc00:	79 28                	jns    dc2a <omTensorListGetOmtByIndex+0x43>
    dc02:	48 8d 05 97 e7 0c 00 	lea    0xce797(%rip),%rax        # dc3a0 <__PRETTY_FUNCTION__.0>
    dc09:	48 89 c1             	mov    %rax,%rcx
    dc0c:	ba 77 00 00 00       	mov    $0x77,%edx
    dc11:	48 8d 05 28 e7 0c 00 	lea    0xce728(%rip),%rax        # dc340 <__PRETTY_FUNCTION__.0+0x10>
    dc18:	48 89 c6             	mov    %rax,%rsi
    dc1b:	48 8d 05 5e e7 0c 00 	lea    0xce75e(%rip),%rax        # dc380 <__PRETTY_FUNCTION__.0+0x50>
    dc22:	48 89 c7             	mov    %rax,%rdi
    dc25:	e8 56 44 ff ff       	call   2080 <__assert_fail@plt>
    dc2a:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    dc2e:	48 8b 40 08          	mov    0x8(%rax),%rax
    dc32:	48 39 45 f0          	cmp    %rax,-0x10(%rbp)
    dc36:	7c 28                	jl     dc60 <omTensorListGetOmtByIndex+0x79>
    dc38:	48 8d 05 61 e7 0c 00 	lea    0xce761(%rip),%rax        # dc3a0 <__PRETTY_FUNCTION__.0>
    dc3f:	48 89 c1             	mov    %rax,%rcx
    dc42:	ba 78 00 00 00       	mov    $0x78,%edx
    dc47:	48 8d 05 f2 e6 0c 00 	lea    0xce6f2(%rip),%rax        # dc340 <__PRETTY_FUNCTION__.0+0x10>
    dc4e:	48 89 c6             	mov    %rax,%rsi
    dc51:	48 8d 05 33 e7 0c 00 	lea    0xce733(%rip),%rax        # dc38b <__PRETTY_FUNCTION__.0+0x5b>
    dc58:	48 89 c7             	mov    %rax,%rdi
    dc5b:	e8 20 44 ff ff       	call   2080 <__assert_fail@plt>
    dc60:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    dc64:	48 8b 10             	mov    (%rax),%rdx
    dc67:	48 8b 45 f0          	mov    -0x10(%rbp),%rax
    dc6b:	48 c1 e0 03          	shl    $0x3,%rax
    dc6f:	48 01 d0             	add    %rdx,%rax
    dc72:	48 8b 00             	mov    (%rax),%rax
    dc75:	c9                   	leave  
    dc76:	c3                   	ret    

000000000000dc77 <om_f16_to_f32>:
    dc77:	f3 0f 1e fa          	endbr64 
    dc7b:	55                   	push   %rbp
    dc7c:	48 89 e5             	mov    %rsp,%rbp
    dc7f:	48 83 ec 40          	sub    $0x40,%rsp
    dc83:	89 f8                	mov    %edi,%eax
    dc85:	66 89 45 cc          	mov    %ax,-0x34(%rbp)
    dc89:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    dc90:	00 00 
    dc92:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    dc96:	31 c0                	xor    %eax,%eax
    dc98:	0f b7 45 cc          	movzwl -0x34(%rbp),%eax
    dc9c:	89 45 dc             	mov    %eax,-0x24(%rbp)
    dc9f:	8b 45 dc             	mov    -0x24(%rbp),%eax
    dca2:	25 00 80 00 00       	and    $0x8000,%eax
    dca7:	89 45 f4             	mov    %eax,-0xc(%rbp)
    dcaa:	8b 45 dc             	mov    -0x24(%rbp),%eax
    dcad:	33 45 f4             	xor    -0xc(%rbp),%eax
    dcb0:	89 45 dc             	mov    %eax,-0x24(%rbp)
    dcb3:	8b 45 dc             	mov    -0x24(%rbp),%eax
    dcb6:	3d ff 7b 00 00       	cmp    $0x7bff,%eax
    dcbb:	0f 96 c0             	setbe  %al
    dcbe:	88 45 db             	mov    %al,-0x25(%rbp)
    dcc1:	8b 45 f4             	mov    -0xc(%rbp),%eax
    dcc4:	c1 e0 10             	shl    $0x10,%eax
    dcc7:	89 c2                	mov    %eax,%edx
    dcc9:	8b 45 dc             	mov    -0x24(%rbp),%eax
    dccc:	c1 e0 0d             	shl    $0xd,%eax
    dccf:	09 d0                	or     %edx,%eax
    dcd1:	89 45 dc             	mov    %eax,-0x24(%rbp)
    dcd4:	8b 45 dc             	mov    -0x24(%rbp),%eax
    dcd7:	89 45 e0             	mov    %eax,-0x20(%rbp)
    dcda:	8b 05 bc e7 0c 00    	mov    0xce7bc(%rip),%eax        # dc49c <bias_mul.4>
    dce0:	89 45 e4             	mov    %eax,-0x1c(%rbp)
    dce3:	f3 0f 10 4d e0       	movss  -0x20(%rbp),%xmm1
    dce8:	f3 0f 10 45 e4       	movss  -0x1c(%rbp),%xmm0
    dced:	f3 0f 59 c1          	mulss  %xmm1,%xmm0
    dcf1:	f3 0f 11 45 e8       	movss  %xmm0,-0x18(%rbp)
    dcf6:	8b 45 e8             	mov    -0x18(%rbp),%eax
    dcf9:	89 45 ec             	mov    %eax,-0x14(%rbp)
    dcfc:	80 7d db 00          	cmpb   $0x0,-0x25(%rbp)
    dd00:	74 07                	je     dd09 <om_f16_to_f32+0x92>
    dd02:	ba 00 00 00 00       	mov    $0x0,%edx
    dd07:	eb 05                	jmp    dd0e <om_f16_to_f32+0x97>
    dd09:	ba 00 00 80 7f       	mov    $0x7f800000,%edx
    dd0e:	8b 45 ec             	mov    -0x14(%rbp),%eax
    dd11:	09 d0                	or     %edx,%eax
    dd13:	89 45 ec             	mov    %eax,-0x14(%rbp)
    dd16:	8b 45 ec             	mov    -0x14(%rbp),%eax
    dd19:	89 45 f0             	mov    %eax,-0x10(%rbp)
    dd1c:	f3 0f 10 45 f0       	movss  -0x10(%rbp),%xmm0
    dd21:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    dd25:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    dd2c:	00 00 
    dd2e:	74 05                	je     dd35 <om_f16_to_f32+0xbe>
    dd30:	e8 4b 44 ff ff       	call   2180 <__stack_chk_fail@plt>
    dd35:	c9                   	leave  
    dd36:	c3                   	ret    

000000000000dd37 <om_f32_to_f16>:
    dd37:	f3 0f 1e fa          	endbr64 
    dd3b:	55                   	push   %rbp
    dd3c:	48 89 e5             	mov    %rsp,%rbp
    dd3f:	48 83 ec 40          	sub    $0x40,%rsp
    dd43:	f3 0f 11 45 cc       	movss  %xmm0,-0x34(%rbp)
    dd48:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    dd4f:	00 00 
    dd51:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    dd55:	31 c0                	xor    %eax,%eax
    dd57:	8b 05 43 e7 0c 00    	mov    0xce743(%rip),%eax        # dc4a0 <bias_mul.3>
    dd5d:	89 45 d4             	mov    %eax,-0x2c(%rbp)
    dd60:	8b 45 cc             	mov    -0x34(%rbp),%eax
    dd63:	89 45 d8             	mov    %eax,-0x28(%rbp)
    dd66:	8b 45 d8             	mov    -0x28(%rbp),%eax
    dd69:	25 00 00 00 80       	and    $0x80000000,%eax
    dd6e:	89 45 f4             	mov    %eax,-0xc(%rbp)
    dd71:	8b 45 d8             	mov    -0x28(%rbp),%eax
    dd74:	33 45 f4             	xor    -0xc(%rbp),%eax
    dd77:	89 45 d8             	mov    %eax,-0x28(%rbp)
    dd7a:	8b 45 d8             	mov    -0x28(%rbp),%eax
    dd7d:	3d 00 00 80 7f       	cmp    $0x7f800000,%eax
    dd82:	0f 97 c0             	seta   %al
    dd85:	88 45 d2             	mov    %al,-0x2e(%rbp)
    dd88:	8b 05 16 e7 0c 00    	mov    0xce716(%rip),%eax        # dc4a4 <sub_rnd.2>
    dd8e:	89 45 dc             	mov    %eax,-0x24(%rbp)
    dd91:	8b 05 11 e7 0c 00    	mov    0xce711(%rip),%eax        # dc4a8 <sub_mul.1>
    dd97:	89 45 e0             	mov    %eax,-0x20(%rbp)
    dd9a:	8b 55 d8             	mov    -0x28(%rbp),%edx
    dd9d:	8b 05 09 e7 0c 00    	mov    0xce709(%rip),%eax        # dc4ac <min_norm.0>
    dda3:	39 c2                	cmp    %eax,%edx
    dda5:	0f 92 c0             	setb   %al
    dda8:	88 45 d3             	mov    %al,-0x2d(%rbp)
    ddab:	8b 45 d8             	mov    -0x28(%rbp),%eax
    ddae:	89 45 e4             	mov    %eax,-0x1c(%rbp)
    ddb1:	f3 0f 10 45 e4       	movss  -0x1c(%rbp),%xmm0
    ddb6:	f3 0f 11 45 e8       	movss  %xmm0,-0x18(%rbp)
    ddbb:	f3 0f 10 4d e8       	movss  -0x18(%rbp),%xmm1
    ddc0:	f3 0f 10 45 dc       	movss  -0x24(%rbp),%xmm0
    ddc5:	f3 0f 59 c1          	mulss  %xmm1,%xmm0
    ddc9:	f3 0f 11 45 e8       	movss  %xmm0,-0x18(%rbp)
    ddce:	f3 0f 10 4d e8       	movss  -0x18(%rbp),%xmm1
    ddd3:	f3 0f 10 45 e0       	movss  -0x20(%rbp),%xmm0
    ddd8:	f3 0f 59 c1          	mulss  %xmm1,%xmm0
    dddc:	f3 0f 11 45 e8       	movss  %xmm0,-0x18(%rbp)
    dde1:	f3 0f 10 4d e4       	movss  -0x1c(%rbp),%xmm1
    dde6:	f3 0f 10 45 d4       	movss  -0x2c(%rbp),%xmm0
    ddeb:	f3 0f 59 c1          	mulss  %xmm1,%xmm0
    ddef:	f3 0f 11 45 e4       	movss  %xmm0,-0x1c(%rbp)
    ddf4:	8b 45 e4             	mov    -0x1c(%rbp),%eax
    ddf7:	89 45 ec             	mov    %eax,-0x14(%rbp)
    ddfa:	8b 45 ec             	mov    -0x14(%rbp),%eax
    ddfd:	89 45 d8             	mov    %eax,-0x28(%rbp)
    de00:	8b 45 d8             	mov    -0x28(%rbp),%eax
    de03:	c1 e8 0d             	shr    $0xd,%eax
    de06:	83 e0 01             	and    $0x1,%eax
    de09:	89 c2                	mov    %eax,%edx
    de0b:	8b 45 d8             	mov    -0x28(%rbp),%eax
    de0e:	01 d0                	add    %edx,%eax
    de10:	89 45 d8             	mov    %eax,-0x28(%rbp)
    de13:	8b 45 d8             	mov    -0x28(%rbp),%eax
    de16:	05 ff 0f 00 00       	add    $0xfff,%eax
    de1b:	89 45 d8             	mov    %eax,-0x28(%rbp)
    de1e:	8b 45 e8             	mov    -0x18(%rbp),%eax
    de21:	89 45 f0             	mov    %eax,-0x10(%rbp)
    de24:	80 7d d3 00          	cmpb   $0x0,-0x2d(%rbp)
    de28:	74 06                	je     de30 <om_f32_to_f16+0xf9>
    de2a:	8b 45 f0             	mov    -0x10(%rbp),%eax
    de2d:	89 45 d8             	mov    %eax,-0x28(%rbp)
    de30:	8b 45 d8             	mov    -0x28(%rbp),%eax
    de33:	c1 e8 0d             	shr    $0xd,%eax
    de36:	89 45 d8             	mov    %eax,-0x28(%rbp)
    de39:	8b 45 d8             	mov    -0x28(%rbp),%eax
    de3c:	3d 00 7c 00 00       	cmp    $0x7c00,%eax
    de41:	76 07                	jbe    de4a <om_f32_to_f16+0x113>
    de43:	c7 45 d8 00 7c 00 00 	movl   $0x7c00,-0x28(%rbp)
    de4a:	80 7d d2 00          	cmpb   $0x0,-0x2e(%rbp)
    de4e:	74 07                	je     de57 <om_f32_to_f16+0x120>
    de50:	c7 45 d8 00 7e 00 00 	movl   $0x7e00,-0x28(%rbp)
    de57:	8b 45 f4             	mov    -0xc(%rbp),%eax
    de5a:	c1 e8 10             	shr    $0x10,%eax
    de5d:	89 c2                	mov    %eax,%edx
    de5f:	8b 45 d8             	mov    -0x28(%rbp),%eax
    de62:	09 d0                	or     %edx,%eax
    de64:	89 45 d8             	mov    %eax,-0x28(%rbp)
    de67:	8b 45 d8             	mov    -0x28(%rbp),%eax
    de6a:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    de6e:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    de75:	00 00 
    de77:	74 05                	je     de7e <om_f32_to_f16+0x147>
    de79:	e8 02 43 ff ff       	call   2180 <__stack_chk_fail@plt>
    de7e:	c9                   	leave  
    de7f:	c3                   	ret    

000000000000de80 <om_bf16_to_f32>:
    de80:	f3 0f 1e fa          	endbr64 
    de84:	55                   	push   %rbp
    de85:	48 89 e5             	mov    %rsp,%rbp
    de88:	48 83 ec 20          	sub    $0x20,%rsp
    de8c:	89 f8                	mov    %edi,%eax
    de8e:	66 89 45 ec          	mov    %ax,-0x14(%rbp)
    de92:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    de99:	00 00 
    de9b:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    de9f:	31 c0                	xor    %eax,%eax
    dea1:	0f b7 45 ec          	movzwl -0x14(%rbp),%eax
    dea5:	c1 e0 10             	shl    $0x10,%eax
    dea8:	89 45 f0             	mov    %eax,-0x10(%rbp)
    deab:	8b 45 f0             	mov    -0x10(%rbp),%eax
    deae:	89 45 f4             	mov    %eax,-0xc(%rbp)
    deb1:	f3 0f 10 45 f4       	movss  -0xc(%rbp),%xmm0
    deb6:	48 8b 45 f8          	mov    -0x8(%rbp),%rax
    deba:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    dec1:	00 00 
    dec3:	74 05                	je     deca <om_bf16_to_f32+0x4a>
    dec5:	e8 b6 42 ff ff       	call   2180 <__stack_chk_fail@plt>
    deca:	c9                   	leave  
    decb:	c3                   	ret    

000000000000decc <om_f32_to_bf16>:
    decc:	f3 0f 1e fa          	endbr64 
    ded0:	55                   	push   %rbp
    ded1:	48 89 e5             	mov    %rsp,%rbp
    ded4:	48 83 ec 20          	sub    $0x20,%rsp
    ded8:	f3 0f 11 45 ec       	movss  %xmm0,-0x14(%rbp)
    dedd:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    dee4:	00 00 
    dee6:	48 89 45 f8          	mov    %rax,-0x8(%rbp)
    deea:	31 c0                	xor    %eax,%eax
    deec:	8b 45 ec             	mov    -0x14(%rbp),%eax
    deef:	89 45 f4             	mov    %eax,-0xc(%rbp)
    def2:	8b 45 f4             	mov    -0xc(%rbp),%eax
    def5:	25 ff ff 01 00       	and    $0x1ffff,%eax
    defa:	3d 00 80 01 00       	cmp    $0x18000,%eax
    deff:	75 07                	jne    df08 <om_f32_to_bf16+0x3c>
    df01:	ba ff 7f 01 00       	mov    $0x17fff,%edx
    df06:	eb 05                	jmp    df0d <om_f32_to_bf16+0x41>
    df08:	ba ff 7f 00 00       	mov    $0x7fff,%edx
    df0d:	8b 45 f4             	mov    -0xc(%rbp),%eax
    df10:	01 d0                	add    %edx,%eax
    df12:	89 45 f4             	mov    %eax,-0xc(%rbp)
    df15:	8b 45 f4             	mov    -0xc(%rbp),%eax
    df18:	c1 e8 10             	shr    $0x10,%eax
    df1b:	66 89 45 f2          	mov    %ax,-0xe(%rbp)
    df1f:	0f b7 45 f2          	movzwl -0xe(%rbp),%eax
    df23:	25 ff 7f 00 00       	and    $0x7fff,%eax
    df28:	3d 80 7f 00 00       	cmp    $0x7f80,%eax
    df2d:	75 13                	jne    df42 <om_f32_to_bf16+0x76>
    df2f:	f3 0f 10 45 ec       	movss  -0x14(%rbp),%xmm0
    df34:	0f 2e c0             	ucomiss %xmm0,%xmm0
    df37:	7b 09                	jnp    df42 <om_f32_to_bf16+0x76>
    df39:	0f b7 45 f2          	movzwl -0xe(%rbp),%eax
    df3d:	83 c0 40             	add    $0x40,%eax
    df40:	eb 24                	jmp    df66 <om_f32_to_bf16+0x9a>
    df42:	0f b7 45 f2          	movzwl -0xe(%rbp),%eax
    df46:	25 ff 7f 00 00       	and    $0x7fff,%eax
    df4b:	85 c0                	test   %eax,%eax
    df4d:	75 13                	jne    df62 <om_f32_to_bf16+0x96>
    df4f:	f3 0f 10 45 ec       	movss  -0x14(%rbp),%xmm0
    df54:	0f 2e c0             	ucomiss %xmm0,%xmm0
    df57:	7b 09                	jnp    df62 <om_f32_to_bf16+0x96>
    df59:	0f b7 45 f2          	movzwl -0xe(%rbp),%eax
    df5d:	83 e8 01             	sub    $0x1,%eax
    df60:	eb 04                	jmp    df66 <om_f32_to_bf16+0x9a>
    df62:	0f b7 45 f2          	movzwl -0xe(%rbp),%eax
    df66:	48 8b 55 f8          	mov    -0x8(%rbp),%rdx
    df6a:	64 48 2b 14 25 28 00 	sub    %fs:0x28,%rdx
    df71:	00 00 
    df73:	74 05                	je     df7a <om_f32_to_bf16+0xae>
    df75:	e8 06 42 ff ff       	call   2180 <__stack_chk_fail@plt>
    df7a:	c9                   	leave  
    df7b:	c3                   	ret    

Disassembly of section .fini:

000000000000df7c <_fini>:
    df7c:	f3 0f 1e fa          	endbr64 
    df80:	48 83 ec 08          	sub    $0x8,%rsp
    df84:	48 83 c4 08          	add    $0x8,%rsp
    df88:	c3                   	ret    
