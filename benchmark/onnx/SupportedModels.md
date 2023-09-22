
Note: 
- models were taken from ContextDependentFusion repo and converted to onnx using `tf2onnx`
- Performance is gathered using `run_onnx_model.py` using the best of 500 runs
- Performance is shown as `small_time, pytorch_time, pytorch_time/small_time, correctness`
- Time is in seconds
- Performance is gathered using zen2 with 6 threads
- onnx to pytorch conversion is done using `onnx2torch`
    - Originally I was using `onnx2pytorch` but it was failing to convert dscnn correctly

### Supported Models

- Mobilenet
    - 0.0014078617095947266, 0.0029659271240234375, 2.1066892464013547, PASSED
- Resnet
    - 0.0006041526794433594, 0.0007834434509277344, 1.2967640094711919, PASSED
- Autoencoder
    - 7.677078247070312e-05, 0.00015401840209960938, 2.0062111801242235, PASSED

### Failing Models

- dscnn
    - failing due to packing of input data. The shape doesn't match what is expected for `convert_tensor2dc()`. 
    - Error message: `uint32_t small::convert_tensor2dc(const ScalarT*, small::BufferTypeEnum, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, ScalarT*) [with ScalarT = float; uint32_t = unsigned int]: Assertion 'C_i == 3' failed.`
