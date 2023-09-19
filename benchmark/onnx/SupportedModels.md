
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
    - 0.0015535354614257812, 0.002940654754638672, 1.8928790669122162, PASSED
- Resnet
    - 0.0005571842193603516, 0.0007615089416503906, 1.3667094565682498, PASSED

### Failing Models

- dscnn
    - failing due to packing of input data. The shape doesn't match what is expected for `convert_tensor2dc()`. 
    - Error message: `uint32_t small::convert_tensor2dc(const ScalarT*, small::BufferTypeEnum, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, ScalarT*) [with ScalarT = float; uint32_t = unsigned int]: Assertion 'C_i == 3' failed.`

- autencoder
    - seg faults