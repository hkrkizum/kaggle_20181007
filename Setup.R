library(keras)
library(tensorflow)

gpu_options <- tf$GPUOptions(per_process_gpu_memory_fraction = 0.3)
config <- tf$ConfigProto(gpu_options = gpu_options)
k_set_session(tf$Session(config = config))