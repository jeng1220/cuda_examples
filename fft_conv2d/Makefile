all: basic advanced advanced_lto
basic: basic.cu
	nvcc -O3 \
	--generate-code arch=compute_80,code=sm_80 \
	--generate-code arch=compute_90,code=sm_90 \
	basic.cu --extended-lambda -lcufft -o basic
advanced: advanced.cu
	nvcc -O3 \
	--generate-code arch=compute_80,code=sm_80 \
	--generate-code arch=compute_90,code=sm_90 \
	advanced.cu --extended-lambda -lcufft -o advanced
advanced_lto: advanced_lto.cu fft_callback.cu nvrtc_helper.h
	nvcc -O3 \
	--generate-code arch=compute_80,code=sm_80 \
	--generate-code arch=compute_90,code=sm_90 \
	advanced_lto.cu --extended-lambda -lcufft -lnvrtc -o advanced_lto
clean:
	rm -f basic advanced advanced_lto
