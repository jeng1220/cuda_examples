all:
	nvcc -O3 \
 	--generate-code arch=compute_70,code=sm_70 \
 	--generate-code arch=compute_80,code=sm_80 \
 	--generate-code arch=compute_90,code=sm_90 \
 	foo.cu --extended-lambda -lcufft -o fft_cross_correlation
clean:
	rm -f fft_cross_correlation
