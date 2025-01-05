all:
	nvcc -O3 \
	--generate-code arch=compute_70,code=sm_70 \
	--generate-code arch=compute_80,code=sm_80 \
	--generate-code arch=compute_90,code=sm_90 \
	foo.cu --extended-lambda -lcufft -o basic
	nvcc -O3 \
	--generate-code arch=compute_70,code=sm_70 \
	--generate-code arch=compute_80,code=sm_80 \
	--generate-code arch=compute_90,code=sm_90 \
	advanced.cu --extended-lambda -lcufft -o advanced
clean:
	rm -f basic advanced
