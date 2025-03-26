all: convolution_serial.c
	gcc -g -Wall -o convolution_serial convolution_serial.c

run:
	make
	./convolution_serial 2048 2048 ../../../test_data/mat_input.csv result_prob3_serial.csv time_prob3_serial.csv

lean:
	$(RM) convolution_serial
