#include "tests.h"
#include <iostream>
#include "Types.h"

#define debug 1

/*
Computtes the software mac for comparison
*/
i32 software_mac(const i8 *weights, const i8 *activation, int n){
    ui32 sum = 0;
    for(int i =0; i < n; i ++){
        sum += activation[i] * weights[i];
    }
    return sum;
}


#ifdef ZEDBOARD
#include "xparameters.h"
#include "xllfifo_hw.h"
#include "xil_io.h"

void send_mac(const i8 *weight, const i8 *activation, int n){
    for(int i = 0; i < n; i++){
        ui32 packed = (static_cast<ui8>(weight[i]) << 8) | static_cast<ui8>(activation[i]);
            Xil_Out32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_TDFD_OFFSET, packed);
    }
    Xil_Out32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_TLF_OFFSET, n * 4);
}


// heavily based off of lab document
ui32 recieve_mac(){
    ui32 data = 0;
    while (Xil_In32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_RDFO_OFFSET) == 0);
    while(true){
        ui32 read_words = Xil_In32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_RLF_OFFSET);

        for(int i = 0; i < (read_words & 0x7FFFFFFFUL); i+=4){
            data = (int32_t)Xil_In32(XPAR_AXI_FIFO_0_BASEADDR + XLLF_RDFD_OFFSET);
        }

        if(!(read_words & (1<<31))){
            break;
        }
    }
    return data;
}

#endif



void run_tests() {

    std::cout << "------------------" << std::endl;
    std::cout << "Starting Tests" << std::endl;

    i8 weight[] = {3, 4, 6, 27};
    i8 activation[] = {4, 6, 4, 3};

    ui32 soft_mac_calc = software_mac(weight, activation, 4);

    std::cout << "Expected calculation for zedboard: " << soft_mac_calc <<  std::endl;

    #ifdef ZEDBOARD

    std::cout << "" << std::endl;
    std::cout << "Testing VDHL Mac" << std::endl;

    send_mac(weight, activation, 4);
    ui32 output = recieve_mac();

    std::cout << "Test 1 output from VDHL MAC operation: " << output << std::endl;
    std::cout << "" << std::endl;

    std::cout << "Updating weights and activation" << std::endl;

    i8 weight2[] =     {1, 2, 3, 4, 4, 4, 4};
    i8 activation2[] = {4, 20, 17, 20, 6, 19, 6};

    soft_mac_calc = software_mac(weight2, activation2, 7);
    send_mac(weight2, activation2, 7);
    output = recieve_mac();
    std::cout << "Expected calculation for zedboard: " << soft_mac_calc <<  std::endl;
    std::cout << "Test 2 output from VDHL MAC operation: " << output << std::endl;
    std::cout << "" << std::endl;

    std::cout << "Updating weights and activation" << std::endl;
    i8 weight3[] =     {-10, 4};
    i8 activation3[] = {-1, 2};
    soft_mac_calc = software_mac(weight3, activation3, 2);
    send_mac(weight3, activation3, 2);
    output = recieve_mac();
    std::cout << "Expected calculation for zedboard: " << soft_mac_calc <<  std::endl;
    std::cout << "Test 3 output from VDHL MAC operation: " << output << std::endl;
    std::cout << "" << std::endl;

    // calculating with team member number
    i8 teamNumber = 6;
    i8 weights4[] = {1};
    i8 activation4[] = {teamNumber};

    std::cout << "" << std::endl;
    std::cout << "Testing team member number" << std::endl;

    soft_mac_calc  = software_mac(weights4, activation4, 1);
    send_mac(weights4, activation4, 1);
    output = recieve_mac();

    std::cout << "Expected calculation for zedboard: " << soft_mac_calc <<  std::endl;
    std::cout << "Test 4 output from VDHL MAC operation: " << output << std::endl;
    std::cout << "" << std::endl;

    #ifdef debug
    std::cout << "Entered debug" << std::endl;
    while (1){
        send_mac(weights4, activation4, 1);
        output = recieve_mac();
    }
    std::cout << "exited debug" << std::endl;
    #endif

    #endif

    std:: cout << "End of Testing" << std::endl;
    std::cout << "------------------" << std::endl;
}