#include <iostream>
#include "Types.h"
#include "tests.h"


#ifdef ZEDBOARD
int main() {
      std::cout << "Running on the ZEDBoard..." << std::endl;
      run_tests();
      return 0;
}
#else
int main() {
      std::cout << "Running on the Lab computer..." << std::endl;
      run_tests();
      return 0;
}
#endif
