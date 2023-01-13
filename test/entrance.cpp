
#include "lms_source.hpp"

// -DMNN_BUILD_QUANTOOLS=ON

int main(int argc, char **argv) {
  if (argc != 2) {
    lms::test_fixed_model();
  } else {
    lms::test_specified_model(argv[1]);

  }
  return 0;
}
