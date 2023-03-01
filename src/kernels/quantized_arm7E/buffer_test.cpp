#include <iostream>
#include "params.h"

int main(int, char**)
{
    using ScalarT = uint8_t;

    std::cerr << "Allocating 100 bytes.\n";
    small::Buffer<ScalarT> b(100);

    std::cerr << "Performing basic API tests.\n";
    if (b.size() != 100)  std::cerr << "FAILED size() test.\n";
    ScalarT *ptr = b.data();
    b[20] = 99;
    if (*(ptr + 20) != 99)  std::cerr << "FAILED data() test.\n";
    if (b[20] != 99)      std::cerr << "FAILED operator[] test.\n";

    try
    {
        for (size_t ix = 0; ix < 5; ++ix)
        {
            std::cerr << "Allocating 50000 bytes.\n";
            small::Buffer<ScalarT> buf(50000);
            std::cerr << "small::detail::buf_offset = "
                      << small::detail::buf_offset << std::endl;
            // Buffer destruction happens here but no freeing.
        }
    }
    catch (std::bad_alloc &e)
    {
        std::cerr << "Caught std::bad_alloc." << std::endl;
    }
    return 0;
}
