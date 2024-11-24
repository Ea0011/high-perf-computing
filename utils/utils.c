#include <time.h>
#include <stdio.h>
#include <stdlib.h>

// Allocates aligned memory by respecting aligned_alloc rules
void* aligned_allocate_buffer(size_t alignment, size_t type_size, size_t element_count) {
    size_t required_size = element_count * type_size;
    size_t allocation_size = (required_size + alignment - 1) & ~(alignment - 1);

    void* ptr = aligned_alloc(alignment, allocation_size);
    if (ptr == NULL) {
        perror("Memory allocation failed");
        return NULL;
    }

    return ptr;
}