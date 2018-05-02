#include <errno.h>
#include <limits.h>
#include <float.h>
#include <stdlib.h>

int parse_int(const char *str, int *i) {
    errno = 0;
    char *endptr;
    long li = strtol(str, &endptr, 10);

    if (errno)
        return errno;
    if (endptr == str)
        return EINVAL;
    if (li > INT_MAX || li < INT_MIN)
        return ERANGE;

    *i = (int) li;
    return 0;
}

int parse_float(const char *str, float *f) {
    errno = 0;
    char *endptr;
    double d = strtod(str, &endptr);

    if (errno)
        return errno;
    if (endptr == str)
        return EINVAL;
    if (d > FLT_MAX || d < FLT_MIN)
        return ERANGE;

    *f = (float) d;
    return 0;
}
