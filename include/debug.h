#ifndef DEBUG_H
# define DEBUG_H
# ifdef DEBUG
#  define DPRINT(...) fprintf(stderr, __VA_ARGS__);
#  define DPRINTL(...) \
   fprintf(stderr, "Line %d in %s:\t", __LINE__, __FILE__ );\
   fprintf(stderr, __VA_ARGS__);
#  define DPRINTFUNC fprintf(stderr, "\n%s in %s:\n", __func__, __FILE__);
# else
#  define NDEBUG
#  define DPRINT(...) do {} while (0)
#  define DPRINTL(...) do {} while (0)
#  define DPRINTFUNC(...) do {} while (0)
# endif
#include <stdarg.h> /* for the ellepsis (...) in DPRINT */
#include <assert.h> /* asserts can be disabled by defining a macro NDEBUG, removes automatically 
                       the asserts when compiling non-debug code */
#endif
