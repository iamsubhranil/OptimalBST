#include <stdio.h>   // printf, scanf
#include <stdlib.h>  // malloc, calloc, random, srand
#include <string.h>  // strcmp
#include <time.h>    // time

// matrix allocation functions
// even though we are allocating a RxC matrix,
// we allocate a single array, to reduce
// pointer indirection.
#define mat_create(x)                          \
    x *mat_create_##x(size_t r, size_t c) {    \
        x *m = (x *)malloc(sizeof(x) * r * c); \
        return m;                              \
    }

mat_create(double);
mat_create(size_t);

// placeholder structure to return both e and root
// together back to the calling function
typedef struct {
    double *e;
    size_t e_num_cols;
    size_t *root;
    size_t root_num_cols;
} Result;

// this macro performs the necessary row major
// transformation of given indices x and y to an
// array m. to calculate the index, a variable
// named m_num_cols must be declared which will
// denote the number of columns in a single row
// of m
#define at(m, x, y) m[((x) * (m##_num_cols)) + (y)]

// debug macros
#define dbgatd(m, x, y)                                                       \
    printf("%d: " #m "[%lu][%lu](%lu * %lu + %lu): %lf\n", __LINE__, x, y, x, \
           m##_num_cols, y, at(m, x, y));

#define dbgd(x) printf("%d: " #x ": %lf\n", __LINE__, (x));
#define dbgs(x) printf("%d: " #x ": %lu\n", __LINE__, (x));

// calculates the optimal bst for a given set of key and
// dummy probabilities.
// Pi contains the probability for key   Ki (K0 is 0.0, since there must
//                                          always be a Di-1 for each Ki)
// Qi contains the probability for dummy Di (hence, size of Q is n + 1)
// n  contains the number of keys in the tree
//
// Returns the optimal root and expected cost arrays back
// to the caller.
Result optimal_bst(double *p, double *q, size_t n) {
    // w[i,j] denotes the total cost for the subtree
    // with keys [Ki, Kj] with dummies [Di-1, Dj]
    // w[i,j] = sum(Pl, l=i to j) + sum(Ql, l=i-1 to j)
    const size_t w_num_cols = n + 1;
    double *w = mat_create_double(n + 2, n + 1);
    // e[i,j] denotes the expected cost for the subtree
    // with keys in [Ki, Kj] with root Kr
    // e[i,j] = Pr + e[i, r-1] + w[i, r-1] + e[r+1, j] + w[r+1, j]
    //        = Pr + e[i, r-1] + e[r+1, j] + w[i, j]
    //      w[i,j] is added because each subtree of Kr is increased in
    //      height by 1. So their actual cost, which is basically
    //      (height * P) is increased by P for all of the vertices.
    const size_t e_num_cols = n + 1;
    double *e = mat_create_double(n + 2, n + 1);

    // root[i,j] denotes the optimal root for the BST with keys
    // in [Ki, Kj]
    const size_t root_num_cols = n + 1;
    size_t *root = mat_create_size_t(n + 1, n + 1);

    // initially, e[i, i-1] and w[i, i-1] will be Qi-1
    // as the expected cost and actual cost will be the
    // cost of the dummy if the tree contains no keys.
    for (size_t i = 1; i <= n + 1; i++) {
        at(e, i, i - 1) = at(w, i, i - 1) = q[i - 1];
    }

    // now, evaluate the optimal tree by its length
    // i.e., first evaluate optimal subtrees of length 1,
    // then 2, and so on
    for (size_t l = 1; l <= n; l++) {
        // at each iteration, we will consider l consecutive
        // keys, so the inner traversal must stop at n - l + 1
        for (size_t i = 1; i <= n - l + 1; i++) {
            // find the terminal index if we start at Ki
            size_t j = i + l - 1;
            // consider the expected cost initially to be infinite
            at(e, i, j) = __DBL_MAX__;
            // find the total cost, which is nothing but the cost
            // of the subtree upto (j - 1), plus the cost of Kj and Dj
            at(w, i, j) = at(w, i, j - 1) + p[j] + q[j];
            // now, consider each of the possible keys in [Ki, Kj]
            // to be the root, and find out the optimal one
            for (size_t r = i; r <= j; r++) {
                // calculate the cost if Kr was the root
                double temp_cost =
                    //                                   extra cost
                    //                                 due to increase
                    //  cost(left)      cost(right)      in height
                    at(e, i, r - 1) + at(e, r + 1, j) + at(w, i, j);
                // check if that's lesser than what we already have
                if (temp_cost < at(e, i, j)) {
                    // it is lesser, so it is the new optimal cost,
                    // and r is the new optimal root
                    at(e, i, j) = temp_cost;
                    at(root, i, j) = r;
                }
            }
        }
    }
    free(w);
    return (Result){e, e_num_cols, root, root_num_cols};
}

// prints the necessary indentations to denote
// a level
//
// print_info : contains flags as to whether or not to print | for
//              a certain level
// level      : current indentation level
void print_level(size_t *print_info, size_t level) {
    for (size_t i = 0; i < level; i++) {
        if (print_info[i])
            printf("|  ");
        else
            printf("   ");
    }
}

// prints the subtree with root r in the range [start, end]
//
// r          : root of current subtree
// root       : array containing the optimal roots
// start, end : the key range which has r as its optimal root
// n          : number of keys in root
//
// level      : current indentation level
// print_info : contains flags as to whether or not to print | for
//              a certain level
// isLeft     : denotes whether r is a left child of its parent
void print_tree_rec(size_t r, size_t *root, size_t start, size_t end, size_t n,
                    size_t level, size_t *print_info, int isLeft) {
    // at left, keys are start..r-1
    // at right, keys are r+1..end

    const size_t root_num_cols = n + 1;

    // if r is 0, it is time to print a dummy node,
    // whose index is stored in start
    if (r == 0) {
        print_level(print_info, level);
        // print dummy
        printf("|- D%lu\n", start);
        return;
    }

    // if r is the right child of its parent, and
    // it is not the root level, print its level
    // while printing the left subtree of r
    print_info[level] = !isLeft && level > 0;

    // check whether the left subtree contains at least one child
    if (r != start) {
        // find the root of subtree in [start, r-1]
        size_t lr = at(root, start, r - 1);
        // print left subtree
        print_tree_rec(lr, root, start, r - 1, n, level + 1, print_info, 1);
    } else {
        // the tree does not contain any children, so
        // print a dummy node
        print_tree_rec(0, root, r - 1, 0, 0, level + 1, print_info, 1);
    }

    // print the level of r
    print_level(print_info, level);
    // print r, the root of current subtree
    printf("|- K%lu\n", r);

    // if r is the left child of its parent, print
    // its level while printing the right subtree of r
    print_info[level] = isLeft;

    // check whether the right subtree contains at least one child
    if (r != end) {
        // find the root of subtree in [r+1, end]
        size_t rr = at(root, r + 1, end);
        // print right subtree
        print_tree_rec(rr, root, r + 1, end, n, level + 1, print_info, 0);
    } else {
        // the right subtree does not contain any children, so
        // print a dummy node
        print_tree_rec(0, root, r, 0, 0, level + 1, print_info, 0);
    }

    // when done printing the children, reset
    // its flag so that its level is not printed
    // anymore
    print_info[level] = 0;
}

// prints the tree with optimal roots stored in
// root, with n keys
void print_tree(size_t *root, size_t n) {
    const size_t root_num_cols = n + 1;
    // root of the whole tree at root[1, n]
    size_t r = at(root, 1, n);
    size_t *print_info = (size_t *)calloc(sizeof(size_t) * n, 1);
    print_tree_rec(r, root, 1, n, n, 0, print_info, 0);
    free(print_info);
}

// driver functions

typedef int (*print_fn)(const char *args, ...);
typedef double (*scan_fn)(FILE *f);

double scan_fp(FILE *f) {
    double d;
    fscanf(f, "%lf", &d);
    return d;
}

double scan_random(FILE *f) {
    (void)f;
    return (double)random() / RAND_MAX;
}

int no_print(const char *args, ...) {
    (void)args;
    return 0;
}

void usage(char *argv[]) {
    printf("Usage:\n");
    printf("1. Read from the standard input\n");
    printf("    %s\n", argv[0]);
    printf("2. Load the values from a file\n");
    printf("    %s -f <file_name>\n", argv[0]);
    printf("   First line of the file should\n");
    printf("   contain number of keys (N),\n");
    printf("   followed by N key probabilities\n");
    printf("   and N + 1 dummy probabilities.\n");
    printf("3. Generate random inputs\n");
    printf("    %s -r <number_of_keys>\n", argv[0]);
    printf("4. Generate random inputs and show the result\n");
    printf("    %s -rs <number_of_keys>\n", argv[0]);
}

int main(int argc, char *argv[]) {
    FILE *f = stdin;
    int state = 0;  // 0 -> stdin, 1 -> file, 2 -> random
    long num = -1;
    print_fn pf = no_print;
    scan_fn sf = scan_fp;
    int print_in_random = 0;
    if (argc > 1) {
        if (strcmp(argv[1], "-f") == 0) {
            if (argc < 3) {
                printf("[Error] Expected filename after '-f'!\n");
                usage(argv);
                return 1;
            }
            f = fopen(argv[2], "r");
            if (f == NULL) {
                printf("[Error] Unable to open '%s'!\n", argv[2]);
                return 2;
            }
            state = 1;
        } else if (strcmp(argv[1], "-r") == 0 ||
                   (print_in_random = 1, strcmp(argv[1], "-rs")) == 0) {
            if (argc < 3) {
                printf("[Error] Expected key count after '-r'!\n");
                usage(argv);
                return 3;
            }
            char *end = NULL;
            num = strtol(argv[2], &end, 10);
            if (*end != 0) {
                printf("[Error] Invalid key count '%s'!\n", argv[2]);
                return 4;
            }
            srand(time(NULL));
            state = 2;
        } else {
            printf("Invalid argument '%s'!\n", argv[1]);
            usage(argv);
            return 5;
        }
    }

    if (state == 0) {
        pf = printf;
    } else if (state == 2) {
        sf = scan_random;
    }

    if (num == -1) {
        pf("Enter number of keys : ");
        fscanf(f, "%ld", &num);
    }

    if (num < 1) {
        printf("[Error] Number of keys must be > 0 (given '%ld')!", num);
        return 6;
    }

    double *p = (double *)malloc(sizeof(double) * (num + 1));
    double *q = (double *)malloc(sizeof(double) * (num + 1));

    pf("Enter %ld key probabilities : ", num);
    p[0] = 0.0;
    for (long i = 1; i < num + 1; i++) {
        p[i] = sf(f);
    }

    pf("Enter %ld dummy probabilities : ", num + 1);
    for (long i = 0; i < num + 1; i++) {
        q[i] = sf(f);
    }

    Result r = optimal_bst(p, q, num);

    if (state != 2 || print_in_random) {
        printf("Expected costs : \n");
        for (long i = 1; i <= num; i++) {
            for (int k = 1; k < i; k++) {
                printf("     ");
            }
            for (long j = i; j <= num; j++) {
                printf("%4.2lf ", at(r.e, i, j));
            }
            printf("\n");
        }

        printf("Roots : \n");
        for (long i = 1; i <= num; i++) {
            for (int k = 1; k < i; k++) {
                printf("   ");
            }
            for (long j = i; j <= num; j++) {
                printf("%2lu ", at(r.root, i, j));
            }
            printf("\n");
        }

        printf("Tree : \n");
        print_tree(r.root, num);
    }

    free(p);
    free(q);
    free(r.e);
    free(r.root);
}
