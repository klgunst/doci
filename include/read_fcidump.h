#ifndef READ_FCIDUMP_H
# define READ_FCIDUMP_H

void read_inputfile(char*, char*, char*, int*, double*, int*, int*, int*);
void read_fcidump(char*, int*, int*, int*, double*, double**, double**);
int strcmp_ign_ws(const char*, const char*);

#endif
