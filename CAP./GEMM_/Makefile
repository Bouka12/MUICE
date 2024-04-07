CC = gcc
CFLAGS = -Wall -Wextra -O3 -fopenmp
LDFLAGS = -lm

# Source files
SRCS = gemm_driver.c my_gemm.c

# Object files
OBJS = $(SRCS:.c=.o)

# Executable name
EXEC = gemm_executable

all: $(EXEC)

$(EXEC): $(OBJS)
    $(CC) $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.c
    $(CC) $(CFLAGS) -c $< -o $@

clean:
    rm -f $(EXEC) $(OBJS)
