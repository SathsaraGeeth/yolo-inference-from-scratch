CC = gcc
CFLAGS = -Wall -O3 -Iinclude
LDFLAGS = -ljpeg

SRC = src/image.c src/inference.c src/utils.c src/tensor.c
OBJ = $(SRC:.c=.o)
TARGET = libllinf.a

all: $(TARGET)

$(TARGET): $(OBJ)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)