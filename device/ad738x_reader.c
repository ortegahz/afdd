/* ad738x_reader.c */
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <endian.h>

int main(void)
{
    int fd = open("/dev/ad738x", O_RDONLY);
    if (fd < 0)
    {
        perror("open");
        return 1;
    }

    struct pollfd p = {.fd = fd, .events = POLLIN};

    while (1)
    {
        if (poll(&p, 1, -1) <= 0)
            continue; /* 被信号打断就重来 */

        uint16_t frame[2];
        ssize_t n = read(fd, frame, sizeof(frame));
        if (n != sizeof(frame))
            continue; /* 理论上不会发生 */

        /* 若需要统一大小端，可用 be16toh()/le16toh() */
        uint16_t v = frame[0];
        uint16_t i = frame[1];

        /* 丢帧检测：示例里 lost>0 就把样本置 0xFFFF */
        if (v == 0xFFFF && i == 0xFFFF)
        {
            printf("frame lost!\n");
            continue;
        }

        printf("Vraw=%u  Iraw=%u\n", v, i);
    }
}