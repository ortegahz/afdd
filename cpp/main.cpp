/* ---------------------------  main.c  --------------------------- */
#include "ai_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* 若使用全局数组，则再包含 test_samples.h  */
#ifdef USE_GLOBAL_DATA
#include "test_samples.h"
#endif

/* ---------------- 环形缓冲区里的全局数组（ai_model.c 提供） ---- */
extern float seq_power[ARRAY_SIZE];
extern float seq_state_pred_idle[ARRAY_SIZE];
extern float seq_state_pred_classifier[ARRAY_SIZE];
extern float seq_state_pred_arc[ARRAY_SIZE];
extern float seq_state_gt_normal[ARRAY_SIZE];
extern float seq_power_mean[ARRAY_SIZE];
extern float info_pred_peaks[ARRAY_SIZE];

/* --------------------- 1. 工具函数 ------------------------------ */
static void save_array_to_file(const char *filename,
                               const float *data,
                               size_t length,
                               const char *format)          /* "comma" or "newline" */
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open %s for writing\n", filename);
        return;
    }
    for (size_t i = 0; i < length; ++i) {
        if (strcmp(format, "comma") == 0)
            fprintf(fp, "%.6f%s", data[i], (i < length - 1) ? "," : "");
        else
            fprintf(fp, "%f\n", data[i]);
    }
    fclose(fp);
    printf("Saved %zu samples -> %s\n", length, filename);
}

/* --------------------- 2. 解析 BIN 文件 ------------------------- */
#ifndef USE_GLOBAL_DATA      /* 只有 bin-模式才需要这些函数 */

static int parse_bin_file(const char *filename,
                          uint16_t **ch_a,
                          uint16_t **ch_b,
                          size_t *num_samples) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open %s\n", filename);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    size_t total_u16 = (size_t) file_size / sizeof(uint16_t);
    if (total_u16 < 2) {
        fclose(fp);
        return -1;
    }

    uint16_t *raw = (uint16_t *) malloc(total_u16 * sizeof(uint16_t));
    if (!raw) {
        fclose(fp);
        return -1;
    }

    if (fread(raw, sizeof(uint16_t), total_u16, fp) != total_u16) {
        printf("Error: read file\n");
        free(raw);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    /* 12-bit 掩码 */
    for (size_t i = 0; i < total_u16; ++i) raw[i] &= 0x0FFF;

    /* 确保总数为偶数 */
    if (total_u16 & 1) --total_u16;

    *num_samples = total_u16 / 2;
    *ch_a = (uint16_t *) malloc(*num_samples * sizeof(uint16_t));
    *ch_b = (uint16_t *) malloc(*num_samples * sizeof(uint16_t));
    if (!*ch_a || !*ch_b) {
        free(raw);
        return -1;
    }

    for (size_t i = 0; i < *num_samples; ++i) {
        (*ch_a)[i] = raw[2 * i];
        (*ch_b)[i] = raw[2 * i + 1];
    }
    free(raw);
    return 0;
}

#endif /* USE_GLOBAL_DATA */

/* --------------------- 3. 读取标签（txt） ------------------------ */
#ifndef USE_GLOBAL_DATA      /* 读取标签只与 bin 文件同名时才生效 */

static void load_labels(const char *label_file,
                        float *seq_state_arc,
                        size_t seq_len) {
    FILE *fp = fopen(label_file, "r");
    if (!fp) {
        printf("Warning: label %s not found\n", label_file);
        return;
    }

    int idx_s, idx_e;
    while (fscanf(fp, "%d %d", &idx_s, &idx_e) == 2) {
        if (idx_s < 0) idx_s = 0;
        if (idx_e > (int) seq_len) idx_e = (int) seq_len;
        for (int i = idx_s; i < idx_e; ++i)
            seq_state_arc[i] = 4096.0f;    /* STATE_INDICATE_VAL */
    }
    fclose(fp);
}

#endif /* USE_GLOBAL_DATA */

/* ----------------------------------------------------------------- */
int main(int argc, char *argv[]) {
#ifndef USE_GLOBAL_DATA
    /* =============== 1. 准备 bin 输入文件名 =============== */
    const char *input_file =
            "/home/manu/mnt/ST8000DM004-2U91/afdd/data/"
            "data_v34/data_sorted/负载抑制/正例-负载抑制性试验2（定频空调）"
            "+1000W阻性负载第三次_20k.bin";
#endif

    const char *output_dir = (argc >= 2) ? argv[1] : "/home/manu/tmp";

    /* ---------------- 2. 装载样本 -------------------------- */
    size_t num_samples = 0;
    uint16_t *ch_a = NULL, *ch_b = NULL;   /* 仅 bin 模式使用 */

#ifdef USE_GLOBAL_DATA
    /* ---- 从全局数组读取 ---- */
    num_samples = g_test_samples_len;
    printf("[ARRAY]  Loaded %zu samples from global array, ARRAY_SIZE = %d\n",
           num_samples, ARRAY_SIZE);
#else
    /* ---- 从 bin 文件读取 ---- */
    if (parse_bin_file(input_file, &ch_a, &ch_b, &num_samples) != 0) {
        printf("Error: parse_bin_file failed\n");
        return 1;
    }
    printf("[BIN]    Loaded %zu samples, ARRAY_SIZE = %d\n",
           num_samples, ARRAY_SIZE);
#endif

    /* ---------------- 3. 初始化 DataRT / 模型 --------------- */
    DataRT data_rt;
    init_datart(&data_rt);
    reset_datart(&data_rt);

    if (init_aimodel() != 1)
        return 1;

    /* ---------------- 4. 结果数组分配 ----------------------- */
    float *full_seq_power = (float *) calloc(num_samples, sizeof(float));
    float *full_seq_state_pred_classifier = (float *) calloc(num_samples, sizeof(float));
    float *full_seq_state_pred_arc = (float *) calloc(num_samples, sizeof(float));
    float *full_seq_state_gt_normal = (float *) calloc(num_samples, sizeof(float));
    float *full_seq_state_pred_idle = (float *) calloc(num_samples, sizeof(float));
    float *full_seq_power_mean = (float *) calloc(num_samples, sizeof(float));
    float *seq_state_arc = (float *) calloc(num_samples, sizeof(float));

    if (!full_seq_power || !full_seq_state_pred_classifier ||
        !full_seq_state_pred_arc || !full_seq_state_gt_normal ||
        !full_seq_state_pred_idle || !full_seq_power_mean || !seq_state_arc) {
        printf("Error: malloc\n");
        return 1;
    }

#ifndef USE_GLOBAL_DATA
    /* ---------------- 5. 读取标签（仅 bin 模式）------------- */
    char label_file[512];
    strcpy(label_file, input_file);
    char *dot = strrchr(label_file, '.');
    if (dot) {
        strcpy(dot, ".txt");
        load_labels(label_file, seq_state_arc, num_samples);
    }
#endif

    /* ---------------- 6. 主循环 ----------------------------- */
    int alarm_cnt = 0;
    for (size_t i = 0; i < num_samples; ++i) {
        /* 取得当前功率样本 ------------------ */
#ifdef USE_GLOBAL_DATA
        float cur_power = g_test_samples[i];
#else
        float cur_power = (float) ch_a[i];
#endif
        update_datart_v4(&data_rt, cur_power);

        if (pipeline_v6(&data_rt) == 1) ++alarm_cnt;

        /* 6-1 保存 seq_power 当前值 --------------------------- */
        int seq_power_idx = (data_rt.seq_power_len - 1 + ARRAY_SIZE) % ARRAY_SIZE;
        full_seq_power[i] = seq_power[seq_power_idx];

        /* 6-2 把环形缓冲区最近 window 个点同步到完整序列 ------- */
        size_t window = (data_rt.seq_len < ARRAY_SIZE) ? data_rt.seq_len : ARRAY_SIZE;
        size_t global_start = (i + 1) - window;

        for (size_t j = 0; j < window; ++j) {
            size_t sample_idx = global_start + j;
            size_t buf_idx = sample_idx % ARRAY_SIZE;

            full_seq_state_pred_classifier[sample_idx] = seq_state_pred_classifier[buf_idx];
            full_seq_state_pred_arc[sample_idx] = seq_state_pred_arc[buf_idx];
            full_seq_state_gt_normal[sample_idx] = seq_state_gt_normal[buf_idx];
            full_seq_state_pred_idle[sample_idx] = seq_state_pred_idle[buf_idx];
            full_seq_power_mean[sample_idx] = seq_power_mean[buf_idx];
        }

        if ((i + 1) % 100000 == 0)
            printf("Progress: %zu/%zu (%.1f%%), alarms=%d\n",
                   i + 1, num_samples,
                   100.0 * (i + 1) / num_samples, alarm_cnt);
    }

    printf("Processing complete. alarms=%d (%.2f%%)\n",
           alarm_cnt, 100.0 * alarm_cnt / num_samples);

    /* ---------------- 7. 保存结果 --------------------------- */
    char path[512];

#define SAVE(name, arr) \
    snprintf(path, sizeof(path), "%s/" name "_cpp.txt", output_dir); \
    save_array_to_file(path, arr, num_samples, "newline");

    SAVE("seq_power", full_seq_power);
    SAVE("seq_state_pred_classifier", full_seq_state_pred_classifier);
    SAVE("seq_state_pred_arc", full_seq_state_pred_arc);
    SAVE("seq_state_gt_normal", full_seq_state_gt_normal);
    SAVE("seq_state_pred_idle", full_seq_state_pred_idle);
    SAVE("seq_power_mean", full_seq_power_mean);

    /* ----- 预测峰值索引单独保存 (长度不同) ------------------ */
    snprintf(path, sizeof(path), "%s/info_pred_peaks_cpp.txt", output_dir);
    save_array_to_file(path, info_pred_peaks, data_rt.info_pred_peaks_len, "newline");

    /* 统计信息 */
    snprintf(path, sizeof(path), "%s/statistics_cpp.txt", output_dir);
    FILE *stat_fp = fopen(path, "w");
    if (stat_fp) {
#ifdef USE_GLOBAL_DATA
        fprintf(stat_fp, "Input: g_test_samples[] (compiled-in)\n");
#else
        fprintf(stat_fp, "Input file: %s\n", input_file);
#endif
        fprintf(stat_fp, "Total samples: %zu\n", num_samples);
        fprintf(stat_fp, "ARRAY_SIZE: %d\n", ARRAY_SIZE);
        fprintf(stat_fp, "Total alarms: %d\n", alarm_cnt);
        fprintf(stat_fp, "Alarm rate: %.2f%%\n",
                100.0 * alarm_cnt / num_samples);
        fclose(stat_fp);
    }

    /* ---------------- 8. 资源释放 --------------------------- */
    destroy_aimodel();

#ifndef USE_GLOBAL_DATA
    free(ch_a);
    free(ch_b);
#endif
    free(seq_state_arc);
    free(full_seq_power);
    free(full_seq_state_pred_classifier);
    free(full_seq_state_pred_arc);
    free(full_seq_state_gt_normal);
    free(full_seq_state_pred_idle);
    free(full_seq_power_mean);

    printf("Done!\n");
    return 0;
}