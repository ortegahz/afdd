/* ---------------------------  main.c  --------------------------- */
#include "ai_model.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* 环形缓冲区里的全局数组（在 ai_model.c 中定义） */
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
                               const char *format)          /* "comma" 或 "newline" */
{
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        printf("Error: Cannot open %s for writing\n", filename);
        return;
    }
    for (size_t i = 0; i < length; ++i) {
        if (strcmp(format, "comma") == 0) {
            fprintf(fp, "%.6f%s", data[i], (i < length - 1) ? "," : "");
        } else {
            fprintf(fp, "%f\n", data[i]);
        }
    }
    fclose(fp);
    printf("Saved %zu samples to %s\n", length, filename);
}

/* --------------------- 2. 解析 BIN 文件 ------------------------- */
/* 把 interleaved(交替) 的 12-bit 原始采样分成 ch_a / ch_b */
static int parse_bin_file(const char *filename,
                          uint16_t **ch_a,
                          uint16_t **ch_b,
                          size_t *num_samples)         /* 返回实际“单通道”样本点数 */
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error: Cannot open %s\n", filename);
        return -1;
    }

    /* 文件大小 */
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

    /* 如果总数为奇数，舍掉最后一个 */
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

/* --------------------- 3. 读取标签（txt） ------------------------ */
static void load_labels(const char *label_file,
                        float *seq_state_arc,
                        size_t seq_len) {
    FILE *fp = fopen(label_file, "r");
    if (!fp) {
        printf("Warning: Cannot open %s\n", label_file);
        return;
    }
    int idx_s, idx_e;
    while (fscanf(fp, "%d %d", &idx_s, &idx_e) == 2) {
        if (idx_s < 0) idx_s = 0;
        if (idx_e > (int) seq_len) idx_e = (int) seq_len;
        for (int i = idx_s; i < idx_e; ++i)
            seq_state_arc[i] = 4096.0f;   /* STATE_INDICATE_VAL */
    }
    fclose(fp);
}

/* ----------------------------------------------------------------- */
int main(int argc, char *argv[]) {
    const char *input_file =
            "/home/manu/mnt/ST8000DM004-2U91/afdd/data/data_v33/data_sorted/串联碳化/正例-串联碳化-额定+1第三次_20k.bin";

    const char *output_dir = (argc >= 3) ? argv[2] : "/home/manu/tmp";

    /* ---------- 1. 解析 BIN ---------- */
    uint16_t *ch_a = NULL, *ch_b = NULL;
    size_t num_samples = 0;
    if (parse_bin_file(input_file, &ch_a, &ch_b, &num_samples) != 0) {
        printf("Error: parse bin failed\n");
        return 1;
    }
    printf("Loaded %zu samples, ARRAY_SIZE = %d\n", num_samples, ARRAY_SIZE);

    /* ---------- 2. 初始化 ---------- */
    DataRT data_rt;
    init_datart(&data_rt);
    reset_datart(&data_rt);
    if (init_aimodel() != 1) {
        free(ch_a);
        free(ch_b);
        return 1;
    }

    /* ---------- 3. 结果数组 ---------- */
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

    /* ---------- 4. 读取标签 ---------- */
    char label_file[512];
    strcpy(label_file, input_file);
    char *dot = strrchr(label_file, '.');
    if (dot) {
        strcpy(dot, ".txt");
        load_labels(label_file, seq_state_arc, num_samples);
    }

    /* ---------- 5. 主循环 ---------- */
    int alarm_cnt = 0;
    for (size_t i = 0; i < num_samples; ++i) {

        float cur_power = (float) ch_a[i];
        update_datart_v4(&data_rt, cur_power);      /* 放入环形缓冲区 */

        if (pipeline_v6(&data_rt) == 1) ++alarm_cnt;

        /* 5-1 保存 seq_power 当前值 */
        int seq_power_idx = (data_rt.seq_power_len - 1 + ARRAY_SIZE) % ARRAY_SIZE;
        full_seq_power[i] = seq_power[seq_power_idx];

        /* 5-2 把环形缓冲区最近 window 个点同步到完整序列 */
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

    /* ---------- 6. 保存结果 ---------- */
    char path[512];

    snprintf(path, sizeof(path), "%s/seq_power_cpp.txt", output_dir);
    save_array_to_file(path, full_seq_power, num_samples, "newline");

    snprintf(path, sizeof(path), "%s/seq_state_pred_classifier_cpp.txt", output_dir);
    save_array_to_file(path, full_seq_state_pred_classifier, num_samples, "newline");

    snprintf(path, sizeof(path), "%s/seq_state_pred_arc_cpp.txt", output_dir);
    save_array_to_file(path, full_seq_state_pred_arc, num_samples, "newline");

    snprintf(path, sizeof(path), "%s/seq_state_gt_normal_cpp.txt", output_dir);
    save_array_to_file(path, full_seq_state_gt_normal, num_samples, "newline");

    snprintf(path, sizeof(path), "%s/seq_state_pred_idle_cpp.txt", output_dir);
    save_array_to_file(path, full_seq_state_pred_idle, num_samples, "newline");

    snprintf(path, sizeof(path), "%s/seq_power_mean_cpp.txt", output_dir);
    save_array_to_file(path, full_seq_power_mean, num_samples, "newline");

    /* ----------- 追加保存预测峰值索引 ---------------------------- */
    snprintf(path, sizeof(path), "%s/info_pred_peaks_cpp.txt", output_dir);
    save_array_to_file(path,
                       info_pred_peaks,
                       data_rt.info_pred_peaks_len,   /* 正确的长度 */
                       "newline");

    /* 统计信息 */
    snprintf(path, sizeof(path), "%s/statistics_cpp.txt", output_dir);
    FILE *stat_fp = fopen(path, "w");
    if (stat_fp) {
        fprintf(stat_fp, "Input file: %s\n", input_file);
        fprintf(stat_fp, "Total samples: %zu\n", num_samples);
        fprintf(stat_fp, "ARRAY_SIZE: %d\n", ARRAY_SIZE);
        fprintf(stat_fp, "Total alarms: %d\n", alarm_cnt);
        fprintf(stat_fp, "Alarm rate: %.2f%%\n",
                100.0 * alarm_cnt / num_samples);
        fclose(stat_fp);
    }

    /* ---------- 7. 清理 ---------- */
    destroy_aimodel();
    free(ch_a);
    free(ch_b);
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