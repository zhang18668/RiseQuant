<template>
  <el-card>
    <template #header><b>策略编写 — 表单提交训练任务</b></template>
    <el-form :model="form" label-width="200px">
      <el-divider content-position="left">数据范围</el-divider>
      <el-form-item label="TDX vipdoc 路径">
        <el-input v-model="form.tdx_path" />
      </el-form-item>
      <el-form-item label="日线起止">
        <el-date-picker v-model="form.data_range" type="daterange" value-format="YYYY-MM-DD" />
      </el-form-item>
      <el-form-item label="train_end / valid_end / test_end">
        <el-date-picker v-model="form.train_end" type="date" value-format="YYYY-MM-DD" placeholder="train_end" />
        <el-date-picker v-model="form.valid_end" type="date" value-format="YYYY-MM-DD" placeholder="valid_end" style="margin-left: 8px"/>
        <el-date-picker v-model="form.test_end" type="date" value-format="YYYY-MM-DD" placeholder="test_end" style="margin-left: 8px"/>
      </el-form-item>

      <el-divider content-position="left">事件 + 金标准</el-divider>
      <el-form-item label="max_gap (首板→二板)">
        <el-input-number v-model="form.max_gap" :min="5" :max="60" />
      </el-form-item>
      <el-form-item label="positive_window (距二板 N 日)">
        <el-input-number v-model="form.positive_window" :min="1" :max="10" />
      </el-form-item>
      <el-form-item label="金标准区间 [lo, hi]">
        <el-input-number v-model="form.golden_lo" :min="0.05" :max="0.5" :step="0.01" />
        ~
        <el-input-number v-model="form.golden_hi" :min="0.1" :max="1" :step="0.01" />
      </el-form-item>
      <el-form-item label="未来观察天数 (horizon)">
        <el-input-number v-model="form.golden_horizon" :min="10" :max="60" />
      </el-form-item>

      <el-divider content-position="left">聚类 + 模型</el-divider>
      <el-form-item label="cluster k 范围">
        <el-input-number v-model="form.k_low" :min="2" :max="10" />
        ~
        <el-input-number v-model="form.k_high" :min="2" :max="12" />
        <span style="margin-left: 16px; color: #888;">(silhouette 自选)</span>
      </el-form-item>
      <el-form-item label="或固定 k (置空=自动)">
        <el-input-number v-model="form.k_fixed" :min="0" :max="12" />
      </el-form-item>
      <el-form-item label="min_train_samples / cluster">
        <el-input-number v-model="form.min_train_samples_per_cluster" :min="20" :max="500" />
      </el-form-item>
      <el-form-item label="min_auc_test (usable 门槛)">
        <el-input-number v-model="form.min_auc_test" :min="0.5" :max="0.8" :step="0.01" />
      </el-form-item>
      <el-form-item label="LightGBM n_estimators">
        <el-input-number v-model="form.n_estimators" :min="50" :max="1000" :step="50" />
      </el-form-item>
      <el-form-item label="bundle 类型">
        <el-radio-group v-model="form.bundle_type">
          <el-radio-button label="both">两种都训</el-radio-button>
          <el-radio-button label="per_cluster">仅方案 A</el-radio-button>
          <el-radio-button label="single_with_cf">仅方案 B</el-radio-button>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="bundle 输出根目录">
        <el-input v-model="form.out" />
      </el-form-item>
      <el-form-item label="保存训练数据 (可追溯)">
        <el-switch v-model="form.save_training_data" />
      </el-form-item>

      <el-form-item>
        <el-button type="primary" size="large" :loading="loading" @click="submit">提交训练任务</el-button>
      </el-form-item>
    </el-form>
    <JobWatcher :job-id="jobId" />
  </el-card>
</template>

<script setup>
import { ref } from 'vue'
import { trainApi } from '../api.js'
import JobWatcher from '../components/JobWatcher.vue'

const form = ref({
  tdx_path: 'C:\\new_tdx\\vipdoc',
  data_range: ['2022-01-01', '2024-12-31'],
  train_end: '2023-12-31', valid_end: '2024-06-30', test_end: '2024-12-31',
  max_gap: 30, positive_window: 5,
  golden_lo: 0.15, golden_hi: 0.35, golden_horizon: 22,
  k_low: 3, k_high: 8, k_fixed: 0,
  min_train_samples_per_cluster: 80, min_auc_test: 0.55,
  n_estimators: 200, bundle_type: 'both',
  out: './models/pattern_cluster', save_training_data: false,
})
const jobId = ref('')
const loading = ref(false)

async function submit() {
  loading.value = true
  try {
    const payload = {
      start: form.value.data_range[0], end: form.value.data_range[1],
      train_end: form.value.train_end, valid_end: form.value.valid_end, test_end: form.value.test_end,
      tdx_path: form.value.tdx_path,
      golden_lo: form.value.golden_lo, golden_hi: form.value.golden_hi, golden_horizon: form.value.golden_horizon,
      max_gap: form.value.max_gap, positive_window: form.value.positive_window,
      k_range: [form.value.k_low, form.value.k_high],
      k_fixed: form.value.k_fixed > 0 ? form.value.k_fixed : null,
      bundle_type: form.value.bundle_type,
      min_train_samples_per_cluster: form.value.min_train_samples_per_cluster,
      min_auc_test: form.value.min_auc_test,
      n_estimators: form.value.n_estimators, out: form.value.out,
      save_training_data: form.value.save_training_data,
    }
    const { data } = await trainApi.submit(payload)
    jobId.value = data.id
  } finally { loading.value = false }
}
</script>
