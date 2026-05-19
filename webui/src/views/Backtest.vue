<template>
  <el-card>
    <template #header><b>回测 — 选 bundle + 参数 + 提交</b></template>
    <el-form :model="form" label-width="180px">
      <el-form-item label="bundle 路径">
        <el-select v-model="form.bundle_dir" filterable allow-create style="width: 100%" placeholder="选一个或粘贴绝对路径">
          <el-option v-for="b in bundles" :key="b.bundle_dir" :label="b.name" :value="b.bundle_dir">
            <span>{{ b.name }}</span>
            <span style="float: right; color: #999;">{{ b.meta && b.meta.n_clusters }} clusters</span>
          </el-option>
        </el-select>
      </el-form-item>
      <el-form-item label="bundle 类型">
        <el-radio-group v-model="form.bundle_type">
          <el-radio-button label="per_cluster">方案 A</el-radio-button>
          <el-radio-button label="single_with_cf">方案 B</el-radio-button>
        </el-radio-group>
      </el-form-item>
      <el-form-item label="回测起止日期">
        <el-date-picker v-model="form.range" type="daterange" value-format="YYYY-MM-DD" />
      </el-form-item>
      <el-form-item label="TDX vipdoc 路径">
        <el-input v-model="form.tdx_path" />
      </el-form-item>
      <el-form-item label="topk">
        <el-input-number v-model="form.topk" :min="1" :max="20" />
      </el-form-item>
      <el-form-item label="持仓天数 (sell_n)">
        <el-input-number v-model="form.sell_n" :min="1" :max="30" />
      </el-form-item>
      <el-form-item label="min_score 阈值">
        <el-input-number v-model="form.min_score" :min="0.1" :max="0.95" :step="0.05" />
      </el-form-item>
      <el-form-item label="滑点">
        <el-input-number v-model="form.slippage" :min="0" :max="0.02" :step="0.0005" />
      </el-form-item>
      <el-form-item label="冷静期 (止损后禁买 N 日)">
        <el-input-number v-model="form.cooldown" :min="0" :max="30" />
      </el-form-item>
      <el-form-item label="限定 cluster (逗号分隔)">
        <el-input v-model="form.clusters" placeholder="留空 = 全部" />
      </el-form-item>
      <el-form-item label="初始资金">
        <el-input-number v-model="form.initial_cash" :min="1000000" :max="100000000" :step="1000000" />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" size="large" :loading="loading" @click="submit">提交回测</el-button>
        <el-button @click="loadBundles">刷新 bundle 列表</el-button>
      </el-form-item>
    </el-form>
    <JobWatcher :job-id="jobId" />
  </el-card>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { backtestApi, bundleApi } from '../api.js'
import JobWatcher from '../components/JobWatcher.vue'

const form = ref({
  bundle_dir: '', bundle_type: 'per_cluster', range: ['2024-01-01', '2024-12-31'],
  tdx_path: 'C:\\new_tdx\\vipdoc', topk: 5, sell_n: 8, min_score: 0.6,
  slippage: 0.003, cooldown: 5, clusters: '', initial_cash: 10_000_000,
})
const bundles = ref([])
const jobId = ref('')
const loading = ref(false)

async function loadBundles() {
  const { data } = await bundleApi.list('./models/pattern_cluster')
  bundles.value = data
  if (!form.value.bundle_dir && data.length) form.value.bundle_dir = data[0].bundle_dir
}
onMounted(loadBundles)

async function submit() {
  loading.value = true
  try {
    const payload = {
      bundle_dir: form.value.bundle_dir,
      bundle_type: form.value.bundle_type,
      bt_start: form.value.range[0], bt_end: form.value.range[1],
      tdx_path: form.value.tdx_path,
      initial_cash: form.value.initial_cash,
      topk: form.value.topk, sell_n: form.value.sell_n,
      slippage: form.value.slippage, min_score: form.value.min_score,
      cooldown: form.value.cooldown,
      clusters: form.value.clusters
        ? form.value.clusters.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n))
        : null,
    }
    const { data } = await backtestApi.submit(payload)
    jobId.value = data.id
  } finally { loading.value = false }
}
</script>
