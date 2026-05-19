<template>
  <div>
    <el-card>
      <template #header><b>参数搜索 (Grid / Random)</b></template>
      <el-form :model="searchForm" label-width="160px">
        <el-form-item label="bundle">
          <el-select v-model="searchForm.bundle_dir" filterable style="width: 100%">
            <el-option v-for="b in bundles" :key="b.bundle_dir" :label="b.name" :value="b.bundle_dir" />
          </el-select>
        </el-form-item>
        <el-form-item label="bundle 类型">
          <el-radio-group v-model="searchForm.bundle_type">
            <el-radio-button label="per_cluster">A</el-radio-button>
            <el-radio-button label="single_with_cf">B</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="回测起止">
          <el-date-picker v-model="searchForm.range" type="daterange" value-format="YYYY-MM-DD" />
        </el-form-item>
        <el-form-item label="min_score 候选 (逗号)">
          <el-input v-model="searchForm.min_scores" placeholder="0.5,0.6,0.7" />
        </el-form-item>
        <el-form-item label="topk 候选 (逗号)">
          <el-input v-model="searchForm.topks" placeholder="3,5,8" />
        </el-form-item>
        <el-form-item label="cooldown 候选 (逗号)">
          <el-input v-model="searchForm.cooldowns" placeholder="3,5,8" />
        </el-form-item>
        <el-form-item label="目标指标">
          <el-radio-group v-model="searchForm.objective">
            <el-radio-button label="sharpe_ratio">Sharpe</el-radio-button>
            <el-radio-button label="annual_return">年化</el-radio-button>
            <el-radio-button label="win_rate">胜率</el-radio-button>
          </el-radio-group>
        </el-form-item>
        <el-form-item label="n_trials (置空 = 全跑)">
          <el-input-number v-model="searchForm.n_trials" :min="0" :max="200" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="loadingSearch" @click="submitSearch">提交搜索</el-button>
        </el-form-item>
      </el-form>
      <JobWatcher :job-id="searchJobId" />
    </el-card>

    <el-card style="margin-top: 16px;">
      <template #header><b>敏感性分析 (固定其他, 扫单参数)</b></template>
      <el-form :model="sensForm" label-width="160px">
        <el-form-item label="bundle">
          <el-select v-model="sensForm.bundle_dir" filterable style="width: 100%">
            <el-option v-for="b in bundles" :key="b.bundle_dir" :label="b.name" :value="b.bundle_dir" />
          </el-select>
        </el-form-item>
        <el-form-item label="回测起止">
          <el-date-picker v-model="sensForm.range" type="daterange" value-format="YYYY-MM-DD" />
        </el-form-item>
        <el-form-item label="变量">
          <el-select v-model="sensForm.param" style="width: 200px">
            <el-option label="min_score" value="min_score" />
            <el-option label="topk" value="topk" />
            <el-option label="cooldown" value="cooldown" />
            <el-option label="sell_n" value="sell_n" />
          </el-select>
        </el-form-item>
        <el-form-item label="扫值 (逗号)">
          <el-input v-model="sensForm.values" placeholder="0.5,0.55,0.6,0.65,0.7,0.75" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="loadingSens" @click="submitSens">提交敏感性</el-button>
        </el-form-item>
      </el-form>
      <JobWatcher :job-id="sensJobId" />
    </el-card>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { bundleApi, optimizeApi } from '../api.js'
import JobWatcher from '../components/JobWatcher.vue'

const bundles = ref([])
const searchForm = ref({
  bundle_dir: '', bundle_type: 'per_cluster', range: ['2024-01-01', '2024-12-31'],
  min_scores: '0.55,0.6,0.65,0.7', topks: '3,5', cooldowns: '5',
  objective: 'sharpe_ratio', n_trials: 0,
})
const sensForm = ref({
  bundle_dir: '', range: ['2024-01-01', '2024-12-31'],
  param: 'min_score', values: '0.5,0.55,0.6,0.65,0.7,0.75',
})
const searchJobId = ref(''), sensJobId = ref('')
const loadingSearch = ref(false), loadingSens = ref(false)

function _parseList(s, parser=parseFloat) {
  return s.split(',').map(x => parser(x.trim())).filter(x => !isNaN(x))
}

async function loadBundles() {
  const { data } = await bundleApi.list('./models/pattern_cluster')
  bundles.value = data
  if (data.length) {
    if (!searchForm.value.bundle_dir) searchForm.value.bundle_dir = data[0].bundle_dir
    if (!sensForm.value.bundle_dir) sensForm.value.bundle_dir = data[0].bundle_dir
  }
}
onMounted(loadBundles)

async function submitSearch() {
  loadingSearch.value = true
  try {
    const grid = {}
    const ms = _parseList(searchForm.value.min_scores); if (ms.length) grid.min_score = ms
    const tk = _parseList(searchForm.value.topks, parseInt); if (tk.length) grid.topk = tk
    const cd = _parseList(searchForm.value.cooldowns, parseInt); if (cd.length) grid.cooldown = cd
    const { data } = await optimizeApi.search({
      bundle_dir: searchForm.value.bundle_dir,
      bundle_type: searchForm.value.bundle_type,
      bt_start: searchForm.value.range[0], bt_end: searchForm.value.range[1],
      grid, n_trials: searchForm.value.n_trials > 0 ? searchForm.value.n_trials : null,
      objective: searchForm.value.objective,
    })
    searchJobId.value = data.id
  } finally { loadingSearch.value = false }
}

async function submitSens() {
  loadingSens.value = true
  try {
    const vals = _parseList(sensForm.value.values, sensForm.value.param === 'min_score' ? parseFloat : parseInt)
    const { data } = await optimizeApi.sensitivity({
      bundle_dir: sensForm.value.bundle_dir,
      bundle_type: 'per_cluster',
      bt_start: sensForm.value.range[0], bt_end: sensForm.value.range[1],
      param: sensForm.value.param, values: vals,
    })
    sensJobId.value = data.id
  } finally { loadingSens.value = false }
}
</script>
