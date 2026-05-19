<template>
  <div>
    <el-card>
      <template #header>
        <b>1) 数据获取（TDX 本地日线）</b>
      </template>
      <el-form :model="fetchForm" label-width="140px">
        <el-form-item label="TDX vipdoc 路径">
          <el-input v-model="fetchForm.tdx_path" placeholder="C:\new_tdx\vipdoc" />
        </el-form-item>
        <el-form-item label="起止日期">
          <el-date-picker
            v-model="fetchForm.range" type="daterange"
            value-format="YYYY-MM-DD" range-separator="至"
            start-placeholder="开始" end-placeholder="结束" />
        </el-form-item>
        <el-form-item label="仅主板">
          <el-switch v-model="fetchForm.main_board_only" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="loadingFetch" @click="submitFetch">提交</el-button>
          <span v-if="fetchJobId" style="margin-left: 12px; color: #888;">job: {{ fetchJobId }}</span>
        </el-form-item>
      </el-form>
      <JobWatcher :job-id="fetchJobId" />
    </el-card>

    <el-card style="margin-top: 16px;">
      <template #header>
        <b>2) 数据清洗</b>
      </template>
      <el-form :model="cleanForm" label-width="140px">
        <el-form-item label="父 fetch job id">
          <el-input v-model="cleanForm.job_id" placeholder="上一步生成" />
        </el-form-item>
        <el-form-item label="剔除 ST">
          <el-switch v-model="cleanForm.exclude_st" />
        </el-form-item>
        <el-form-item label="|涨跌幅| 上限 %">
          <el-input-number v-model="cleanForm.max_abs_change_pct" :min="10" :max="50" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="loadingClean" @click="submitClean">提交</el-button>
          <span v-if="cleanJobId" style="margin-left: 12px; color: #888;">job: {{ cleanJobId }}</span>
        </el-form-item>
      </el-form>
      <JobWatcher :job-id="cleanJobId" />
    </el-card>

    <el-card style="margin-top: 16px;">
      <template #header>
        <b>3) 数据 Dry-run（金标准事件数诊断）</b>
      </template>
      <el-form :model="dryForm" label-width="140px">
        <el-form-item label="起止日期">
          <el-date-picker
            v-model="dryForm.range" type="daterange"
            value-format="YYYY-MM-DD" />
        </el-form-item>
        <el-form-item label="目标金标准区间">
          <el-input-number v-model="dryForm.target_lo" :min="0.05" :max="0.5" :step="0.01" />
          ~
          <el-input-number v-model="dryForm.target_hi" :min="0.05" :max="1" :step="0.01" />
        </el-form-item>
        <el-form-item label="max_gap">
          <el-input-number v-model="dryForm.max_gap" :min="5" :max="60" />
        </el-form-item>
        <el-form-item label="min events">
          <el-input-number v-model="dryForm.min_events" :min="50" :max="2000" :step="50" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="loadingDry" @click="submitDryrun">提交</el-button>
          <span v-if="dryJobId" style="margin-left: 12px; color: #888;">job: {{ dryJobId }}</span>
        </el-form-item>
      </el-form>
      <JobWatcher :job-id="dryJobId" />
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { dataApi } from '../api.js'
import JobWatcher from '../components/JobWatcher.vue'

const fetchForm = ref({ tdx_path: 'C:\\new_tdx\\vipdoc', range: ['2022-01-01', '2024-12-31'], main_board_only: true })
const cleanForm = ref({ job_id: '', exclude_st: true, max_abs_change_pct: 30 })
const dryForm = ref({ range: ['2022-01-01', '2024-12-31'], target_lo: 0.15, target_hi: 0.35, max_gap: 30, min_events: 300 })

const fetchJobId = ref(''), cleanJobId = ref(''), dryJobId = ref('')
const loadingFetch = ref(false), loadingClean = ref(false), loadingDry = ref(false)

async function submitFetch() {
  loadingFetch.value = true
  try {
    const { data } = await dataApi.fetch({
      source: 'tdx', tdx_path: fetchForm.value.tdx_path,
      start: fetchForm.value.range[0], end: fetchForm.value.range[1],
      main_board_only: fetchForm.value.main_board_only,
    })
    fetchJobId.value = data.id
    cleanForm.value.job_id = data.id
  } finally { loadingFetch.value = false }
}

async function submitClean() {
  if (!cleanForm.value.job_id) return
  loadingClean.value = true
  try {
    const { data } = await dataApi.clean(cleanForm.value)
    cleanJobId.value = data.id
  } finally { loadingClean.value = false }
}

async function submitDryrun() {
  loadingDry.value = true
  try {
    const { data } = await dataApi.dryrun({
      start: dryForm.value.range[0], end: dryForm.value.range[1],
      tdx_path: fetchForm.value.tdx_path,
      target_lo: dryForm.value.target_lo, target_hi: dryForm.value.target_hi,
      max_gap: dryForm.value.max_gap, min_events: dryForm.value.min_events,
      golden_lo: [0.10, 0.15, 0.20], golden_hi: [0.40, 0.35, 0.30],
    })
    dryJobId.value = data.id
  } finally { loadingDry.value = false }
}
</script>
