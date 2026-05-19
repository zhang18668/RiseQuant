<template>
  <el-card>
    <template #header>
      <b>任务列表</b>
      <el-select v-model="kind" placeholder="全部" clearable style="float: right; width: 180px"
                 @change="load">
        <el-option v-for="k in ['fetch','clean','dryrun','train','backtest','optimize_search','optimize_sensitivity']" :key="k" :label="k" :value="k" />
      </el-select>
    </template>
    <el-table :data="jobs" size="small" @row-click="row => $router.push({ name: 'jobs', query: { id: row.id } })">
      <el-table-column prop="id" label="ID" width="120" />
      <el-table-column prop="kind" label="类型" width="140" />
      <el-table-column label="状态" width="100">
        <template #default="{ row }">
          <el-tag :type="stateType(row.state)" size="small">{{ row.state }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="created_at" label="提交时间" width="200" />
      <el-table-column prop="finished_at" label="完成时间" width="200" />
      <el-table-column label="目标">
        <template #default="{ row }">
          <span v-if="row.kind === 'train'">{{ row.params && row.params.start }} ~ {{ row.params && row.params.end }}</span>
          <span v-else-if="row.kind === 'backtest'">{{ row.params && row.params.bt_start }} ~ {{ row.params && row.params.bt_end }}</span>
          <span v-else-if="row.kind === 'dryrun'">{{ row.params && row.params.start }} ~ {{ row.params && row.params.end }}</span>
          <span v-else>{{ JSON.stringify(row.params).slice(0, 80) }}</span>
        </template>
      </el-table-column>
    </el-table>
    <JobWatcher v-if="$route.query.id" :job-id="$route.query.id" />
  </el-card>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { jobApi } from '../api.js'
import JobWatcher from '../components/JobWatcher.vue'

const jobs = ref([])
const kind = ref('')

function stateType(s) {
  return { pending: 'info', running: 'warning', done: 'success', failed: 'danger' }[s] || ''
}

async function load() {
  const { data } = await jobApi.list(kind.value || undefined)
  jobs.value = data
}

onMounted(load)
setInterval(load, 5000)
</script>
